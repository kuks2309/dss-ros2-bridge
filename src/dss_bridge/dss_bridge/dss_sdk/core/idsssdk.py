import asyncio
import threading
import json
import time
import uuid
import socket
from typing import Dict, Callable, Any

from nats.aio.client import Client as NATS

from ..config.sdk_config import DSSSDKInitParams, DSSSDKCarControl
from .sim_control_sender import SimControlSender
from ..protobuf import dss_pb2
from ..core.osi_manager import OSIManager

from ..utils.logger import logger


class IDSSSDK:
    def __init__(self, loop, 
                 nats_address='nats://127.0.0.1:4222',
                 **kwargs):
        self.nats_address = nats_address
        self.nats_loop = loop
        self.nats_client = NATS()

        self.guid = str(uuid.uuid4())
        self.subscribers_ready = threading.Event()

        osi_version = OSIManager.get_osi_version()
        logger.sdk(f"OSI version: {osi_version}")
        
        # 사용자 콜백 저장소
        self._sensor_callbacks: Dict[str, Callable] = {}
        self._simulation_callbacks: Dict[str, Callable] = {}
        
        # 시뮬레이션 상태
        self._is_connected = False
        self._is_playing = False
        
        # UDP 제어
        self._control_sender = None
        self._control_timer = None
        
        # 자동 heartbeat 관리
        self._heartbeat_timer = None
        self._running = False
        
        # NATS 워커 스레드 시작
        threading.Thread(target=self.nats_worker, daemon=True).start()


    @classmethod
    def create(cls, loop, 
               nats_address='nats://127.0.0.1:4222',
               **kwargs):
        logger.sdk("Creating IDSSSDK instance...")
        return cls(loop, nats_address, **kwargs)
    

    def nats_worker(self):
        asyncio.set_event_loop(self.nats_loop)
        self.nats_loop.run_until_complete(self.nats_connect())
        self.nats_loop.run_forever()


    async def nats_connect(self):
        try:
            await self.nats_client.connect(self.nats_address)
            logger.sdk("Connected to NATS")
            self._is_connected = True

            # Subscribe to NATS topics
            await self.nats_client.subscribe("dss.adminConsole.heartBeat", cb=self.handle_admin_heartbeat)
            await self.nats_client.subscribe("dss.scenarioPlayer.heartBeat", cb=self.handle_scenarioPlayer_heartbeat)
            await self.nats_client.subscribe("dss.simulation.heartBeat", cb=self.handle_simulation_heartbeat)
            
            # 시나리오 run 성공하면 날라옴
            await self.nats_client.subscribe("dss.simulation.started", cb=self.handle_simulation_started)
            await self.nats_client.subscribe("dss.simulation.ended", cb=self.handle_simulation_ended)
            await self.nats_client.subscribe("dss.simulation.aborted", cb=self.handle_simulation_aborted)
            await self.nats_client.subscribe("dss.simulation.error", cb=self.handle_simulation_error)

            # GT = OSI 메시지 형태로 보냄
            await self.nats_client.subscribe("dss.simulation.groundTruth", cb=self.handle_simulation_groundTruth)
            # GT = JSON 메시지 형태로 보냄
            await self.nats_client.subscribe("dss.simulation.groundTruth.json", cb=self.handle_simulation_groundTruth_json)

            # Sensors
            await self.nats_client.subscribe("dss.sensor.camera.rgb", cb=self.handle_sensor_camera_rgb)
            await self.nats_client.subscribe("dss.sensor.lidar", cb=self.handle_sensor_lidar)
            await self.nats_client.subscribe("dss.sensor.imu", cb=self.handle_sensor_imu)
            await self.nats_client.subscribe("dss.sensor.gps", cb=self.handle_sensor_gps)
            await self.nats_client.subscribe("dss.odom", cb=self.handle_odom)

            self.subscribers_ready.set()

        except Exception as e:
            print(f"Failed to connect: {e}")
            self._is_connected = False


    @property
    def is_connected(self):
        return self.nats_client and self.nats_client.is_connected
    

    def initialize(self, params: DSSSDKInitParams):
        """SDK 초기화"""
        logger.sdk("SDK initialized with NATS communication")
        
        # UDP 제어 설정
        if hasattr(params, 'server'):
            logger.sdk(f"params.server: {params.server}")
            server_ip = params.server.decode('utf-8') if isinstance(params.server, bytes) else params.server
            self._setup_udp_control(server_ip, params.heartbeat_port)


    def _setup_udp_control(self, target_ip='127.0.0.1', target_port=4222):
        """UDP 제어 설정"""
        self._control_sender = SimControlSender(target_ip, target_port)
        
        dummy = dss_pb2.DssSetControl(
            identifier='dss_sdk_py', 
            timestamp=int(time.time() * 1000), 
            brake=1.0
        )
        self._control_sender.update_message(dummy)


    def start(self):
        """SDK 시작"""
        if self._running:
            logger.warning("SDK already running")
            return
        
        self._running = True
        
        # NATS 연결 대기
        logger.sdk("Waiting for NATS connection...")
        if not self.subscribers_ready.wait(timeout=10):
            raise RuntimeError("Failed to connect to NATS within 10 seconds")
        
        # 자동 heartbeat 시작 (1초 간격)
        self._start_heartbeat_timer()
        
        # 자동 제어 시작 (60Hz)
        self._start_control_timer()
        
        logger.sdk("DSS SDK started successfully")


    def _start_heartbeat_timer(self):
        """Heartbeat 타이머 시작 (1초 간격)"""
        def heartbeat_loop():
            while self._running:
                try:
                    self.send_heartbeat()
                    time.sleep(1.0)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    time.sleep(1.0)
        
        self._heartbeat_timer = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_timer.start()


    def _start_control_timer(self):
        """제어 타이머 시작 (60Hz)"""
        def control_loop():
            while self._running:
                try:
                    if self._control_sender:
                        self._control_sender.send_latest()
                    time.sleep(1.0 / 60.0)  # 60Hz
                except Exception as e:
                    logger.error(f"Control error: {e}")
                    time.sleep(0.1)
        
        self._control_timer = threading.Thread(target=control_loop, daemon=True)
        self._control_timer.start()


    def send_heartbeat(self):
        """Heartbeat 전송"""
        if self.is_connected:
            heartbeat = {
                "gameId": self.guid,
                "name": "dss_sdk_py"
            }
            msg = json.dumps(heartbeat)
            future = asyncio.run_coroutine_threadsafe(
                self.nats_client.publish("dss.sdk.heartBeat", msg.encode()),
                self.nats_loop
            )
            try:
                future.result(timeout=2)
                print("Heartbeat sent")
            except Exception as e:
                print(f"Failed to send heartbeat: {e}")

    # =============== 사용자 콜백 API ===============
    
    def register_sensor_callback(self, sensor_type: str, callback_fn: Callable):
        """센서 콜백 등록"""
        valid_sensors = ['camera', 'lidar', 'imu', 'gps', 'odom', 'ground_truth', 'ground_truth_json']
        if sensor_type not in valid_sensors:
            raise ValueError(f"Invalid sensor type. Must be one of: {valid_sensors}")
        
        self._sensor_callbacks[sensor_type] = callback_fn
        logger.sdk(f"Sensor callback registered: {sensor_type}")


    def register_simulation_callback(self, event_type: str, callback_fn: Callable):
        """시뮬레이션 상태 콜백 등록"""
        valid_events = ['started', 'ended', 'aborted', 'error']
        if event_type not in valid_events:
            raise ValueError(f"Invalid event type. Must be one of: {valid_events}")
        
        self._simulation_callbacks[event_type] = callback_fn
        logger.sdk(f"Simulation callback registered: {event_type}")


    def set_car_control(self, control: DSSSDKCarControl):
        """차량 제어 설정"""
        if self._control_sender:
            control_msg = dss_pb2.DssSetControl(
                identifier=self.guid,
                timestamp=int(time.time() * 1000),
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                parkBrake=control.park_brake,
                targetGear=control.target_gear,
                headLight=control.head_light,       
                tailLight=control.tail_light,    
                turnSignal=control.turn_signal,   
                horn=control.horn,  
                lightMode=control.light_mode,    
                wiperMode=control.wiper_mode, 
            )
            self._control_sender.update_message(control_msg)


    def _safe_callback_call(self, callback_fn: Callable, *args, **kwargs):
        """안전한 콜백 호출"""
        try:
            callback_fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    # =============== Message Handlers ===============

    async def handle_admin_heartbeat(self, msg):
        try:
            data = json.loads(msg.data.decode())
            logger.sdk(f"Received admin heartbeat: {data}")
        except Exception as e:
            logger.sdk(f"Failed to handle admin heartbeat: {e}")


    async def handle_scenarioPlayer_heartbeat(self, msg):
        try:
            data = json.loads(msg.data.decode())
            logger.sdk(f"Received scenarioPlayer heartbeat: {data}")
        except Exception as e:
            logger.sdk(f"Failed to handle scenarioPlayer heartbeat: {e}")


    async def handle_simulation_heartbeat(self, msg):
        try:
            data = json.loads(msg.data.decode())
            logger.sdk(f"Received simulation heartbeat: {data}")
        except Exception as e:
            logger.sdk(f"Failed to handle simulation heartbeat: {e}")


    async def handle_simulation_started(self, msg):
        logger.sdk("Simulation started message received")
        self._is_playing = True
        if 'started' in self._simulation_callbacks:
            self._safe_callback_call(self._simulation_callbacks['started'])


    async def handle_simulation_ended(self, msg):
        logger.sdk("Simulation ended message received")
        self._is_playing = False
        if 'ended' in self._simulation_callbacks:
            self._safe_callback_call(self._simulation_callbacks['ended'])


    async def handle_simulation_error(self, msg):
        logger.sdk("Simulation error message received")
        self._is_playing = False
        if 'error' in self._simulation_callbacks:
            self._safe_callback_call(self._simulation_callbacks['error'])


    async def handle_simulation_aborted(self, msg):
        logger.sdk("Simulation aborted message received")
        self._is_playing = False
        if 'aborted' in self._simulation_callbacks:
            self._safe_callback_call(self._simulation_callbacks['aborted'])


    async def handle_simulation_groundTruth(self, msg):
        if 'ground_truth' in self._sensor_callbacks:
            self._safe_callback_call(self._sensor_callbacks['ground_truth'], msg.data)


    async def handle_simulation_groundTruth_json(self, msg):
        if 'ground_truth_json' in self._sensor_callbacks:
            self._safe_callback_call(self._sensor_callbacks['ground_truth_json'], msg.data)


    async def handle_sensor_camera_rgb(self, msg):
        try:
            dss_image = dss_pb2.DSSImage()
            dss_image.ParseFromString(msg.data)
            
            if 'camera' in self._sensor_callbacks:
                self._safe_callback_call(self._sensor_callbacks['camera'], dss_image)
        except Exception as e:
            logger.sdk(f'Exception during camera processing: {e}')


    async def handle_sensor_lidar(self, msg):
        logger.sdk("LiDAR data received")
        if 'lidar' in self._sensor_callbacks:
            self._safe_callback_call(self._sensor_callbacks['lidar'], msg.data)


    async def handle_sensor_imu(self, msg):
        try:
            dss_imu = dss_pb2.DSSIMU()
            dss_imu.ParseFromString(msg.data)
            
            if 'imu' in self._sensor_callbacks:
                self._safe_callback_call(self._sensor_callbacks['imu'], dss_imu)
        except Exception as e:
            logger.sdk(f'Failed to convert DSSIMU: {e}')


    async def handle_sensor_gps(self, msg):
        try:
            dss_gps = dss_pb2.DSSGPS()
            dss_gps.ParseFromString(msg.data)
            
            if 'gps' in self._sensor_callbacks:
                self._safe_callback_call(self._sensor_callbacks['gps'], dss_gps)
        except Exception as e:
            logger.sdk(f'Failed to convert DSSGPS: {e}')


    async def handle_odom(self, msg):
        try:
            logger.sdk("handle_odom data received")
            dss_odom = dss_pb2.DSSOdom()
            dss_odom.ParseFromString(msg.data)
            
            if 'odom' in self._sensor_callbacks:
                self._safe_callback_call(self._sensor_callbacks['odom'], dss_odom)
        except Exception as e:
            logger.sdk(f'Failed to convert DSSOdom: {e}')


    def cleanup(self):
        """리소스 정리"""
        self._running = False
        logger.sdk("Cleaning up DSS SDK...")
        
        # 타이머 정리
        if self._heartbeat_timer and self._heartbeat_timer.is_alive():
            self._heartbeat_timer.join(timeout=2)
        if self._control_timer and self._control_timer.is_alive():
            self._control_timer.join(timeout=2)
        
        # NATS 연결 정리
        if self.nats_client and self.nats_client.is_connected:
            future = asyncio.run_coroutine_threadsafe(
                self.nats_client.close(),
                self.nats_loop
            )
            try:
                future.result(timeout=2)
            except Exception as e:
                logger.error(f"Failed to close NATS connection: {e}")
        
        logger.sdk("Cleanup completed")
