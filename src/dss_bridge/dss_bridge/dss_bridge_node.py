import sys
import asyncio
import cv2
from cv_bridge import CvBridge
import numpy as np
import json
import math
import struct

# DSS SDK 임포트
from dss_sdk.core.idsssdk import IDSSSDK
from dss_sdk.config.sdk_config import *
from dss_sdk.core.osi_manager import OSIManager
from dss_sdk.protobuf import dss_pb2

# ROS2 임포트
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header, Empty
from sensor_msgs.msg import NavSatFix, Image, Imu, CameraInfo
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy



# =============== 전역 변수 ===============
dss_sdk = None
ros_node = None
running = True

bridge = CvBridge() # 이미지 콜백 시 사용

# =============== DSS SDK 콜백 함수들 ===============
def on_camera_data(dss_image: dss_pb2.DSSImage):
    global ros_node

    try:
        if ros_node.first_camera_info_:
            ros_node._build_camera_info(dss_image)
            ros_node.first_camera_info_ = False

        height = dss_image.height
        width = dss_image.width

        image_data_array = np.ndarray(
            shape=(height, width, 4),
            dtype=np.uint8,
            buffer=dss_image.data
        )

        rgb_image = cv2.cvtColor(image_data_array, cv2.COLOR_BGRA2RGB)
        img_msg = bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        img_msg.header.stamp = ros_node.get_clock().now().to_msg()
        img_msg.header.frame_id = dss_image.header.frame_id

        cam_info = ros_node._camera_info
        cam_info.header = img_msg.header

        ros_node.camera_info_pub.publish(cam_info)
        ros_node.img_pub.publish(img_msg)

    except Exception as e:
        print(f"[ERROR] Camera processing: {e}")



def on_lidar_data(lidar_data, max_range=100.0):
    """LiDAR 데이터 처리"""
    global ros_node

    if not ros_node:
        return

    ros_node.get_logger().info(f"[LIDAR] Data received: {len(lidar_data)} bytes")
    lidar = dss_pb2.DssLidarPointCloud()
    lidar.ParseFromString(lidar_data)

    ros_msg = PointCloud2()
    ros_msg.header.stamp = ros_node.get_clock().now().to_msg()
    ros_msg.header.frame_id = lidar.frame_id
    ros_msg.height = 1  # point cloud는 일반적으로 1행
    ros_msg.width = lidar.width * lidar.height
    ros_msg.is_dense = True
    ros_msg.is_bigendian = False

    # 수동 필드 정의
    ros_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='ring', offset=16, datatype=PointField.FLOAT32, count=1),
    ]

    ros_msg.point_step = 20  # 5 x float32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width

    converted_data = bytearray()
    data = lidar.data
    step = lidar.point_step
    point_count = lidar.width * lidar.height

    for i in range(point_count):
        offset = i * step
        if offset + step > len(data):
            break

        ring = 0.0  # 기본값 설정

        try:
            if step == 16:
                x, y, z, intensity = struct.unpack_from("<ffff", data, offset)
            elif step == 8:
                xi, yi, zi, ii = struct.unpack_from("<hhhh", data, offset)
                x = (xi / 32767.0) * max_range
                y = (yi / 32767.0) * max_range
                z = (zi / 32767.0) * max_range
                intensity = (ii / 32767.0)
            elif step == 10:
                xi, yi, zi, ii, ri = struct.unpack_from("<hhhhh", data, offset)
                x = (xi / 32767.0) * max_range
                y = (yi / 32767.0) * max_range
                z = (zi / 32767.0) * max_range
                intensity = (ii / 32767.0)
                ring = float(ri)
            else:
                ros_node.get_logger().warn(f"[LIDAR] Unsupported point_step: {step}")
                continue
        except struct.error as e:
            ros_node.get_logger().warn(f"[LIDAR] unpack error at point {i}: {e}")
            continue

        converted_data.extend(struct.pack("<fffff", x, y, z, intensity, ring))

    ros_msg.data = bytes(converted_data)
    num_points = len(ros_msg.data) // ros_msg.point_step
    ros_node.get_logger().info(f"LiDAR: {num_points} points, step={step}")

    ros_node.lidar_pub.publish(ros_msg)



def on_imu_data(dss_imu: dss_pb2.DSSIMU):
    """IMU 데이터 처리 + ROS 퍼블리시"""
    global ros_node
    if not ros_node:
        return
        
    try:
        # DSSIMU → ROS 2 Imu 메시지 변환
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = ros_node.get_clock().now().to_msg()
        imu_msg.header.frame_id = dss_imu.header.frame_id

        # Orientation
        imu_msg.orientation.x = -dss_imu.orientation.x
        imu_msg.orientation.y = dss_imu.orientation.y
        imu_msg.orientation.z = -dss_imu.orientation.z
        imu_msg.orientation.w = -dss_imu.orientation.w

        # Orientation covariance
        if len(dss_imu.orientation_covariance) == 9:
            imu_msg.orientation_covariance = list(dss_imu.orientation_covariance)
        else:
            imu_msg.orientation_covariance[0] = -1.0

        # Angular velocity
        imu_msg.angular_velocity.x = -dss_imu.angular_velocity.x
        imu_msg.angular_velocity.y = dss_imu.angular_velocity.y
        imu_msg.angular_velocity.z = -dss_imu.angular_velocity.z

        # Angular velocity covariance
        if len(dss_imu.angular_velocity_covariance) == 9:
            imu_msg.angular_velocity_covariance = list(dss_imu.angular_velocity_covariance)
        else:
            imu_msg.angular_velocity_covariance[0] = -1.0

        # Linear acceleration
        imu_msg.linear_acceleration.x = dss_imu.linear_acceleration.x
        imu_msg.linear_acceleration.y = -dss_imu.linear_acceleration.y
        imu_msg.linear_acceleration.z = dss_imu.linear_acceleration.z

        # Linear acceleration covariance
        if len(dss_imu.linear_acceleration_covariance) == 9:
            imu_msg.linear_acceleration_covariance = list(dss_imu.linear_acceleration_covariance)
        else:
            imu_msg.linear_acceleration_covariance[0] = -1.0

        # ROS 토픽으로 퍼블리시
        ros_node.imu_pub.publish(imu_msg)
        ros_node.get_logger().info(f"[IMU] Orientation: x={dss_imu.orientation.x:.2f}, y={dss_imu.orientation.y:.2f}, z={dss_imu.orientation.z:.2f}")

    except Exception as e:
        ros_node.get_logger().error(f'[ERROR] Failed to convert DSSIMU: {e}')


def on_gps_data(dss_gps: dss_pb2.DSSGPS):
    """GPS 데이터 처리 + ROS 퍼블리시"""
    global ros_node
    if not ros_node:
        return
        
    try:
        # DSSGPS → ROS 2 NavSatFix 메시지 변환
        gps_msg = NavSatFix()
        gps_msg.header = Header()
        gps_msg.header.stamp = ros_node.get_clock().now().to_msg()
        gps_msg.header.frame_id = dss_gps.header.frame_id

        gps_msg.latitude = dss_gps.latitude
        gps_msg.longitude = dss_gps.longitude
        gps_msg.altitude = dss_gps.altitude

        # Position covariance
        if len(dss_gps.position_covariance) == 9:
            gps_msg.position_covariance = list(dss_gps.position_covariance)
            gps_msg.position_covariance_type = dss_gps.position_covariance_type
        else:
            gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        # ROS 토픽으로 퍼블리시
        ros_node.gps_pub.publish(gps_msg)
        ros_node.get_logger().info(f"[GPS] Location: lat={dss_gps.latitude:.6f}, lon={dss_gps.longitude:.6f}, alt={dss_gps.altitude:.2f}")

    except Exception as e:
        ros_node.get_logger().error(f'[ERROR] Failed to convert DSSGPS: {e}')


def on_odom_data(dss_odom: dss_pb2.DSSOdom):
    """Odometry 데이터 처리 + ROS 퍼블리시"""
    global ros_node
    if not ros_node:
        return
        
    try:
        # DSSOdom → ROS 2 Odometry 메시지 변환
        odom_msg = Odometry()
        odom_msg.header = Header()
        odom_msg.header.stamp = ros_node.get_clock().now().to_msg()
        odom_msg.header.frame_id = dss_odom.header.frame_id
        odom_msg.child_frame_id = dss_odom.child_frame_id

        # Position
        odom_msg.pose.pose.position.x = dss_odom.pose.position.x
        odom_msg.pose.pose.position.y = -dss_odom.pose.position.y
        odom_msg.pose.pose.position.z = dss_odom.pose.position.z

        # Orientation
        odom_msg.pose.pose.orientation.x = -dss_odom.pose.orientation.x
        odom_msg.pose.pose.orientation.y = dss_odom.pose.orientation.y
        odom_msg.pose.pose.orientation.z = -dss_odom.pose.orientation.z
        odom_msg.pose.pose.orientation.w = -dss_odom.pose.orientation.w

        # Pose covariance
        if len(dss_odom.pose_covariance) == 36:
            odom_msg.pose.covariance = list(dss_odom.pose_covariance)
        else:
            odom_msg.pose.covariance[0] = -1.0

        # Twist (velocity)
        odom_msg.twist.twist.linear.x = dss_odom.twist.linear.x
        odom_msg.twist.twist.linear.y = -dss_odom.twist.linear.y
        odom_msg.twist.twist.linear.z = dss_odom.twist.linear.z

        odom_msg.twist.twist.angular.x = dss_odom.twist.angular.x
        odom_msg.twist.twist.angular.y = -dss_odom.twist.angular.y
        odom_msg.twist.twist.angular.z = -dss_odom.twist.angular.z

        # Twist covariance
        if len(dss_odom.twist_covariance) == 36:
            odom_msg.twist.covariance = list(dss_odom.twist_covariance)
        else:
            odom_msg.twist.covariance[0] = -1.0

        # ROS 토픽으로 퍼블리시
        ros_node.odom_pub.publish(odom_msg)
        pos = dss_odom.pose.position
        ros_node.get_logger().info(f"[ODOM] Position: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}")

    except Exception as e:
        ros_node.get_logger().error(f'[ERROR] Failed to convert DSSOdom: {e}')


def on_ground_truth_data(gt_data):
    """Ground Truth 데이터 처리"""
    global ros_node
    try:
        # OSI Manager를 사용해서 GT 데이터 파싱
        gt = OSIManager.parse_ground_truth(gt_data)
        if gt and ros_node:
            ros_node.get_logger().info(f"[GT] timestamp: {gt.timestamp.seconds}")
            ros_node.get_logger().info(f"[GT] host car ID: {gt.host_vehicle_id.value}")
            
            # 이동 객체 (차량, 보행자 등)
            if gt.moving_object:
                ros_node.get_logger().info(f"[GT] moving object: {len(gt.moving_object)}")
                for obj in gt.moving_object:
                    pos = obj.base.position
                    ros_node.get_logger().info(f"  - object ID: {obj.id.value}, location: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
            
            # 정적 객체 (건물, 표지판 등)
            if gt.stationary_object:
                ros_node.get_logger().info(f"[GT] static objects: {len(gt.stationary_object)}")
            
            # 교통 신호등
            if gt.traffic_light:
                ros_node.get_logger().info(f"[GT] traffic_light: {len(gt.traffic_light)}")
                for light in gt.traffic_light:
                    ros_node.get_logger().info(f"  - traffic_light status: {light.classification.mode}")
            
            # 차선 정보
            if gt.lane:
                ros_node.get_logger().info(f"[GT] lane: {len(gt.lane)}")
                
    except Exception as e:
        if ros_node:
            ros_node.get_logger().error(f"[GT] parsing error: {e}")


# =============== 시뮬레이션 상태 콜백들 ===============

def on_sim_started():
    global ros_node
    if ros_node:
        ros_node.get_logger().info("[SIM] Simulation started!")


def on_sim_ended():
    global ros_node
    if ros_node:
        ros_node.get_logger().info("[SIM] Simulation ended!")


def on_sim_aborted():
    global ros_node
    if ros_node:
        ros_node.get_logger().warn("[SIM] Simulation aborted!")


def on_sim_error():
    global ros_node
    if ros_node:
        ros_node.get_logger().error("[SIM] Simulation error!")


# =============== ROS 노드 클래스 ===============

class DSSBridgeNode(Node):
    def __init__(self, nats_ip, sim_ip, heartbeat_port, nats_port):
        super().__init__('dss_bridge_node')

        # 파라미터 값 가져오기
        self.nats_ip = nats_ip
        self.sim_ip = sim_ip
        self.heartbeat_port = heartbeat_port
        self.nats_port = nats_port

        # QoS 프로파일 설정
        self.setup_qos_profiles()

        # ROS2 Publishers
        self.img_pub = self.create_publisher(Image, '/dss/image', 1)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/dss/camera_info', 1)
        self.gps_pub = self.create_publisher(NavSatFix, '/dss/gps', 1)
        self.imu_pub = self.create_publisher(Imu, '/dss/imu', 1)
        self.odom_pub = self.create_publisher(Odometry, '/dss/odom', 1)
        self.lidar_pub = self.create_publisher(PointCloud2, '/dss/lidar', 10)
        self.sim_gt_json = self.create_publisher(String, '/dss/simulation_gt', 10)

        self.sim_started_pub = self.create_publisher(Empty, '/dss/simulation_started', self.reliable_qos)
        self.sim_ended_pub = self.create_publisher(Empty, '/dss/simulation_ended', self.reliable_qos)
        self.sim_aborted_pub = self.create_publisher(Empty, '/dss/simulation_aborted', 10)
        self.sim_error_pub = self.create_publisher(Empty, '/dss/simulation_error', 10)
        
        # ROS2 Subscriber (제어 명령 수신)
        self.create_subscription(String, '/dss/SetControl', self.set_control_callback, 1)
        
        # DSS SDK 초기화
        self.setup_dss_sdk()
        
        self.get_logger().info('DSS Bridge Node started successfully')
        self.first_camera_info_ = True

    def _build_camera_info(self, camera_actor):
        """Build camera info."""
        camera_info = CameraInfo()
        camera_info.width = camera_actor.width
        camera_info.height = camera_actor.height
        camera_info.distortion_model = "plumb_bob"
        cx = camera_info.width / 2.0
        cy = camera_info.height / 2.0
        fx = camera_info.width / (2.0 * math.tan(camera_actor.fov * math.pi / 360.0))
        fy = fx
        camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        self._camera_info = camera_info

    def setup_qos_profiles(self):
        """센서별 최적화된 QoS 프로파일 설정"""
        
        # 카메라용: 최신 이미지만 중요
        self.camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  
            durability=DurabilityPolicy.VOLATILE,       
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 라이다용: 큰 데이터
        self.large_data_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=3 
        )
        
        # 고주파 센서용 (IMU): 높은 처리량
        self.high_freq_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 일반 센서용 (GPS, Odom)
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # 제어 명령용
        self.control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 시뮬레이션 상태용
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

    def setup_dss_sdk(self):
        """DSS SDK 초기화 및 콜백 등록"""
        global dss_sdk
        
        # NATS 이벤트 루프 생성
        nats_loop = asyncio.new_event_loop()
        
        # DSS SDK 인스턴스 생성
        dss_sdk = IDSSSDK.create(
            loop=nats_loop,
            nats_address=f'nats://{self.nats_ip}:{self.nats_port}'
        )
        
        # 초기화 파라미터 설정
        init_params = DSSSDKInitParams(
            server=self.sim_ip,
            heartbeat_port=self.heartbeat_port,
            nats_port=self.nats_port
        )

        # SDK 초기화
        dss_sdk.initialize(init_params)
        
        # 센서 콜백 등록
        dss_sdk.register_sensor_callback('camera', on_camera_data)
        dss_sdk.register_sensor_callback('lidar', on_lidar_data)
        dss_sdk.register_sensor_callback('imu', on_imu_data)
        dss_sdk.register_sensor_callback('gps', on_gps_data)
        dss_sdk.register_sensor_callback('odom', on_odom_data)
        dss_sdk.register_sensor_callback('ground_truth', on_ground_truth_data)
        
        # 시뮬레이션 상태 콜백 등록
        dss_sdk.register_simulation_callback('started', on_sim_started)
        dss_sdk.register_simulation_callback('ended', on_sim_ended)
        dss_sdk.register_simulation_callback('aborted', on_sim_aborted)
        dss_sdk.register_simulation_callback('error', on_sim_error)
        
        self.get_logger().info("[DSS SDK] All callbacks registered successfully")
        
        # SDK 시작
        dss_sdk.start()
        
        self.get_logger().info('DSS SDK initialized and started')

    def set_control_callback(self, msg):
        """ROS에서 제어 명령 수신 → DSS SDK로 전송"""
        global dss_sdk
        if not dss_sdk:
            self.get_logger().error("DSS SDK not initialized")
            return
            
        self.get_logger().info(f'Received /SetControl: {msg.data}')
        
        try:
            data = json.loads(msg.data)
            
            # DSS 제어 객체 생성
            control = DSSSDKCarControl(
                steer=data.get("steer", 0.0),
                throttle=data.get("throttle", 0.0),
                brake=data.get("brake", 0.0),
                park_brake=data.get("parkBrake", False),
                target_gear=data.get("targetGear", 0),
                head_light=data.get("headLight", False),
                tail_light=data.get("tailLight", False),
                turn_signal=data.get("turnSignal", 0),
                horn=data.get("horn", False),
                light_mode=data.get("lightMode", 0),
                wiper_mode=data.get("wiperMode", 0),
            )
            
            # 차량 제어 명령 전송
            dss_sdk.set_car_control(control)
            self.get_logger().info(f'Control sent: steer={control.steer:.2f}, throttle={control.throttle:.2f}, brake={control.brake:.2f}')
            
        except Exception as e:
            self.get_logger().error(f"Failed to parse control message: {e}")

    def __del__(self):
        global dss_sdk
        if dss_sdk:
            dss_sdk.cleanup()


# =============== 메인 함수  ===============

def main(args=None):
    global ros_node, running
    
    nats_ip = '172.28.176.1'
    sim_ip = '172.28.176.1'
    heartbeat_port = 8886
    nats_port = 4222

    # ROS2 초기화
    rclpy.init(args=args)
    
    try:
        ros_node = DSSBridgeNode(nats_ip, sim_ip, heartbeat_port, nats_port)
        rclpy.spin(ros_node)
        
    except KeyboardInterrupt:
        print("\n[DSS SDK] User interrupted")
        running = False
        
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        running = False
        
    finally:
        global dss_sdk
        if dss_sdk:
            dss_sdk.cleanup()
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()
        print("[DSS SDK] Cleanup completed")


if __name__ == '__main__':
    main()