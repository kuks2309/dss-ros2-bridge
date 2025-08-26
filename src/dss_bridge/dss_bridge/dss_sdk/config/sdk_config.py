class DSSSDKInitParams:
    def __init__(self, server: str, heartbeat_port: int, nats_port: int):
        self.server = server.encode('utf-8')
        self.heartbeat_port = heartbeat_port
        self.nats_port = nats_port


class DSSSDKCarControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, 
                 park_brake=False, target_gear=0, head_light=False,
                 tail_light=False, turn_signal=0, horn=False, light_mode=0, wiper_mode=0):
        # 기본 제어
        self.throttle = throttle
        self.steer = steer
        self.brake = brake
        self.park_brake = park_brake
        self.target_gear = target_gear
        
        # 조명 및 신호
        self.head_light = head_light
        self.tail_light = tail_light
        self.turn_signal = turn_signal
        self.horn = horn
        self.light_mode = light_mode
        self.wiper_mode = wiper_mode

