import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import signal

class SpinTestNode(Node):
    def __init__(self):
        super().__init__('spin_test_node')
        # /dss/SetControl 토픽 퍼블리셔
        self.control_pub = self.create_publisher(String, '/dss/SetControl', 10)
        
        self.timer = self.create_timer(1.0 / 60.0, self.update_control)
        self.get_logger().info('SpinTestNode started, publishing spin commands to /dss/SetControl')

    def update_control(self):
        """원형 회전용 고정 제어 명령 퍼블리시"""
        steer = 0.7    # 오른쪽으로 고정 회전
        throttle = 0.3 # 일정 속도 전진
        brake = 0.0

        self.current_throttle = throttle
        self.current_steer = steer

        payload = {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "parkBrake": False,
            "targetGear": 1 if self.current_throttle >= 0 else -1,
            "headLight": True,
            "tailLight": False,
            "turnSignal": 1 if self.current_steer < -0.1 else (2 if self.current_steer > 0.1 else 0),
            "horn": False,
            "lightMode": 0,
            "wiperMode": 0
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.control_pub.publish(msg)
        self.get_logger().info(f"[CONTROL] {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = SpinTestNode()

    # SIGINT 처리
    signal.signal(signal.SIGINT, lambda sig, frame: rclpy.shutdown())

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()