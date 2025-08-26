import socket
import threading

from ..protobuf import dss_pb2

class SimControlSender:
    def __init__(self, target_ip='127.0.0.1', target_port=4222):
        self.target = (target_ip, target_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.latest_msg = None
        self.lock = threading.Lock()

    def update_message(self, proto_msg: dss_pb2.DssSetControl):
        with self.lock:
            self.latest_msg = proto_msg

    def send_latest(self):
        with self.lock:
            if self.latest_msg:
                try:
                    data = self.latest_msg.SerializeToString()
                    self.sock.sendto(data, self.target)
                except Exception as e:
                    print(f"UDP send error: {e}")