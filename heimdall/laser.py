"""LaserCube network communication utilities."""

import collections
import socket
import struct
import threading
import time

ALIVE_PORT = 45456
CMD_PORT = 45457
DATA_PORT = 45458

CMD_GET_FULL_INFO = 0x77
CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA = 0x78
CMD_SET_OUTPUT = 0x80
CMD_SAMPLE_DATA = 0xA9
CMD_GET_RINGBUFFER_EMPTY_SAMPLE_COUNT = 0x8A

cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
cmd_sock.bind(("0.0.0.0", CMD_PORT))

data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
data_sock.bind(("0.0.0.0", DATA_PORT))

LaserInfo = collections.namedtuple(
    "LaserInfo",
    [
        "model_name",
        "fw_major",
        "fw_minor",
        "output_enabled",
        "dac_rate",
        "max_dac_rate",
        "rx_buffer_free",
        "rx_buffer_size",
        "battery_percent",
        "temperature",
        "connection_type",
        "model_number",
        "serial_number",
        "ip_addr",
    ],
)

known_lasers = {}


class LaserCube:
    """Minimal wrapper for a LaserCube device."""

    def __init__(self, addr, gen_frame):
        self.addr = addr
        self.gen_frame = gen_frame
        self.info = None
        self.remote_buf_free = 0
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False

    def recv(self, msg):
        cmd = msg[0]
        if cmd == CMD_GET_FULL_INFO and len(msg) > 1:
            fields = struct.unpack("<xxBB?5xIIxHHBBB11xB26x", msg[:50])
            serial = struct.unpack("6B", msg[26:32])
            ip = struct.unpack("4B", msg[32:36])
            name = msg[38:].split(b"\0", 1)[0].decode()
            info = LaserInfo(
                name,
                *fields,
                ":".join(f"{b:02x}" for b in serial),
                ".".join(str(b) for b in ip),
            )
            if info != self.info:
                self.info = info
                self.remote_buf_free = info.rx_buffer_free
        elif cmd == CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA:
            pass
        elif cmd == CMD_GET_RINGBUFFER_EMPTY_SAMPLE_COUNT:
            self.remote_buf_free = struct.unpack("<xxH", msg)[0]

    def send_cmd(self, cmd_bytes):
        for _ in range(2):
            cmd_sock.sendto(bytes(cmd_bytes), (self.addr, CMD_PORT))

    def _run(self):
        msg_num = 0
        frame_num = 0
        self.send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 1])
        self.send_cmd([CMD_SET_OUTPUT, 1])
        while self.running:
            pts = self.gen_frame()[:]
            while pts:
                if self.remote_buf_free < 5000:
                    time.sleep(0.1)
                    self.remote_buf_free += 100
                hdr = bytes([CMD_SAMPLE_DATA, 0x00, msg_num % 256, frame_num % 256])
                payload = b"".join(pts[:140])
                pts = pts[140:]
                self.remote_buf_free -= len(payload) // 10
                data_sock.sendto(hdr + payload, (self.addr, DATA_PORT))
                msg_num += 1
            frame_num += 1
        self.send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 0])
        self.send_cmd([CMD_SET_OUTPUT, 0])


def _scanner():
    while True:
        cmd_sock.sendto(bytes([CMD_GET_FULL_INFO]), ("255.255.255.255", CMD_PORT))
        time.sleep(1)


def start_scanner():
    threading.Thread(target=_scanner, daemon=True).start()
