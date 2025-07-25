"""
LaserCube network interface module.
Handles UDP communication with LaserCube laser projectors.
"""

import collections
import select
import socket
import struct
import threading
import time
from typing import Callable, Dict, Optional, Tuple, List


# LaserCube network ports
ALIVE_PORT = 45456
CMD_PORT = 45457
DATA_PORT = 45458

# LaserCube command IDs
CMD_GET_FULL_INFO = 0x77
CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA = 0x78
CMD_SET_OUTPUT = 0x80
CMD_GET_RINGBUFFER_EMPTY_SAMPLE_COUNT = 0x8a
CMD_SAMPLE_DATA = 0xa9


LaserInfo = collections.namedtuple('LaserInfo', [
    'model_name', 'fw_major', 'fw_minor', 'output_enabled',
    'dac_rate', 'max_dac_rate', 'rx_buffer_free', 'rx_buffer_size',
    'battery_percent', 'temperature', 'connection_type',
    'model_number', 'serial_number', 'ip_addr'
])


class LaserCube:
    """LaserCube laser projector interface."""
    
    def __init__(self, addr: str, gen_frame_fn: Callable, cmd_sock: socket.socket, data_sock: socket.socket):
        self.addr = addr
        self.gen_frame = gen_frame_fn
        self.info: Optional[LaserInfo] = None
        self.remote_buf_free = 0
        self.running = True
        self.cmd_sock = cmd_sock
        self.data_sock = data_sock
        threading.Thread(target=self._main_loop, daemon=True).start()

    def stop(self):
        """Stop the laser cube operation."""
        self.running = False

    def recv(self, msg: bytes):
        """Process incoming message from LaserCube."""
        cmd = msg[0]
        if cmd == CMD_GET_FULL_INFO:
            fields = struct.unpack('<xxBB?5xIIxHHBBB11xB26x', msg)
            serial = struct.unpack('6B', msg[26:32])
            ip = struct.unpack('4B', msg[32:36])
            name = msg[38:].split(b'\0', 1)[0].decode()
            info = LaserInfo(
                name, *fields,
                ':'.join(f"{b:02x}" for b in serial),
                '.'.join(str(b) for b in ip)
            )
            if info != self.info:
                self.info = info
                self.remote_buf_free = info.rx_buffer_free
        elif cmd == CMD_GET_RINGBUFFER_EMPTY_SAMPLE_COUNT:
            self.remote_buf_free = struct.unpack('<xxH', msg)[0]

    def _send_cmd(self, cmd_bytes: List[int]):
        """Send command to LaserCube."""
        cmd_data = bytes(cmd_bytes)
        self.cmd_sock.sendto(cmd_data, (self.addr, CMD_PORT))
        self.cmd_sock.sendto(cmd_data, (self.addr, CMD_PORT))

    def _main_loop(self):
        """Main laser control loop."""
        msg_num = 0
        frame_num = 0
        self._send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 1])
        self._send_cmd([CMD_SET_OUTPUT, 1])
        
        while self.running:
            pts = self.gen_frame()
            while pts:
                if self.remote_buf_free < 5000:
                    time.sleep(100/self.info.dac_rate if self.info else 0.001)
                    self.remote_buf_free += 100
                hdr = bytes([CMD_SAMPLE_DATA, 0, msg_num % 256, frame_num % 256])
                pkt = hdr + b''.join(pts[:140])
                for _ in pts[:140]:
                    self.remote_buf_free -= 1
                self.data_sock.sendto(pkt, (self.addr, DATA_PORT))
                msg_num += 1
                del pts[:140]
            frame_num += 1
            
        self._send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 0])
        self._send_cmd([CMD_SET_OUTPUT, 0])


class LaserController:
    """High-level LaserCube controller."""
    
    def __init__(self, frame_generator: Callable):
        self.frame_generator = frame_generator
        self.known_lasers: Dict[str, LaserCube] = {}
        self.running = True
        
        # Setup UDP sockets
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.cmd_sock.bind(("0.0.0.0", CMD_PORT))
        
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_sock.bind(("0.0.0.0", DATA_PORT))
        
        # Start scanner thread
        threading.Thread(target=self._scanner, daemon=True).start()
    
    def _scanner(self):
        """Scan for LaserCubes on the network."""
        while self.running:
            self.cmd_sock.sendto(bytes([CMD_GET_FULL_INFO]), ('255.255.255.255', CMD_PORT))
            time.sleep(1)
    
    def run(self):
        """Main controller loop."""
        try:
            while self.running:
                ready, _, _ = select.select([self.cmd_sock, self.data_sock], [], [])
                for sock in ready:
                    msg, (addr, _) = sock.recvfrom(4096)
                    if addr not in self.known_lasers:
                        if msg and msg[0] == CMD_GET_FULL_INFO:
                            self.known_lasers[addr] = LaserCube(
                                addr, self.frame_generator, self.cmd_sock, self.data_sock
                            )
                        else:
                            continue
                    lc = self.known_lasers[addr]
                    new = lc.info is None
                    lc.recv(msg)
                    if new and lc.info:
                        print("Found LaserCube:", lc.info)
        except KeyboardInterrupt:
            print("Stopping laser controller...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the laser controller."""
        self.running = False
        for lc in self.known_lasers.values():
            lc.stop()
        self.cmd_sock.close()
        self.data_sock.close()


def create_frame_from_bbox(bbox_data: Optional[Tuple[float, float, float, float]], 
                          screen_width: int, screen_height: int) -> List[bytes]:
    """Generate ILDA frame data from bounding box coordinates."""
    if bbox_data is None:
        # Center point when no detection
        pts = [(2048, 2048)]
    else:
        x1, y1, x2, y2 = bbox_data
        # Normalize to screen coordinates
        nx1, ny1 = x1 / screen_width, y1 / screen_height
        nx2, ny2 = x2 / screen_width, y2 / screen_height
        # Convert to laser coordinates (0-4095)
        lx1 = int((1 - nx1) * 4095)
        ly1 = int((1 - ny1) * 4095)
        lx2 = int((1 - nx2) * 4095)
        ly2 = int((1 - ny2) * 4095)
        # Create rectangle points
        pts = [(lx1, ly1), (lx2, ly1), (lx2, ly2), (lx1, ly2), (lx1, ly1)]
    
    frame = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        for i in range(200):
            xi = x0 + (x1 - x0) * i / 200
            yi = y0 + (y1 - y0) * i / 200
            # Pack as ILDA point: X, Y, R, G, B (all 16-bit)
            frame.append(struct.pack('<HHHHH', int(xi), int(yi), 4095, 4095, 4095))
    
    return frame