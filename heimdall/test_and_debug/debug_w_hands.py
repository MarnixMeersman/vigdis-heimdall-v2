# hand_laser_projection.py
# Real-time hand landmark projection via LaserCube with stable GUI and backpressure handling

# Requirements:
# pip install opencv-python mediapipe

import collections
import select
import socket
import struct
import threading
import time

import cv2
import mediapipe as mp
import pyautogui

# ---------------- LaserCube Setup ----------------
ALIVE_PORT = 45456
CMD_PORT = 45457
DATA_PORT = 45458

CMD_GET_FULL_INFO = 0x77
CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA = 0x78
CMD_SET_OUTPUT = 0x80
CMD_GET_RINGBUFFER_EMPTY_SAMPLE_COUNT = 0x8a
CMD_SAMPLE_DATA = 0xa9

LaserInfo = collections.namedtuple('LaserInfo', [
    'model_name','fw_major','fw_minor','output_enabled',
    'dac_rate','max_dac_rate','rx_buffer_free','rx_buffer_size',
    'battery_percent','temperature','connection_type',
    'model_number','serial_number','ip_addr'
])

known_lasers = {}

# Shared state
landmark_lock = threading.Lock()
current_landmarks = []   # latest landmarks
frame_lock = threading.Lock()
debug_frame = None       # latest frame for GUI

# ---------------- UDP Sockets ----------------
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
cmd_sock.bind(('0.0.0.0', CMD_PORT))

data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
data_sock.bind(('0.0.0.0', DATA_PORT))
# Optional: set non-blocking to catch EAGAIN
# data_sock.setblocking(False)

# ---------------- LaserCube Class ----------------
class LaserCube:
    def __init__(self, addr, gen_frame_fn):
        self.addr = addr
        self.gen_frame = gen_frame_fn
        self.info = None
        self.remote_buf_free = 0
        self.running = True
        threading.Thread(target=self._main_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def recv(self, msg):
        cmd = msg[0]
        # Full info response
        if cmd == CMD_GET_FULL_INFO and len(msg) >= 64:
            try:
                fields = struct.unpack('<xxBB?5xIIxHHBBB11xB26x', msg[:64])
                serial = struct.unpack('6B', msg[26:32])
                ip = struct.unpack('4B', msg[32:36])
                name = msg[38:64].split(b'\0',1)[0].decode()
                info = LaserInfo(
                    name, *fields,
                    ':'.join(f"{b:02x}" for b in serial),
                    '.'.join(str(b) for b in ip)
                )
                if info != self.info:
                    self.info = info
                    self.remote_buf_free = info.rx_buffer_free
            except struct.error:
                pass
        # Buffer size response
        elif cmd == CMD_GET_RINGBUFFER_EMPTY_SAMPLE_COUNT and len(msg) >= 4:
            try:
                self.remote_buf_free = struct.unpack('<xxH', msg[:4])[0]
            except struct.error:
                pass

    def _send_cmd(self, cmd_bytes):
        cmd_sock.sendto(bytes(cmd_bytes), (self.addr, CMD_PORT))
        cmd_sock.sendto(bytes(cmd_bytes), (self.addr, CMD_PORT))

    def _main_loop(self):
        # enable output
        self._send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA,1])
        self._send_cmd([CMD_SET_OUTPUT,1])
        # wait for info
        while self.info is None and self.running:
            time.sleep(0.1)
        msg_num = 0
        frame_num = 0
        while self.running:
            points = self.gen_frame()
            # send in chunks
            for i in range(0, len(points), 140):
                if not self.running:
                    break
                # throttle if needed
                if self.info is None or self.remote_buf_free < 5000:
                    time.sleep(100 / (self.info.dac_rate if self.info else 20000))
                    self.remote_buf_free += 100
                chunk = points[i:i+140]
                pkt = bytes([CMD_SAMPLE_DATA, 0, msg_num % 256, frame_num % 256]) + b''.join(chunk)
                # send safely
                try:
                    data_sock.sendto(pkt, (self.addr, DATA_PORT))
                except OSError:
                    # buffer full, skip this chunk
                    continue
                # update buffer count
                self.remote_buf_free -= len(chunk)
                msg_num += 1
            frame_num += 1
        # disable
        self._send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA,0])
        self._send_cmd([CMD_SET_OUTPUT,0])

# ---------------- Hand Detection (background) ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Starting hand detection. Press ESC in preview to exit.")

def detection_loop():
    global current_landmarks, debug_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        landmarks = []
        if res.multi_hand_landmarks:
            for lmset in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lmset, mp_hands.HAND_CONNECTIONS)
                for lm in lmset.landmark:
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
        with landmark_lock:
            current_landmarks = landmarks.copy()
        with frame_lock:
            debug_frame = frame.copy()

def detection_gui_loop():
    # separate GUI thread to ensure imshow in main context
    while True:
        with frame_lock:
            f = debug_frame.copy() if debug_frame is not None else None
        if f is not None:
            cv2.imshow('Hand Debug', f)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        time.sleep(0.01)

threading.Thread(target=detection_loop, daemon=True).start()
threading.Thread(target=detection_gui_loop, daemon=True).start()

# ---------------- LaserCube Discovery ----------------
def scanner_loop():
    while True:
        cmd_sock.sendto(bytes([CMD_GET_FULL_INFO]), ('255.255.255.255', CMD_PORT))
        time.sleep(1)
threading.Thread(target=scanner_loop, daemon=True).start()

# ---------------- Frame Generation ----------------
def gen_frame():
    with landmark_lock:
        pts_px = list(current_landmarks)
    if not pts_px:
        return [struct.pack('<HHHHH', 2048, 2048, 4095, 4095, 4095)]
    sw, sh = pyautogui.size()
    pts = []
    for x, y in pts_px:
        nx, ny = x / sw, y / sh
        lx, ly = int(nx * 4095), int((1 - ny) * 4095)
        lx = max(0, min(4095, lx))
        ly = max(0, min(4095, ly))
        pts.append((lx, ly))
    frame = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        for i in range(11):
            xi = int(x0 + (x1 - x0) * i / 10)
            yi = int(y0 + (y1 - y0) * i / 10)
            xi = max(0, min(4095, xi))
            yi = max(0, min(4095, yi))
            frame.append(struct.pack('<HHHHH', xi, yi, 4095, 4095, 4095))
    return frame

# ---------------- Main UDP + Laser Loop ----------------
def main_loop():
    print("Starting Hand Laser Projection. Press ESC in preview to exit.")
    try:
        while True:
            ready, _, _ = select.select([cmd_sock, data_sock], [], [], 0.03)
            for sock in ready:
                msg, (addr, _) = sock.recvfrom(4096)
                if addr not in known_lasers and msg and msg[0] == CMD_GET_FULL_INFO:
                    known_lasers[addr] = LaserCube(addr, gen_frame)
                    continue
                lc = known_lasers.get(addr)
                if lc:
                    was_new = lc.info is None
                    lc.recv(msg)
                    if was_new and lc.info:
                        print(f"Discovered LaserCube: {lc.info}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for lc in known_lasers.values():
            lc.stop()

if __name__ == '__main__':
    main_loop()
