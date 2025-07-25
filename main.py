# Requirements:
# pip install inference-gpu trackers supervision opencv-python

import collections
import select
import socket
import struct
import threading
import time

import cv2
import numpy as np
import pyautogui
import supervision as sv
from inference import get_model
from trackers import SORTTracker

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

# Drone detection parameters
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
DRONE_CLASS_ID = 0  # adjust to your model's class ID for "drone"

# Initialize detector and SORT tracker
model = get_model("yolov8m-640")
tracker = SORTTracker(max_age=10, min_hits=3, iou_threshold=0.3)

# Video source
VIDEO_PATH = "path/to/your/video.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# Shared state for current bounding boxbbox_lock = threading.Lock()
current_bbox = None  # (x1,y1,x2,y2)

# Setup UDP sockets for LaserCube communication
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
cmd_sock.bind(("0.0.0.0", CMD_PORT))

data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
data_sock.bind(("0.0.0.0", DATA_PORT))

# Namedtuple for LaserCube info
LaserInfo = collections.namedtuple('LaserInfo', [
    'model_name', 'fw_major', 'fw_minor', 'output_enabled',
    'dac_rate', 'max_dac_rate', 'rx_buffer_free', 'rx_buffer_size',
    'battery_percent', 'temperature', 'connection_type',
    'model_number', 'serial_number', 'ip_addr'
])

known_lasers = {}

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
        if cmd == CMD_GET_FULL_INFO:
            fields = struct.unpack('<xxBB?5xIIxHHBBB11xB26x', msg)
            serial = struct.unpack('6B', msg[26:32])
            ip = struct.unpack('4B', msg[32:36])
            name = msg[38:].split(b'\0',1)[0].decode()
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
        # CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA ack ignored

    def _send_cmd(self, cmd_bytes):
        cmd_sock.sendto(bytes(cmd_bytes), (self.addr, CMD_PORT))
        cmd_sock.sendto(bytes(cmd_bytes), (self.addr, CMD_PORT))

    def _main_loop(self):
        msg_num = 0
        frame_num = 0
        self._send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA,1])
        self._send_cmd([CMD_SET_OUTPUT,1])
        while self.running:
            pts = self.gen_frame()
            while pts:
                if self.remote_buf_free < 5000:
                    time.sleep(100/self.info.dac_rate)
                    self.remote_buf_free += 100
                hdr = bytes([CMD_SAMPLE_DATA,0, msg_num%256, frame_num%256])
                pkt = hdr + b''.join(pts[:140])
                for _ in pts[:140]: self.remote_buf_free -= 1
                data_sock.sendto(pkt,(self.addr,DATA_PORT))
                msg_num +=1
                del pts[:140]
            frame_num+=1
        self._send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA,0])
        self._send_cmd([CMD_SET_OUTPUT,0])

# Background detection & tracking thread
def detection_loop():
    global current_bbox
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Inference
        result = model.infer(frame, confidence=CONFIDENCE_THRESHOLD)[0]
        # Convert to supervision Detections, apply NMS
        detections = sv.Detections.from_inference(result).with_nms(NMS_THRESHOLD)
        # Filter class and run SORT
        detections = detections[(detections.class_id == DRONE_CLASS_ID)]
        tracked = tracker.update(detections)
        # Pick first track if any
        if len(tracked.xyxy) > 0:
            x1,y1,x2,y2 = tracked.xyxy[0]
            with bbox_lock:
                current_bbox = (x1,y1,x2,y2)
        else:
            with bbox_lock:
                current_bbox = None
        time.sleep(0.01)

threading.Thread(target=detection_loop, daemon=True).start()

# LAN broadcast to find LaserCubes
def scanner():
    while True:
        cmd_sock.sendto(bytes([CMD_GET_FULL_INFO]),('255.255.255.255',CMD_PORT))
        time.sleep(1)
threading.Thread(target=scanner, daemon=True).start()

# Generate ILDA frame based on current_bbox
def gen_frame():
    with bbox_lock:
        bb = current_bbox
    if bb is None:
        pts = [(2048,2048)]
    else:
        x1,y1,x2,y2 = bb
        sw, sh = pyautogui.size()
        nx1, ny1 = x1/sw, y1/sh
        nx2, ny2 = x2/sw, y2/sh
        lx1 = int((1-nx1)*4095); ly1 = int((1-ny1)*4095)
        lx2 = int((1-nx2)*4095); ly2 = int((1-ny2)*4095)
        pts = [(lx1,ly1),(lx2,ly1),(lx2,ly2),(lx1,ly2),(lx1,ly1)]
    frame = []
    for (x0,y0),(x1,y1) in zip(pts,pts[1:]):
        for i in range(200):
            xi = x0 + (x1-x0)*i/200
            yi = y0 + (y1-y0)*i/200
            frame.append(struct.pack('<HHHHH',int(xi),int(yi),4095,4095,4095))
    return frame

# Main loop: dispatch incoming UDP to LaserCube instances
try:
    while True:
        ready,_,_ = select.select([cmd_sock,data_sock],[],[])
        for sock in ready:
            msg,(addr,_) = sock.recvfrom(4096)
            if addr not in known_lasers:
                if msg and msg[0]==CMD_GET_FULL_INFO:
                    known_lasers[addr] = LaserCube(addr, gen_frame)
                else:
                    continue
            lc = known_lasers[addr]
            new = lc.info is None
            lc.recv(msg)
            if new:
                print("Found LaserCube:", lc.info)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    for lc in known_lasers.values(): lc.stop()
    cap.release()
