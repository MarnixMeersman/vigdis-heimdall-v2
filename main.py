# -----------------------------------------------------------------------------
# © 2025 Vigdis Space and Defence. All rights reserved.
# Produced by Vigdis Space and Defence.
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
import collections
import math
import select
import socket
import struct
import threading
import time

import pygame

# —————— LaserCube network setup (as before) ——————

ALIVE_PORT = 45456
CMD_PORT   = 45457
DATA_PORT  = 45458

CMD_GET_FULL_INFO                           = 0x77
CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA     = 0x78
CMD_SET_OUTPUT                              = 0x80
CMD_SAMPLE_DATA                             = 0xa9
CMD_GET_RINGBUFFER_EMPTY_SAMPLE_COUNT       = 0x8a

cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
cmd_sock.bind(('0.0.0.0', CMD_PORT))

data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
data_sock.bind(('0.0.0.0', DATA_PORT))

LaserInfo = collections.namedtuple('LaserInfo', [
    'model_name',
    'fw_major', 'fw_minor', 'output_enabled',
    'dac_rate', 'max_dac_rate',
    'rx_buffer_free', 'rx_buffer_size',
    'battery_percent', 'temperature', 'connection_type',
    'model_number', 'serial_number', 'ip_addr'])

known_lasers = {}

class LaserCube:
    def __init__(self, addr, gen_frame):
        self.addr = addr
        self.gen_frame = gen_frame
        self.info = None
        self.remote_buf_free = 0
        self.running = True
        threading.Thread(target=self.main, daemon=True).start()

    def stop(self):
        self.running = False

    def recv(self, msg):
        cmd = msg[0]
        if cmd == CMD_GET_FULL_INFO and len(msg) > 1:
            fields = struct.unpack('<xxBB?5xIIxHHBBB11xB26x', msg[:50])
            serial = struct.unpack('6B', msg[26:32])
            ip = struct.unpack('4B', msg[32:36])
            name = msg[38:].split(b'\0',1)[0].decode()
            info = LaserInfo(
                name, *fields,
                ':'.join(f'{b:02x}' for b in serial),
                '.'.join(str(b) for b in ip)
            )
            if info != self.info:
                self.info = info
                self.remote_buf_free = info.rx_buffer_free

        elif cmd == CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA:
            pass

        elif cmd == CMD_GET_RINGBUFFER_EMPTY_SAMPLE_COUNT:
            self.remote_buf_free = struct.unpack('<xxH', msg)[0]

    def send_cmd(self, cmd_bytes):
        for _ in range(2):
            cmd_sock.sendto(bytes(cmd_bytes), (self.addr, CMD_PORT))

    def main(self):
        msg_num = 0
        frame_num = 0

        # Enable output + buffer reporting
        self.send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 1])
        self.send_cmd([CMD_SET_OUTPUT, 1])

        while self.running:
            pts = self.gen_frame()[:]  # copy list of packed points
            # chunk into MTU‐safe packets
            while pts:
                if self.remote_buf_free < 5000:
                    time.sleep(100/self.info.dac_rate)
                    self.remote_buf_free += 100

                hdr = bytes([CMD_SAMPLE_DATA, 0x00, msg_num%256, frame_num%256])
                payload = b''.join(pts[:140])
                pts = pts[140:]
                self.remote_buf_free -= len(payload)//10  # approx points
                data_sock.sendto(hdr + payload, (self.addr, DATA_PORT))
                msg_num += 1

            frame_num += 1

        # Disable on exit
        self.send_cmd([CMD_ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 0])
        self.send_cmd([CMD_SET_OUTPUT, 0])

def scanner():
    while True:
        cmd_sock.sendto(bytes([CMD_GET_FULL_INFO]), ('255.255.255.255', CMD_PORT))
        time.sleep(1)
threading.Thread(target=scanner, daemon=True).start()

# —————— Pygame + gen_frame setup ——————

# ILDA coordinate range
ILDA_MAX = 0xFFF

# Shared state
mouse_x, mouse_y = 0.5, 0.5    # normalized [0..1]
brightness = 1.0              # 1.0 = full, 0.2 = dim

# When the user clicks, toggle between bright and dim
def handle_click():
    global brightness
    brightness = 0.2 if brightness > 0.5 else 1.0

# Create a gen_frame function that draws a filled square centered on mouse
SQUARE_SIZE = 0.1  # relative size of the square

def gen_frame():
    pts = []
    # compute square in normalized coords
    half = SQUARE_SIZE / 2
    corners = [
        (mouse_x - half, mouse_y - half),
        (mouse_x + half, mouse_y - half),
        (mouse_x + half, mouse_y + half),
        (mouse_x - half, mouse_y + half),
    ]
    # Triangulate square into two triangles
    tris = [ (corners[0], corners[1], corners[2]),
             (corners[0], corners[2], corners[3]) ]

    for tri in tris:
        # simple barycentric fill: sample a grid inside triangle
        for i in range(0, 10):
            for j in range(0, 10):
                # interpolate
                u = (i/9)*(1 - j/9)
                v = j/9
                w = 1 - u - v
                nx = u*tri[0][0] + v*tri[1][0] + w*tri[2][0]
                ny = u*tri[0][1] + v*tri[1][1] + w*tri[2][1]
                # map to 0..ILDA_MAX
                x = int(max(0, min(ILDA_MAX, nx * ILDA_MAX)))
                y = int(max(0, min(ILDA_MAX, ny * ILDA_MAX)))
                # color modulated by brightness
                r = int(ILDA_MAX * brightness)
                g = int(ILDA_MAX * brightness)
                b = int(ILDA_MAX * brightness)
                pts.append(struct.pack('<HHHHH', x, y, r, g, b))
    return pts

# Launch Pygame to track the mouse
pygame.init()
screen = pygame.display.set_mode((400,400))
pygame.display.set_caption("LaserCube Control — move & click")

# Pass our custom gen_frame into the LaserCube once we find the first laser
gen_frame_ref = gen_frame

def main_loop():
    global mouse_x, mouse_y
    # Wait until we discover at least one LaserCube
    while not known_lasers:
        time.sleep(0.1)
    # Hook the first cube we see
    addr, cube = next(iter(known_lasers.items()))
    cube.gen_frame = gen_frame_ref

    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                # normalize
                mouse_x = mx / screen.get_width()
                mouse_y = 1 - (my / screen.get_height())  # invert Y
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                handle_click()

        # simple visual feedback in window
        screen.fill((0,0,0))
        sx = int(mouse_x * 400)
        sy = int((1-mouse_y) * 400)
        color = (int(255*brightness),)*3
        pygame.draw.rect(screen, color,
                         pygame.Rect(sx-10, sy-10, 20, 20))
        pygame.display.flip()
        clock.tick(60)

    # clean exit
    cube.stop()
    pygame.quit()

if __name__ == '__main__':
    try:
        main_loop()
    except KeyboardInterrupt:
        for c in known_lasers.values():
            c.stop()
        pygame.quit()
        print("Exiting.")