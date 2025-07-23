"""Entry point for controlling a LaserCube with camera tracking."""

import cv2
import pygame
import struct
import time
from typing import Tuple

from .hand import HandTracker
from .laser import LaserCube, known_lasers, start_scanner

ILDA_MAX = 0xFFF

power = 1.0
color_index = 0
COLORS = [
    (ILDA_MAX, 0, 0),
    (0, ILDA_MAX, 0),
    (0, 0, ILDA_MAX),
]


def make_point_frame(point: Tuple[int, int]):
    """Convert a single (x, y) point to ILDA formatted bytes."""
    x, y = point
    px = int(max(0, min(ILDA_MAX, x / 640 * ILDA_MAX)))
    py = int(max(0, min(ILDA_MAX, 1 - y / 480) * ILDA_MAX))
    r, g, b = COLORS[color_index]
    r = int(r * power)
    g = int(g * power)
    b = int(b * power)
    pt = struct.pack("<HHHHH", px, py, r, g, b)
    return [pt] * 10  # repeat to keep laser on


def main():
    start_scanner()
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    cube = None
    while cube is None:
        if known_lasers:
            cube = next(iter(known_lasers.values()))
        for sock in []:
            pass
        time.sleep(0.1)

    cube.gen_frame = lambda: []

    clock = pygame.time.Clock()
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        points = tracker.detect(frame)

        screen.fill((0, 0, 0))
        if points:
            pt = points[0]
            pygame.draw.circle(screen, (0, 255, 0), pt, 5)
            cube.gen_frame = lambda p=pt: make_point_frame(p)
        else:
            cube.gen_frame = lambda: []
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                global color_index, power
                if ev.key == pygame.K_UP:
                    power = min(1.0, power + 0.1)
                elif ev.key == pygame.K_DOWN:
                    power = max(0.1, power - 0.1)
                elif ev.key == pygame.K_LEFT:
                    color_index = (color_index - 1) % len(COLORS)
                elif ev.key == pygame.K_RIGHT:
                    color_index = (color_index + 1) % len(COLORS)

        clock.tick(30)

    cap.release()
    pygame.quit()
    cube.stop()


if __name__ == "__main__":
    main()
