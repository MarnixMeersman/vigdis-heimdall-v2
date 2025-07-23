"""Entry point for controlling a LaserCube with camera tracking."""

import cv2
import pygame
import struct
import time
from typing import Tuple

from .detection import Detector
from .laser import LaserCube, known_lasers, start_scanner

ILDA_MAX = 0xFFF

power = 1.0
color_index = 0
COLORS = [
    (ILDA_MAX, 0, 0),
    (0, ILDA_MAX, 0),
    (0, 0, ILDA_MAX),
]


def make_frame(box: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    pts = []
    for x in (x1, x2):
        for y in (y1, y2):
            px = int(max(0, min(ILDA_MAX, x / 640 * ILDA_MAX)))
            py = int(max(0, min(ILDA_MAX, 1 - y / 480) * ILDA_MAX))
            r, g, b = COLORS[color_index]
            r = int(r * power)
            g = int(g * power)
            b = int(b * power)
            pts.append(struct.pack("<HHHHH", px, py, r, g, b))
    return pts


def main():
    start_scanner()
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    detector = Detector()
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
        boxes = detector.detect(frame)
        boxes = detector.track(boxes)

        screen.fill((0, 0, 0))
        for box in boxes:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(*box), 2)
            cube.gen_frame = lambda b=box: make_frame(b)
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
