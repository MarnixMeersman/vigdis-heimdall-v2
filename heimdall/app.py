import cv2
import pygame
import struct
import time
from typing import List, Tuple

from .detection import Detector
from .laser import LaserCube, known_lasers, start_scanner
from .sort import Sort

ILDA_MAX = 0xFFF

power = 1.0
color_index = 0
COLORS = [
    (ILDA_MAX, 0, 0),
    (0, ILDA_MAX, 0),
    (0, 0, ILDA_MAX),
]


def make_point_frame(points: List[Tuple[int, int]]):
    """Convert points to ILDA formatted bytes."""
    pts = []
    for x, y in points:
        px = int(max(0, min(ILDA_MAX, x / 640 * ILDA_MAX)))
        py = int(max(0, min(ILDA_MAX, 1 - y / 480) * ILDA_MAX))
        r, g, b = COLORS[color_index]
        r = int(r * power)
        g = int(g * power)
        b = int(b * power)
        pt = struct.pack("<HHHHH", px, py, r, g, b)
        pts.extend([pt] * 10)
    return pts


def main():
    start_scanner()
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    detector = Detector()
    tracker = Sort()
    cap = cv2.VideoCapture(0)

    cube = None
    while cube is None:
        if known_lasers:
            cube = next(iter(known_lasers.values()))
        time.sleep(0.1)

    cube.gen_frame = lambda: []

    clock = pygame.time.Clock()
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detector.detect(frame)
        centers = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in boxes]
        tracks = tracker.update(centers)

        screen.fill((0, 0, 0))
        for (x1, y1, x2, y2) in boxes:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(x1, y1, x2 - x1, y2 - y1), 2)
        for pt in tracks:
            pygame.draw.circle(screen, (255, 0, 0), pt, 4)
        if tracks:
            cube.gen_frame = lambda p=tracks: make_point_frame(p)
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
