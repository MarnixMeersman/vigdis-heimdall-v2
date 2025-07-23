"""Minimal SORT tracker for 2D points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter


@dataclass
class Track:
    kf: KalmanFilter
    id: int
    age: int = 0
    time_since_update: int = 0

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:2].reshape(-1)

    def update(self, xy: Tuple[float, float]) -> None:
        self.time_since_update = 0
        self.kf.update(np.array(xy))

    def get_state(self) -> Tuple[int, int]:
        x, y = self.kf.x[:2].reshape(-1)
        return int(x), int(y)


class Sort:
    """Very small tracker matching points by Euclidean distance."""

    def __init__(self, max_age: int = 5, dist_threshold: float = 50.0) -> None:
        self.max_age = max_age
        self.dist_threshold = dist_threshold
        self.tracks: List[Track] = []
        self.next_id = 0

    def _create_track(self, xy: Tuple[int, int]) -> Track:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P *= 500.0
        kf.R *= 10.0
        kf.Q[2:, 2:] *= 0.01
        kf.x[:2] = np.array(xy).reshape(2, 1)
        track = Track(kf=kf, id=self.next_id)
        self.next_id += 1
        return track

    def update(self, detections: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Predict existing tracks
        predictions = [t.predict() for t in self.tracks]
        matched = set()

        for det in detections:
            det_arr = np.array(det)
            best_idx = None
            best_dist = self.dist_threshold
            for i, pred in enumerate(predictions):
                if i in matched:
                    continue
                dist = np.linalg.norm(det_arr - pred)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                self.tracks[best_idx].update(det)
                matched.add(best_idx)
            else:
                self.tracks.append(self._create_track(det))

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t.get_state() for t in self.tracks]
