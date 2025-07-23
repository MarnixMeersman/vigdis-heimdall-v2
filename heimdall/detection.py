"""Roboflow hand detection helper."""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
from roboflow import Roboflow


class Detector:
    """Wrapper around a Roboflow object detection model."""

    def __init__(self, model_id: str | None = None, api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        model_id = model_id or os.environ.get("ROBOFLOW_MODEL")
        if api_key is None or model_id is None:
            raise RuntimeError("ROBOFLOW_API_KEY and ROBOFLOW_MODEL must be set")

        rf = Roboflow(api_key=api_key)
        workspace, project, version = model_id.split("/")
        self.model = rf.workspace(workspace).project(project).version(version).model

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes for detected hands in (x1, y1, x2, y2) format."""
        preds = self.model.predict(frame, confidence=40, overlap=30).json()
        boxes: List[Tuple[int, int, int, int]] = []
        for det in preds.get("predictions", []):
            x, y, w, h = det["x"], det["y"], det["width"], det["height"]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            boxes.append((x1, y1, x2, y2))
        return boxes
