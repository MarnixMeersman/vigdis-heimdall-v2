"""YOLO detection and object tracking helpers."""

from typing import List, Tuple

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - library optional
    YOLO = None

try:
    from trackers import BYTETracker  # type: ignore
except Exception:  # pragma: no cover - library optional
    BYTETracker = None


class Detector:
    """Wrapper around YOLO and Roboflow trackers."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        if YOLO is None:
            raise ImportError("ultralytics package required for detection")
        self.model = YOLO(model_path)
        self.tracker = BYTETracker() if BYTETracker else None

    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes for detected objects."""
        results = self.model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        return [tuple(map(int, b)) for b in boxes]

    def track(self, boxes: List[Tuple[int, int, int, int]]):
        if self.tracker is None:
            return boxes
        return self.tracker.update(boxes)
