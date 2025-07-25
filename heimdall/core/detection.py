"""
Object detection and tracking module using YOLO and SORT.
"""

import threading
import time
from typing import Optional, Tuple, Callable
import cv2
import supervision as sv
from inference import get_model
from trackers import SORTTracker


class ObjectDetector:
    """YOLO-based object detector with tracking."""
    
    def __init__(self, 
                 model_name: str = "yolov8m-640",
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.3,
                 target_class_id: int = 0):
        """
        Initialize object detector.
        
        Args:
            model_name: YOLO model identifier
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            target_class_id: Class ID to track (0 for drone)
        """
        self.model = get_model(model_name)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_class_id = target_class_id
        self.tracker = SORTTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        
        # Thread-safe state
        self.bbox_lock = threading.Lock()
        self.current_bbox: Optional[Tuple[float, float, float, float]] = None
        self.running = False
        self.detection_thread: Optional[threading.Thread] = None
    
    def start_detection(self, video_source):
        """Start detection in a separate thread."""
        if self.running:
            return
            
        self.running = True
        self.video_source = video_source
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop detection thread."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
    
    def get_current_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the current bounding box in thread-safe manner."""
        with self.bbox_lock:
            return self.current_bbox
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        cap = cv2.VideoCapture(self.video_source)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            result = self.model.infer(frame, confidence=self.confidence_threshold)[0]
            
            # Convert to supervision Detections and apply NMS
            detections = sv.Detections.from_inference(result).with_nms(self.nms_threshold)
            
            # Filter by target class
            detections = detections[detections.class_id == self.target_class_id]
            
            # Update tracker
            tracked = self.tracker.update(detections)
            
            # Update current bbox (pick first track if any)
            if len(tracked.xyxy) > 0:
                x1, y1, x2, y2 = tracked.xyxy[0]
                with self.bbox_lock:
                    self.current_bbox = (x1, y1, x2, y2)
            else:
                with self.bbox_lock:
                    self.current_bbox = None
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
        
        cap.release()


class DetectionFrameGenerator:
    """Frame generator that uses detection results."""
    
    def __init__(self, detector: ObjectDetector, screen_width: int = None, screen_height: int = None):
        self.detector = detector
        # Get screen size if not provided
        if screen_width is None or screen_height is None:
            try:
                import pyautogui
                self.screen_width, self.screen_height = pyautogui.size()
            except ImportError:
                self.screen_width, self.screen_height = 1920, 1080
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height
    
    def __call__(self):
        """Generate frame data based on current detection."""
        from .laser import create_frame_from_bbox
        bbox = self.detector.get_current_bbox()
        return create_frame_from_bbox(bbox, self.screen_width, self.screen_height)


def create_detector_with_video(video_path: str, 
                              model_name: str = "yolov8m-640",
                              confidence_threshold: float = 0.3) -> Tuple[ObjectDetector, Callable]:
    """
    Convenience function to create detector and frame generator for video input.
    
    Returns:
        Tuple of (detector, frame_generator_function)
    """
    detector = ObjectDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold
    )
    
    # Start detection on the video
    detector.start_detection(video_path)
    
    # Create frame generator
    frame_generator = DetectionFrameGenerator(detector)
    
    return detector, frame_generator