"""
Object detection and tracking module using YOLO and SORT.
"""

# Suppress inference library model dependency warnings BEFORE imports
import os
os.environ.update({
    'PALIGEMMA_ENABLED': 'False',
    'FLORENCE2_ENABLED': 'False',
    'QWEN_2_5_ENABLED': 'False',
    'CORE_MODEL_SAM_ENABLED': 'False',
    'CORE_MODEL_SAM2_ENABLED': 'False',
    'CORE_MODEL_CLIP_ENABLED': 'False',
    'SMOLVLM2_ENABLED': 'False',
    'DEPTH_ESTIMATION_ENABLED': 'False',
    'MOONDREAM2_ENABLED': 'False',
    'CORE_MODEL_TROCR_ENABLED': 'False',
    'CORE_MODEL_GROUNDINGDINO_ENABLED': 'False',
    'CORE_MODEL_YOLO_WORLD_ENABLED': 'False',
    'CORE_MODEL_PE_ENABLED': 'False'
})

import threading
import time
import warnings
from typing import Optional, Tuple, Callable
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from motpy import MultiObjectTracker, Detection
import torch

# Suppress ML library warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class ObjectDetector:
    """YOLO-based object detector with tracking."""
    
    def __init__(self, 
                 model_name: str = "models/yolov8x",
                 confidence_threshold: float = 0.1,
                 nms_threshold: float = 0.5,
                 target_class_id: int = 4,  # Default to airplane (class 4) for UAV detection
                 target_class_ids: list = None,  # Support multiple class IDs
                 use_gpu: bool = True,
                 prefer_ultralytics: bool = True):
        """
        Initialize object detector.
        
        Args:
            model_name: YOLO model identifier
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            target_class_id: Primary class ID to track (4 for airplane/UAV)
            target_class_ids: List of class IDs to track (e.g., [4, 14, 33] for airplane, bird, kite)
            use_gpu: Enable GPU acceleration using Metal Performance Shaders on Apple Silicon
            prefer_ultralytics: Use Ultralytics YOLO for better GPU support when available
        """
        # Setup device for GPU acceleration
        self.device = self._setup_device(use_gpu)
        device_name = "Apple Metal (MPS)" if self.device == "mps" else "CUDA GPU" if self.device == "cuda" else "CPU"
        print(f"🚀 Inference device: {device_name} ({self.device})")
        
        # Choose model backend based on GPU support and availability
        self.use_ultralytics = (prefer_ultralytics and ULTRALYTICS_AVAILABLE and 
                               self.device != "cpu")
        
        if self.use_ultralytics:
            # Convert model name for Ultralytics format
            ultralytics_model = model_name.replace("-640", ".pt") if "-640" in model_name else f"{model_name}.pt"
            self.model = YOLO(ultralytics_model)
            # Move model to device for GPU acceleration
            if hasattr(self.model.model, 'to'):
                self.model.model.to(self.device)
            print(f"Using Ultralytics YOLO model: {ultralytics_model}")
        else:
            self.model = get_model(model_name)
            print(f"Using Roboflow inference model: {model_name}")
        
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_class_id = target_class_id
        # Set up target class IDs for multi-class UAV/drone detection
        if target_class_ids is None:
            # Default: airplane (4), bird (14), kite (33) - objects that might be UAVs/drones
            self.target_class_ids = [4, 14, 33] if target_class_id == 4 else [target_class_id]
        else:
            self.target_class_ids = target_class_ids
        self.tracker = MultiObjectTracker(
            dt=0.1, 
            tracker_kwargs={
                'max_staleness': 15,  # Keep tracks longer for UAVs
                'min_hits': 2,        # Reduce minimum hits needed
                'iou_threshold': 0.2  # Lower IoU threshold for small UAVs
            }
        )
        
        # Thread-safe state
        self.bbox_lock = threading.Lock()
        self.current_bbox: Optional[Tuple[float, float, float, float]] = None
        self.running = False
        self.detection_thread: Optional[threading.Thread] = None
    
    def _setup_device(self, use_gpu: bool) -> str:
        """Setup computation device with Apple Metal support."""
        if not use_gpu:
            return "cpu"
        
        # Check for Apple Silicon MPS support
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        # Fallback to CUDA if available
        elif torch.cuda.is_available():
            return "cuda"
        else:
            print("GPU acceleration not available, falling back to CPU")
            return "cpu"
    
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
    
    def _run_inference(self, frame):
        """Run inference on frame using appropriate backend."""
        if self.use_ultralytics:
            # Use Ultralytics YOLO with GPU acceleration
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = self.model(frame, device=self.device, verbose=False, conf=self.confidence_threshold)
            detections = sv.Detections.from_ultralytics(results[0])
        else:
            # Use Roboflow inference (CPU only)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.infer(frame, confidence=self.confidence_threshold)[0]
            detections = sv.Detections.from_inference(result)
        
        return detections
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        cap = cv2.VideoCapture(self.video_source)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            detections = self._run_inference(frame)
            
            # Apply NMS and confidence filtering
            detections = detections.with_nms(self.nms_threshold)
            if not self.use_ultralytics:
                # Confidence filtering for Roboflow (Ultralytics handles this internally)
                detections = detections[detections.confidence >= self.confidence_threshold]
            
            # Filter by target class IDs (support multiple classes for UAV/drone detection)
            class_mask = np.isin(detections.class_id, self.target_class_ids)
            detections = detections[class_mask]
            
            # Update tracker
            if len(detections.xyxy) > 0:
                # Convert detections to motpy format
                detection_boxes = detections.xyxy
                confidence_scores = detections.confidence if detections.confidence is not None else [0.5] * len(detection_boxes)
                class_ids = detections.class_id if detections.class_id is not None else [self.target_class_id] * len(detection_boxes)
                
                # Create Detection objects for motpy
                detection_list = [
                    Detection(box=np.array(box), score=conf, class_id=cls_id) 
                    for box, conf, cls_id in zip(detection_boxes, confidence_scores, class_ids)
                ]
                tracks = self.tracker.step(detection_list)
                
                # Update current bbox (pick first track if any)
                if tracks:
                    track = tracks[0]
                    x1, y1, x2, y2 = track.box
                    with self.bbox_lock:
                        self.current_bbox = (x1, y1, x2, y2)
            else:
                # Step with empty detections to age out tracks
                self.tracker.step([])
                with self.bbox_lock:
                    self.current_bbox = None
            
            time.sleep(0.005)  # Faster detection loop for UAVs
        
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