"""
Main application logic for Vigdis-Heimdall v2.
Combines object detection/hand tracking with laser projection.
"""

import time
from typing import Optional

from .config import AppConfig, get_default_config
from .core.laser import LaserController
from .core.detection import create_detector_with_video
from .core.hand import create_hand_tracker_with_camera


class HeimdallApp:
    """Main Heimdall application."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize Heimdall application.
        
        Args:
            config: Application configuration. Uses default if None.
        """
        self.config = config or get_default_config()
        self.laser_controller: Optional[LaserController] = None
        self.detector = None
        self.hand_tracker = None
        self.running = False
    
    def run_detection_mode(self):
        """Run in object detection mode."""
        print("Starting Heimdall in detection mode...")
        print(f"Video source: {self.config.detection.video_path}")
        print(f"Model: {self.config.detection.model_name}")
        
        # Create detector and frame generator
        self.detector, frame_generator = create_detector_with_video(
            video_path=self.config.detection.video_path,
            model_name=self.config.detection.model_name,
            confidence_threshold=self.config.detection.confidence_threshold
        )
        
        # Create laser controller
        self.laser_controller = LaserController(frame_generator)
        
        print("Detection mode started. Press Ctrl+C to stop.")
        try:
            self.laser_controller.run()
        except KeyboardInterrupt:
            print("Stopping detection mode...")
        finally:
            self.stop()
    
    def run_hand_mode(self):
        """Run in hand tracking mode."""
        print("Starting Heimdall in hand tracking mode...")
        print(f"Camera index: {self.config.hand.camera_index}")
        
        # Create hand tracker and frame generator
        self.hand_tracker, frame_generator = create_hand_tracker_with_camera(
            camera_index=self.config.hand.camera_index
        )
        
        # Create laser controller
        self.laser_controller = LaserController(frame_generator)
        
        print("Hand tracking mode started. Press Ctrl+C to stop.")
        try:
            self.laser_controller.run()
        except KeyboardInterrupt:
            print("Stopping hand tracking mode...")
        finally:
            self.stop()
    
    def run(self):
        """Run the application in the configured mode."""
        if self.running:
            print("Application is already running!")
            return
            
        self.running = True
        
        try:
            if self.config.mode == "detection":
                self.run_detection_mode()
            elif self.config.mode == "hand":
                self.run_hand_mode()
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
        finally:
            self.running = False
    
    def stop(self):
        """Stop the application and clean up resources."""
        if self.laser_controller:
            self.laser_controller.stop()
        
        if self.detector:
            self.detector.stop_detection()
        
        if self.hand_tracker:
            self.hand_tracker.stop_tracking()
        
        self.running = False
        print("Application stopped.")


def create_detection_app(video_path: str = "test_data/visible.mp4", 
                        model_name: str = "yolov8m-640",
                        confidence_threshold: float = 0.3) -> HeimdallApp:
    """
    Create application configured for object detection.
    
    Args:
        video_path: Path to video file
        model_name: YOLO model name
        confidence_threshold: Detection confidence threshold
        
    Returns:
        Configured HeimdallApp instance
    """
    from .config import DetectionConfig, AppConfig
    
    detection_config = DetectionConfig(
        video_path=video_path,
        model_name=model_name,
        confidence_threshold=confidence_threshold
    )
    
    config = AppConfig(detection=detection_config, mode="detection")
    return HeimdallApp(config)


def create_hand_app(camera_index: int = 0,
                   max_num_hands: int = 2,
                   min_detection_confidence: float = 0.5) -> HeimdallApp:
    """
    Create application configured for hand tracking.
    
    Args:
        camera_index: Camera device index
        max_num_hands: Maximum number of hands to track
        min_detection_confidence: Minimum confidence for hand detection
        
    Returns:
        Configured HeimdallApp instance
    """
    from .config import HandConfig, AppConfig
    
    hand_config = HandConfig(
        camera_index=camera_index,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence
    )
    
    config = AppConfig(hand=hand_config, mode="hand")
    return HeimdallApp(config)