"""
Configuration management for Vigdis-Heimdall v2.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LaserConfig:
    """LaserCube configuration."""
    alive_port: int = 45456
    cmd_port: int = 45457
    data_port: int = 45458


@dataclass 
class DetectionConfig:
    """Object detection configuration."""
    model_name: str = "yolov8m-640"
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.3
    drone_class_id: int = 0
    video_path: str = "test_data/visible.mp4"


@dataclass
class HandConfig:
    """Hand tracking configuration."""
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    camera_index: int = 0


@dataclass
class TrackerConfig:
    """SORT tracker configuration."""
    max_age: int = 10
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class AppConfig:
    """Main application configuration."""
    laser: LaserConfig
    detection: DetectionConfig
    hand: HandConfig
    tracker: TrackerConfig
    mode: str = "detection"  # "detection" or "hand"
    
    def __init__(self,
                 laser: Optional[LaserConfig] = None,
                 detection: Optional[DetectionConfig] = None,
                 hand: Optional[HandConfig] = None,
                 tracker: Optional[TrackerConfig] = None,
                 mode: str = "detection"):
        self.laser = laser or LaserConfig()
        self.detection = detection or DetectionConfig()
        self.hand = hand or HandConfig()
        self.tracker = tracker or TrackerConfig()
        self.mode = mode


def get_default_config() -> AppConfig:
    """Get default application configuration."""
    return AppConfig()


def get_detection_config() -> AppConfig:
    """Get configuration optimized for object detection."""
    detection_cfg = DetectionConfig(
        model_name="yolov8m-640",
        confidence_threshold=0.3,
        video_path="test_data/visible.mp4"
    )
    return AppConfig(detection=detection_cfg, mode="detection")


def get_hand_config() -> AppConfig:
    """Get configuration optimized for hand tracking."""
    hand_cfg = HandConfig(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return AppConfig(hand=hand_cfg, mode="hand")