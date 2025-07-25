"""
Vigdis-Heimdall v2 - Laser projector control with object detection and hand tracking.
"""

from .app import HeimdallApp, create_detection_app, create_hand_app
from .config import AppConfig, get_default_config, get_detection_config, get_hand_config

__version__ = "2.0.0"
__all__ = [
    "HeimdallApp", 
    "create_detection_app", 
    "create_hand_app",
    "AppConfig",
    "get_default_config",
    "get_detection_config", 
    "get_hand_config"
]