#!/usr/bin/env python3
"""
Basic object tracking example using the modular Heimdall API.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall import create_detection_app


def main():
    """Run basic object tracking example."""
    print("Basic Object Tracking Example")
    print("============================")
    
    # Create app with custom parameters
    app = create_detection_app(
        video_path="test_data/visible.mp4",
        model_name="yolov8m-640", 
        confidence_threshold=0.4  # Higher confidence for fewer false positives
    )
    
    print("Starting object tracking...")
    print("Press Ctrl+C to stop")
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    main()