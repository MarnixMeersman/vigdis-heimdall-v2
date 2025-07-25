#!/usr/bin/env python3
"""
Vigdis-Heimdall v2 main entrypoint.
Maintains backward compatibility while using the new modular architecture.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from heimdall import create_detection_app, create_hand_app, get_default_config


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Vigdis-Heimdall v2 - Laser projector control with object detection and hand tracking"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["detection", "hand"], 
        default="detection",
        help="Operation mode: 'detection' for object tracking, 'hand' for hand tracking"
    )
    
    parser.add_argument(
        "--video", 
        default="test_data/visible.mp4",
        help="Video file path for detection mode (default: test_data/visible.mp4)"
    )
    
    parser.add_argument(
        "--model",
        default="yolov8m-640", 
        help="YOLO model name for detection (default: yolov8m-640)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for hand tracking mode (default: 0)"
    )
    
    parser.add_argument(
        "--hands",
        type=int,
        default=2,
        help="Maximum number of hands to track (default: 2)"
    )

    args = parser.parse_args()
    
    # Check if video file exists for detection mode
    if args.mode == "detection" and not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        print("Available test data:")
        test_data_dir = Path("test_data")
        if test_data_dir.exists():
            for file in test_data_dir.glob("*"):
                print(f"  - {file}")
        else:
            print("  No test_data directory found")
        return 1
    
    try:
        if args.mode == "detection":
            print(f"Starting Heimdall in detection mode with video: {args.video}")
            app = create_detection_app(
                video_path=args.video,
                model_name=args.model,
                confidence_threshold=args.confidence
            )
        else:  # hand mode
            print(f"Starting Heimdall in hand tracking mode with camera: {args.camera}")
            app = create_hand_app(
                camera_index=args.camera,
                max_num_hands=args.hands
            )
        
        app.run()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


def run_original_compatible():
    """
    Run in original main.py compatible mode for backward compatibility.
    This maintains the exact behavior of the original main.py file.
    """
    print("Running in original compatibility mode...")
    
    # Use the default video path from the original
    app = create_detection_app(
        video_path="test_data/visible.mp4",
        model_name="yolov8m-640",
        confidence_threshold=0.3
    )
    
    app.run()


if __name__ == "__main__":
    # If no arguments are provided, run in compatibility mode
    if len(sys.argv) == 1:
        run_original_compatible()
    else:
        sys.exit(main())