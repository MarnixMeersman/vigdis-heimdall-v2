#!/usr/bin/env python3
"""
Hand tracking and projection example using the modular Heimdall API.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall import create_hand_app


def main():  
    """Run hand tracking example."""
    print("Hand Tracking and Projection Example") 
    print("===================================")
    
    # Create app with custom parameters
    app = create_hand_app(
        camera_index=0,               # Use first camera
        max_num_hands=2,              # Track up to 2 hands
        min_detection_confidence=0.7  # Higher confidence for stable tracking
    )
    
    print("Starting hand tracking...")
    print("Show your hands to the camera")
    print("Press Ctrl+C to stop")
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    main()