# Vigdis-Heimdall v2 - Complete Usage Guide

This document provides comprehensive instructions for using Vigdis-Heimdall v2 in all its modes and configurations.

## Table of Contents
- [Quick Start](#quick-start)
- [Command Line Usage](#command-line-usage)
- [Programmatic API](#programmatic-api)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Absolute Minimum

```bash
# Run with defaults (detection mode, default video)
python main.py
```

This is identical to the original v1 behavior and will:
- Use object detection mode
- Process `test_data/visible.mp4`
- Use YOLOv8m model with 0.3 confidence threshold
- Project bounding boxes onto detected objects via LaserCube

## Command Line Usage

### Basic Commands

#### Object Detection (Default Mode)

```bash
# Most basic usage
python main.py

# Explicit detection mode
python main.py --mode detection

# With custom video file
python main.py --video my_video.mp4

# High accuracy with slower model
python main.py --model yolov8l --confidence 0.5

# Fast detection with lower accuracy
python main.py --model yolov8n --confidence 0.2
```

#### Hand Tracking Mode

```bash
# Basic hand tracking
python main.py --mode hand

# Use second camera
python main.py --mode hand --camera 1

# Track only one hand for better performance
python main.py --mode hand --hands 1

# High confidence tracking
python main.py --mode hand --camera 0 --hands 2
```

### Advanced Command Examples

#### Performance Optimization

```bash
# Maximum speed (lower accuracy)
python main.py --model yolov8n --confidence 0.2

# Maximum accuracy (slower)
python main.py --model yolov8x --confidence 0.7

# Balanced performance
python main.py --model yolov8m-640 --confidence 0.4
```

#### Multiple Videos/Cameras

```bash
# Process different video files
python main.py --video recording1.mp4
python main.py --video recording2.mp4

# Use different cameras for hand tracking
python main.py --mode hand --camera 0  # Built-in camera
python main.py --mode hand --camera 1  # External USB camera
```

### Complete Parameter Reference

| Parameter | Type | Default | Description | Valid Values |
|-----------|------|---------|-------------|--------------|
| `--mode` | string | `detection` | Operation mode | `detection`, `hand` |
| `--video` | string | `test_data/visible.mp4` | Video file path | Any video file path |
| `--model` | string | `yolov8m-640` | YOLO model name | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` (with optional `-640`) |
| `--confidence` | float | `0.3` | Detection threshold | `0.0` to `1.0` |
| `--camera` | integer | `0` | Camera device index | `0`, `1`, `2`, etc. |
| `--hands` | integer | `2` | Max hands to track | `1` to `4` |

## Programmatic API

### Simple Usage

#### Detection App

```python
from heimdall import create_detection_app

# Minimal setup
app = create_detection_app()
app.run()

# Custom parameters
app = create_detection_app(
    video_path="my_drone_footage.mp4",
    model_name="yolov8n",  # Faster model
    confidence_threshold=0.5
)
app.run()
```

#### Hand Tracking App

```python
from heimdall import create_hand_app

# Minimal setup
app = create_hand_app()
app.run()

# Custom parameters
app = create_hand_app(
    camera_index=1,  # External camera
    max_num_hands=1,  # Single hand for performance
    min_detection_confidence=0.8  # High confidence
)
app.run()
```

### Advanced Configuration

#### Full Configuration Control

```python
from heimdall import HeimdallApp
from heimdall.config import AppConfig, DetectionConfig, HandConfig, LaserConfig, TrackerConfig

# Create custom detection configuration
detection_config = DetectionConfig(
    model_name="yolov8m-640",
    confidence_threshold=0.6,
    nms_threshold=0.4,
    drone_class_id=0,
    video_path="/path/to/my/video.mp4"
)

# Create custom tracker configuration
tracker_config = TrackerConfig(
    max_age=15,  # Keep tracks longer
    min_hits=5,  # Require more hits before confirming
    iou_threshold=0.2  # Stricter overlap threshold
)

# Create custom laser configuration
laser_config = LaserConfig(
    alive_port=45456,
    cmd_port=45457,
    data_port=45458
)

# Combine into app configuration
config = AppConfig(
    detection=detection_config,
    tracker=tracker_config,
    laser=laser_config,
    mode="detection"
)

# Run the app
app = HeimdallApp(config)
app.run()
```

#### Hand Tracking Configuration

```python
from heimdall import HeimdallApp
from heimdall.config import AppConfig, HandConfig

# High-performance hand tracking
hand_config = HandConfig(
    camera_index=0,
    max_num_hands=1,  # Single hand
    min_detection_confidence=0.8,  # High detection threshold
    min_tracking_confidence=0.7   # High tracking threshold
)

config = AppConfig(hand=hand_config, mode="hand")
app = HeimdallApp(config)
app.run()
```

### Component-Level Usage

#### Using Individual Components

```python
# Manual laser control
from heimdall.core.laser import LaserController, create_frame_from_bbox

def custom_frame_generator():
    # Generate custom laser patterns
    bbox = (100, 100, 300, 300)  # Fixed rectangle
    return create_frame_from_bbox(bbox, 1920, 1080)

controller = LaserController(custom_frame_generator)
controller.run()

# Manual object detection
from heimdall.core.detection import ObjectDetector

detector = ObjectDetector(
    model_name="yolov8n",
    confidence_threshold=0.5
)

detector.start_detection("my_video.mp4")

# Monitor detections
import time
for i in range(100):  # Monitor for 10 seconds
    bbox = detector.get_current_bbox()
    if bbox:
        x1, y1, x2, y2 = bbox
        print(f"Object detected at ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    time.sleep(0.1)

detector.stop_detection()

# Manual hand tracking
from heimdall.core.hand import HandTracker

tracker = HandTracker(max_num_hands=2)
tracker.start_tracking(camera_index=0)

# Monitor hand landmarks
import time
for i in range(100):
    landmarks = tracker.get_current_landmarks()
    print(f"Detected {len(landmarks)} hands")
    for i, hand in enumerate(landmarks):
        if hand:  # Hand has 21 landmarks
            wrist = hand[0]  # Wrist is landmark 0
            print(f"  Hand {i} wrist at: {wrist}")
    time.sleep(0.1)

tracker.stop_tracking()
```

## Configuration

### Model Selection Guide

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `yolov8n` | Fastest | Lowest | Real-time, low-power devices |
| `yolov8s` | Fast | Low | Good balance for most applications |
| `yolov8m` | Medium | Medium | **Recommended default** |
| `yolov8l` | Slow | High | High accuracy requirements |
| `yolov8x` | Slowest | Highest | Maximum accuracy, powerful hardware |

Add `-640` suffix (e.g., `yolov8m-640`) for 640x640 input resolution.

### Confidence Threshold Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| `0.1-0.3` | Many detections, some false positives | Detect all possible objects |
| `0.3-0.5` | **Balanced (recommended)** | General purpose |
| `0.5-0.7` | Fewer, more confident detections | Reduce false positives |
| `0.7-0.9` | Very conservative detection | Only very confident detections |

### Hand Tracking Parameters

| Parameter | Low Value | High Value | Recommendation |
|-----------|-----------|------------|----------------|
| `min_detection_confidence` | More responsive, less stable | More stable, less responsive | 0.5-0.7 for general use |
| `min_tracking_confidence` | Tracks through more motion | Loses tracking faster | 0.5 for active hands, 0.7 for steady hands |
| `max_num_hands` | Better performance | More hands detected | 1-2 for most applications |

## Examples

### Complete Example Scripts

#### High-Performance Detection

```python
#!/usr/bin/env python3
"""High-performance object detection with custom settings."""

from heimdall import create_detection_app

def main():
    # Configure for speed over accuracy
    app = create_detection_app(
        video_path="high_speed_drone.mp4",
        model_name="yolov8n",  # Fastest model
        confidence_threshold=0.2  # Lower threshold for speed
    )
    
    print("Starting high-speed detection...")
    try:
        app.run()
    except KeyboardInterrupt:
        print("Detection stopped by user")

if __name__ == "__main__":
    main()
```

#### Precision Hand Tracking

```python
#!/usr/bin/env python3
"""Precision hand tracking for detailed work."""

from heimdall import create_hand_app

def main():
    # Configure for maximum precision
    app = create_hand_app(
        camera_index=0,
        max_num_hands=1,  # Single hand for precision
        min_detection_confidence=0.9,  # Very high confidence
        min_tracking_confidence=0.8
    )
    
    print("Starting precision hand tracking...")
    print("Keep hand movements slow and deliberate")
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("Hand tracking stopped by user")

if __name__ == "__main__":
    main()
```

#### Custom Frame Generator

```python
#!/usr/bin/env python3
"""Custom laser pattern generator."""

import struct
import math
import time
from heimdall.core.laser import LaserController

def circle_pattern_generator():
    """Generate a moving circle pattern."""
    t = time.time()
    center_x = 2048 + int(1000 * math.sin(t))
    center_y = 2048 + int(1000 * math.cos(t))
    radius = 500
    
    points = []
    for i in range(100):  # 100 points for smooth circle
        angle = 2 * math.pi * i / 100
        x = center_x + int(radius * math.cos(angle))
        y = center_y + int(radius * math.sin(angle))
        # Red circle
        points.append(struct.pack('<HHHHH', x, y, 4095, 0, 0))
    
    return points

def main():
    controller = LaserController(circle_pattern_generator)
    
    print("Starting custom circle pattern...")
    print("Press Ctrl+C to stop")
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("Pattern stopped by user")

if __name__ == "__main__":
    main()
```

### Testing and Debug Examples

#### Camera Test

```bash
# Test different cameras
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"
```

#### Video File Test

```bash
# Test video file compatibility
python -c "
import cv2
cap = cv2.VideoCapture('test_data/visible.mp4')
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f'Video: {width}x{height}, {fps} FPS, {frames} frames')
    cap.release()
else:
    print('Could not open video file')
"
```

## Troubleshooting

### Common Issues and Solutions

#### "Video file not found"

```bash
# Check file exists
ls -la test_data/visible.mp4

# Use absolute path
python main.py --video "$(pwd)/test_data/visible.mp4"

# Check file permissions
chmod 644 test_data/visible.mp4
```

#### "Camera not accessible"

```bash
# List available cameras (Linux/Mac)
ls /dev/video*

# Check camera permissions (Linux)
sudo usermod -a -G video $USER
# Then logout and login again

# Test camera directly
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works:' if cap.isOpened() else 'Camera failed'); cap.release()"
```

#### "LaserCube not found"

1. **Check power**: Ensure LaserCube is powered on
2. **Check network**: Computer and LaserCube on same WiFi network
3. **Check firewall**: Allow Python through firewall
4. **Manual test**:
   ```bash
   # Test UDP broadcast
   python -c "
   import socket
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
   sock.sendto(bytes([0x77]), ('255.255.255.255', 45457))
   print('Broadcast sent')
   sock.close()
   "
   ```

#### Performance Issues

```bash
# Reduce model size
python main.py --model yolov8n

# Reduce confidence threshold (fewer calculations)
python main.py --confidence 0.5

# Single hand tracking
python main.py --mode hand --hands 1

# Monitor system resources
top -p $(pgrep -f python)
```

#### Import Errors

```bash
# Install missing dependencies
pip install -r requirements.txt

# Upgrade packages
pip install --upgrade ultralytics opencv-python mediapipe

# Check Python version
python --version  # Should be 3.8+

# Verify installations
python -c "import cv2, mediapipe, ultralytics; print('All imports successful')"
```

### Debug Mode

Enable detailed logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Then run your app
from heimdall import create_detection_app
app = create_detection_app()
app.run()
```

### Performance Monitoring

```python
import time
import psutil
from heimdall import create_detection_app

def monitor_performance():
    start_time = time.time()
    start_cpu = psutil.cpu_percent()
    start_memory = psutil.virtual_memory().percent
    
    app = create_detection_app()
    
    try:
        app.run()
    except KeyboardInterrupt:
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        
        print(f"Runtime: {end_time - start_time:.1f} seconds")
        print(f"CPU usage: {end_cpu:.1f}%")
        print(f"Memory usage: {end_memory:.1f}%")

if __name__ == "__main__":
    monitor_performance()
```

## Advanced Usage Patterns

### Batch Processing

```python
import os
from heimdall import create_detection_app

def process_video_directory(video_dir):
    """Process all videos in a directory."""
    for filename in os.listdir(video_dir):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dir, filename)
            print(f"Processing {filename}...")
            
            app = create_detection_app(
                video_path=video_path,
                confidence_threshold=0.4
            )
            
            try:
                app.run()
            except KeyboardInterrupt:
                print(f"Skipped {filename}")
                continue

# Usage
process_video_directory("./videos/")
```

### Integration with Other Systems

```python
from heimdall.core.detection import ObjectDetector
import json
import time

def log_detections_to_file():
    """Log detection results to JSON file."""
    detector = ObjectDetector(confidence_threshold=0.5)
    detector.start_detection("surveillance.mp4")
    
    detections_log = []
    start_time = time.time()
    
    try:
        while True:
            bbox = detector.get_current_bbox()
            timestamp = time.time() - start_time
            
            if bbox:
                detection = {
                    "timestamp": timestamp,
                    "bbox": bbox,
                    "confidence": "detected"
                }
            else:
                detection = {
                    "timestamp": timestamp,
                    "bbox": None,
                    "confidence": "none"
                }
            
            detections_log.append(detection)
            time.sleep(0.1)  # 10 FPS logging
            
    except KeyboardInterrupt:
        # Save results
        with open("detection_log.json", "w") as f:
            json.dump(detections_log, f, indent=2)
        print(f"Saved {len(detections_log)} detection records")
    
    finally:
        detector.stop_detection()

# Usage
log_detections_to_file()
```

This completes the comprehensive usage guide for Vigdis-Heimdall v2!