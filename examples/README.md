# Heimdall Examples

This directory contains comprehensive, well-documented examples demonstrating the full capabilities of the Heimdall laser projection and object tracking system.

## 📋 Available Examples

### 1. Complete UAV Tracking (`complete_uav_tracking.py`)

**The most comprehensive UAV tracking pipeline with live plotting and video annotation.**

**Features:**
- Machine learning model selection and configuration
- SORT tracker for multi-object tracking with persistent IDs
- Real-time coordinate plotting with matplotlib
- Professional video annotation with tracking information
- Data export (CSV, JSON) for analysis
- Performance metrics and statistics
- Configurable detection and tracking parameters

**Usage:**
```bash
# Basic usage
python complete_uav_tracking.py --video test_data/videoplayback.mp4

# With custom model and confidence
python complete_uav_tracking.py --video test_data/videoplayback.mp4 --model models/yolov8m --confidence 0.5

# Time range processing
python complete_uav_tracking.py --video test_data/videoplayback.mp4 --start 30 --end 120

# Export tracking data
python complete_uav_tracking.py --video test_data/videoplayback.mp4 --export-data
```

**Key Features:**
- Support for multiple YOLO models (YOLOv8n, YOLOv8m, YOLOv8x, YOLOv11)
- SORT tracker parameter tuning
- Live coordinate plotting
- Professional video annotation
- Performance monitoring
- Data export capabilities

---

### 2. Laser Coordinate Control (`laser_coordinate_control.py`)

**Direct laser coordinate control with power management and pattern generation.**

**Features:**
- Precise coordinate positioning (0-4095 range)
- Power level control (0-100%) with safety limits
- RGB color control for multi-color effects
- Pre-defined patterns (circles, squares, spirals, etc.)
- Interactive GUI and command-line interfaces
- Calibration tools for alignment
- Safety features and limits

**Usage:**
```bash
# Interactive GUI mode
python laser_coordinate_control.py --interactive

# Fixed position
python laser_coordinate_control.py --coordinates 2048 2048 --power 25

# Pattern drawing
python laser_coordinate_control.py --pattern circle --power 15

# Calibration mode
python laser_coordinate_control.py --calibrate
```

**Available Patterns:**
- `circle` - Circular motion patterns
- `square` - Square/rectangular patterns  
- `figure_eight` - Figure-8 Lissajous patterns
- `spiral` - Spiral in/out patterns
- `rainbow` - Color-changing patterns
- `calibration` - Alignment and setup patterns

---

### 3. Hand Tracking to Laser (`hands_to_laser_webcam.py`)

**Real-time hand tracking with laser projection via webcam.**

**Features:**
- Multi-hand tracking (up to 2 hands simultaneously)
- Multiple projection modes:
  - Single point (index finger tip)
  - Hand outline (all landmarks)  
  - Drawing/writing mode with trails
  - Gesture-based control
- Hand gesture recognition for commands
- Calibration for accurate mapping
- Recording and playback capabilities
- Real-time performance monitoring

**Usage:**
```bash
# Default point tracking mode
python hands_to_laser_webcam.py

# Hand outline mode
python hands_to_laser_webcam.py --mode outline

# Drawing mode with recording
python hands_to_laser_webcam.py --mode drawing --record session1

# Calibration
python hands_to_laser_webcam.py --calibrate
```

**Hand Gestures:**
- 👊 **Fist**: Laser off
- 👆 **Point**: Single point mode
- ✋ **Open hand**: Hand outline mode
- ✌️ **Peace**: Drawing mode
- 👍 **Thumbs up**: Increase laser power
- 👎 **Thumbs down**: Decrease laser power

---

### 4. Mouse-Controlled Laser (`mouse_controlled_laser.py`)

**Intuitive mouse control for laser projection with multiple control modes.**

**Features:**
- Direct mouse position to laser coordinate mapping
- Multiple control modes:
  - Direct mode: Mouse position = laser position
  - Drawing mode: Click and drag to draw
  - Precision mode: Scaled movement for fine control
  - Gesture mode: Mouse gestures for commands
- Multi-monitor support with coordinate mapping
- Mouse button controls for power and color
- Recording and playback of mouse sessions
- Interactive GUI and headless modes

**Usage:**
```bash
# GUI mode (default)
python mouse_controlled_laser.py

# Drawing mode
python mouse_controlled_laser.py --mode drawing

# Precision mode with reduced sensitivity
python mouse_controlled_laser.py --mode precision --precision 0.5

# Headless mode
python mouse_controlled_laser.py --headless

# Calibration
python mouse_controlled_laser.py --calibrate
```

**Mouse Controls:**
- **Move**: Control laser position
- **Left click**: Enable laser / start drawing
- **Right click**: Disable laser / stop drawing
- **Middle click**: Change color
- **Scroll wheel**: Adjust laser power
- **Shift + move**: Precision mode
- **Ctrl + click**: Add calibration point

---

## 🚀 Getting Started

### Prerequisites

1. **Hardware Requirements:**
   - LaserCube laser projector (connected to network)
   - Webcam (for hand tracking)
   - Mouse (for mouse control)

2. **Software Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional Dependencies:**
   ```bash
   # For advanced calibration
   pip install opencv-python
   
   # For GUI interfaces  
   pip install tkinter
   
   # For mouse control
   pip install pyautogui
   ```

### Basic Setup

1. **Ensure LaserCube is connected** to your network and discoverable
2. **Test video file** - Place test videos in `test_data/` directory
3. **Camera setup** - Ensure webcam is working for hand tracking examples
4. **Safety first** - Always start with low power settings (<20%)

### Quick Start Examples

```bash
# 1. Test UAV tracking on a video
cd examples
python complete_uav_tracking.py --video ../test_data/videoplayback.mp4 --model models/yolov8n

# 2. Interactive laser control
python laser_coordinate_control.py --interactive

# 3. Hand tracking (requires webcam)
python hands_to_laser_webcam.py --calibrate
python hands_to_laser_webcam.py --mode point

# 4. Mouse control
python mouse_controlled_laser.py --calibrate  
python mouse_controlled_laser.py --mode drawing
```

---

## 🔧 Configuration and Customization

### Model Configuration

All examples support configurable machine learning models:

**Available YOLO Models:**
- `models/yolov8n` - Fastest, least accurate
- `models/yolov8m` - Balanced speed/accuracy
- `models/yolov8x` - Slowest, most accurate  
- `models/yolo11n-seg` - Latest YOLOv11 with segmentation

**Target Classes (COCO dataset):**
- Class 4: Airplane
- Class 14: Bird
- Class 33: Kite

### Tracker Configuration

SORT tracker parameters can be tuned:
- `max_age`: Frames to keep inactive tracks (default: 30)
- `min_hits`: Detections needed to confirm track (default: 3)
- `iou_threshold`: IoU threshold for association (default: 0.3)

### Laser Configuration

Common laser settings:
- **Power range**: 0-100% (safety limits apply)
- **Coordinate range**: 0-4095 for both X and Y
- **Color**: RGB values 0-255 each
- **Update rate**: 30 Hz default (adjustable)

---

## 📊 Data Export and Analysis

### Tracking Data Export

The UAV tracking example exports comprehensive data:

**CSV Format** (`*_tracking_data.csv`):
- Frame-by-frame tracking data
- Track IDs, coordinates, confidence scores
- Bounding box information
- Timestamps

**JSON Format** (`*_tracking_data.json`):
- Complete session metadata
- Hierarchical tracking data
- Video properties and settings
- Performance statistics

### Session Recording

Hand and mouse examples support session recording:
- **JSON format** with timestamped events
- **Playback capability** for repeated demonstrations
- **Analysis tools** for gesture/movement patterns

---

## 🛡️ Safety Guidelines

### Laser Safety

⚠️ **IMPORTANT SAFETY INFORMATION**

1. **Power Limits**: Always start with low power (<20%)
2. **Beam Path**: Ensure safe projection area before operation
3. **Eye Safety**: Never direct laser toward people or reflective surfaces
4. **Supervision**: Always supervise laser operation
5. **Regulations**: Follow local laser safety regulations
6. **Emergency**: Know how to quickly disable laser output

### Best Practices

1. **Calibration**: Always calibrate before first use
2. **Testing**: Test with low power in safe environment
3. **Monitoring**: Monitor performance and temperature
4. **Maintenance**: Regular cleaning and inspection
5. **Documentation**: Keep logs of usage and settings

---

## 🐛 Troubleshooting

### Common Issues

**1. LaserCube Not Found**
- Check network connection
- Verify LaserCube is powered on
- Check firewall settings
- Try manual IP configuration

**2. Poor Tracking Performance**
- Adjust confidence thresholds
- Try different YOLO models
- Improve lighting conditions
- Reduce video resolution if needed

**3. Hand Tracking Issues**
- Ensure good lighting
- Clean camera lens
- Adjust detection confidence
- Try different camera positions

**4. Mouse Control Not Responsive**
- Check mouse permissions
- Verify pyautogui installation
- Try different control modes
- Calibrate mouse-to-laser mapping

### Performance Optimization

1. **GPU Acceleration**: Enable for faster processing
2. **Model Selection**: Balance speed vs. accuracy
3. **Resolution**: Lower resolution for better FPS
4. **Threading**: Examples use multithreading for performance
5. **Calibration**: Proper calibration improves accuracy

---

## 📝 Example-Specific Documentation

Each example includes comprehensive built-in help:

```bash
# View detailed help for any example
python complete_uav_tracking.py --help
python laser_coordinate_control.py --help  
python hands_to_laser_webcam.py --help
python mouse_controlled_laser.py --help
```

### Advanced Usage Patterns

**Chaining Examples:**
1. Use `complete_uav_tracking.py` to analyze videos and export data
2. Use `laser_coordinate_control.py` to replay tracking patterns  
3. Use `hands_to_laser_webcam.py` for interactive demonstrations
4. Use `mouse_controlled_laser.py` for precise manual control

**Development Workflow:**
1. Start with `laser_coordinate_control.py` to verify hardware setup
2. Use calibration modes to establish accurate coordinate mapping
3. Test with low-power settings before increasing intensity
4. Record sessions for analysis and demonstration purposes

---

## 🤝 Contributing

These examples serve as both demonstrations and starting points for custom applications. Key extension points:

1. **Custom Patterns**: Add new patterns to `laser_coordinate_control.py`
2. **Gesture Recognition**: Extend gesture sets in `hands_to_laser_webcam.py`
3. **Tracking Algorithms**: Integrate new trackers in `complete_uav_tracking.py`
4. **Control Modes**: Add new control modes to `mouse_controlled_laser.py`

### Code Style

All examples follow consistent patterns:
- Comprehensive docstrings and comments
- Configuration through command-line arguments
- Extensive error handling and safety checks
- Performance monitoring and statistics
- Clean separation of concerns

---

## 📞 Support

For technical support:
1. Check the troubleshooting section above
2. Review example-specific help (`--help` flag)
3. Examine log output for error details
4. Verify hardware connections and settings
5. Test with minimal configurations first

---

*Last updated: $(date)*
*Examples tested with Heimdall v2.0*