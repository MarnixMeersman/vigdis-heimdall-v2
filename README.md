# Vigdis-Heimdall v2

Vigdis-Heimdall v2 is a prototype Python application to control a LaserCube projector over WiFi while tracking objects from a USB camera. Hand positions are detected using a Roboflow model and smoothed with a lightweight SORT tracker.

The example app projects simple masks onto detected objects. Use the arrow keys to adjust laser power and cycle between red, green and blue output.

## Repository layout

- `heimdall/` – package containing reusable modules
  - `laser.py` – LaserCube network interface
  - `detection.py` – Roboflow detector wrapper
  - `sort.py` – lightweight SORT implementation for smoothing
  - `app.py` – high-level control loop combining camera input and laser output
- `main.py` – small entrypoint launching `heimdall.app`
- `test.py` – reference script from the LaserCube community for experimentation

## Getting Started

1. Install Python 3.8+.
2. Install dependencies:
   ```bash
   pip install pygame opencv-python roboflow filterpy
   ```
   Set the Roboflow model to use by defining environment variables:
   ```bash
   export ROBOFLOW_API_KEY=<your_key>
   export ROBOFLOW_MODEL=<workspace/project/version>
   ```
3. Connect your LaserCube and camera to the same network.
4. Run `python main.py` to start the prototype.

## Roadmap

- Improve mask projection quality and coordinate mapping
- Expose additional keyboard shortcuts and UI controls
- Package application for easier deployment

## Disclaimer

Operating a laser projector can be hazardous. Use appropriate safety precautions and comply with all regulations.
