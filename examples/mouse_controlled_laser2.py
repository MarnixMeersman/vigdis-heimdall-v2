#!/usr/bin/env python3
"""
Simple Mouse-Controlled Laser (Using heimdall.core.laser)

A minimal example demonstrating how easy it is to control a LaserCube
using the heimdall.core.laser module. This script tracks your mouse
position and projects a laser point accordingly.

This version is much simpler than mouse_controlled_laser.py because it
uses the high-level LaserController and LaserPoint classes from the
heimdall package instead of implementing the protocol from scratch.

Usage:
    python mouse_controlled_laser2.py

Features:
- Automatic LaserCube discovery on the network
- Real-time mouse position tracking
- Smooth laser movement
- Automatic coordinate conversion
- Graceful shutdown on Ctrl+C

Requirements:
    - LaserCube hardware on the same network
    - pyautogui package for mouse tracking
    - heimdall package
"""

import logging
import time
from typing import List

# Import the simplified laser control classes
from heimdall.core.laser import LaserController, LaserPoint

# Laser settings
LASER_POWER = 1024  # 25% power (range: 0-4095)


def create_mouse_frame() -> List[LaserPoint]:
    """
    Generate a frame with laser points following the mouse cursor.
    
    Returns:
        List of LaserPoint objects representing the current frame
    """
    try:
        import pyautogui
        
        # Get screen dimensions and mouse position
        screen_width, screen_height = pyautogui.size()
        mouse_x, mouse_y = pyautogui.position()
        
        # Convert mouse position to laser coordinates (0-4095)
        # Match the coordinate system from mouse_controlled_laser.py
        laser_x = 4095 - int((mouse_x * 4095) / screen_width)
        laser_y = 4095 - int((mouse_y * 4095) / screen_height)
        
        # Create multiple points at the same position for better visibility
        frame = []
        for _ in range(50):  # 50 points per frame
            frame.append(LaserPoint(
                x=laser_x, 
                y=laser_y, 
                r=LASER_POWER,  # Red
                g=LASER_POWER,  # Green  
                b=LASER_POWER   # Blue (creates white light)
            ))
        
        return frame
        
    except ImportError:
        # Fallback if pyautogui is not installed - center point
        center_point = LaserPoint(2048, 2048, LASER_POWER, LASER_POWER, LASER_POWER)
        return [center_point] * 50
        
    except Exception as e:
        logging.error(f"Error getting mouse position: {e}")
        # Return empty frame on error
        return []


def main():
    """Main function to run the mouse-controlled laser."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔴 Mouse-Controlled Laser (heimdall.core.laser version)")
    print("=" * 60)
    print("This script will:")
    print("1. Discover LaserCube devices on your network")
    print("2. Track your mouse position")
    print("3. Project laser points at the mouse location")
    print(f"4. Use {LASER_POWER}/4095 power level ({(LASER_POWER/4095)*100:.1f}%)")
    print("\nPress Ctrl+C to stop safely.")
    print("=" * 60)
    
    # Create the laser controller with our mouse tracking function
    controller = LaserController(create_mouse_frame)
    
    try:
        # Start the controller (begins device discovery)
        print("\n🔍 Starting LaserCube discovery...")
        controller.start()
        
        # Wait for laser connection
        print("⏳ Waiting for LaserCube connection...")
        connected = False
        
        while not connected:
            lasers = controller.get_connected_lasers()
            if lasers:
                for laser in lasers:
                    if laser.info:
                        print(f"\n✅ Connected to: {laser.info}")
                        print(f"   Model: {laser.info.model_name}")
                        print(f"   Firmware: v{laser.info.fw_major}.{laser.info.fw_minor}")
                        print(f"   Battery: {laser.info.battery_percent}%")
                        print(f"   Temperature: {laser.info.temperature}°C")
                        connected = True
                        break
            
            if not connected:
                time.sleep(1)
        
        print("\n🎯 Laser ready! Move your mouse to control the laser position.")
        print("   The laser will follow your mouse cursor in real-time.")
        print("\nPress Ctrl+C to stop.")
        
        # Keep the main thread alive while the laser tracks the mouse
        while True:
            # Show connection status every 10 seconds
            time.sleep(10)
            connected_count = len(controller.get_connected_lasers())
            print(f"📡 Status: {connected_count} laser(s) connected")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping laser controller...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Unexpected error: {e}")
        
    finally:
        # Always stop the controller safely
        controller.stop()
        print("✅ Laser controller stopped safely.")


if __name__ == "__main__":
    main()