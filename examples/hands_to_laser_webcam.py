#!/usr/bin/env python3
"""
Hand Tracking to Laser Projection via Webcam

This example demonstrates real-time hand tracking with laser projection:
1. Capture video from webcam using OpenCV
2. Track hand landmarks using MediaPipe
3. Map hand positions to laser coordinates
4. Project hand movements onto surfaces using LaserCube
5. Multiple interaction modes and gesture recognition
6. Real-time visualization and debugging

Features:
- Multi-hand tracking (up to 2 hands simultaneously)
- Configurable hand-to-laser mapping modes:
  * Single point (index finger tip)
  * Hand outline (all landmarks)
  * Gesture-based drawing
  * Writing/drawing mode
- Calibration for different projection surfaces
- Hand gesture recognition for control
- Real-time performance monitoring
- Recording and playback capabilities

Usage:
    python hands_to_laser_webcam.py                    # Default mode
    python hands_to_laser_webcam.py --camera 1         # Use camera 1
    python hands_to_laser_webcam.py --mode outline     # Hand outline mode
    python hands_to_laser_webcam.py --calibrate        # Calibration mode
    python hands_to_laser_webcam.py --record session1  # Record session

Hand Gestures:
- Closed fist: Laser off
- Index finger pointing: Single point mode
- Open hand: Hand outline mode
- Peace sign: Drawing mode
- Thumbs up: Increase laser power
- Thumbs down: Decrease laser power

Requirements:
    - Webcam or USB camera
    - LaserCube hardware
    - Good lighting for hand tracking
    - MediaPipe dependencies
"""

import sys
import argparse
import cv2
import json
import numpy as np
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall.core.hand import HandTracker
from heimdall.core.laser import LaserController
import struct


class HandPoint(NamedTuple):
    """Represents a hand landmark point."""
    x: int
    y: int
    landmark_id: int
    hand_id: int


class ProjectionMode(Enum):
    """Different laser projection modes."""
    POINT = "point"          # Single point (fingertip)
    OUTLINE = "outline"      # Hand outline
    DRAWING = "drawing"      # Drawing/writing mode
    GESTURES = "gestures"    # Gesture-based control


class GestureType(Enum):
    """Recognized hand gestures."""
    UNKNOWN = "unknown"
    FIST = "fist"
    POINT = "point"
    OPEN_HAND = "open_hand"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


class HandToLaserController:
    """
    Advanced hand tracking to laser projection controller.
    
    This class handles the complete pipeline from webcam input to laser output,
    including hand tracking, gesture recognition, coordinate mapping, and
    laser control with multiple projection modes.
    """
    
    def __init__(self,
                 camera_index: int = 0,
                 projection_mode: ProjectionMode = ProjectionMode.POINT,
                 max_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 laser_power: float = 15.0,
                 smooth_factor: float = 0.7,
                 projection_area: Tuple[int, int, int, int] = (0, 0, 4095, 4095),
                 enable_gestures: bool = True):
        """
        Initialize the hand-to-laser controller.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            projection_mode: Laser projection mode
            max_hands: Maximum number of hands to track
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking  
            laser_power: Initial laser power (0-100%)
            smooth_factor: Smoothing factor for laser movement (0-1)
            projection_area: Laser coordinate bounds (x1, y1, x2, y2)
            enable_gestures: Enable gesture recognition
        """
        
        # Configuration
        self.camera_index = camera_index
        self.projection_mode = projection_mode
        self.laser_power = laser_power
        self.smooth_factor = smooth_factor
        self.projection_area = projection_area
        self.enable_gestures = enable_gestures
        
        # State tracking
        self.current_laser_x = 2048  # Center
        self.current_laser_y = 2048  # Center
        self.laser_enabled = False
        self.drawing_trail = deque(maxlen=100)  # For drawing mode
        self.gesture_history = deque(maxlen=10)  # For gesture stability
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = deque(maxlen=30)
        
        # Recording
        self.recording = False
        self.recorded_session = []
        
        print(f"🖐️  Hand-to-Laser Controller Initialized:")
        print(f"   Camera: {camera_index}")
        print(f"   Projection mode: {projection_mode.value}")
        print(f"   Max hands: {max_hands}")
        print(f"   Detection confidence: {min_detection_confidence}")
        print(f"   Tracking confidence: {min_tracking_confidence}")
        print(f"   Laser power: {laser_power}%")
        print(f"   Projection area: {projection_area}")
        print(f"   Gestures enabled: {enable_gestures}")
        
        # Initialize hand tracker
        self.hand_tracker = HandTracker(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize laser controller
        self.laser_controller = LaserController(self.generate_laser_frame)
        
        # Calibration data
        self.calibration_points = []  # For mapping webcam to laser coordinates
        self.calibration_matrix = None
        self.load_calibration()
        
    def load_calibration(self):
        """Load calibration data from file."""
        calib_file = Path("hand_laser_calibration.json")
        if calib_file.exists():
            try:
                with open(calib_file, 'r') as f:
                    data = json.load(f)
                    self.calibration_points = data.get('points', [])
                    if len(self.calibration_points) >= 4:
                        self.calculate_calibration_matrix()
                        print("✅ Calibration loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load calibration: {e}")
    
    def save_calibration(self):
        """Save calibration data to file."""
        calib_file = Path("hand_laser_calibration.json")
        try:
            data = {
                'points': self.calibration_points,
                'timestamp': datetime.now().isoformat(),
                'projection_area': self.projection_area
            }
            with open(calib_file, 'w') as f:
                json.dump(data, f, indent=2)
            print("✅ Calibration saved successfully")
        except Exception as e:
            print(f"❌ Could not save calibration: {e}")
    
    def calculate_calibration_matrix(self):
        """Calculate transformation matrix from webcam to laser coordinates."""
        if len(self.calibration_points) < 4:
            return
        
        # Extract source and destination points
        src_points = np.array([[p['webcam_x'], p['webcam_y']] for p in self.calibration_points], dtype=np.float32)
        dst_points = np.array([[p['laser_x'], p['laser_y']] for p in self.calibration_points], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        self.calibration_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        print(f"🎯 Calibration matrix calculated from {len(self.calibration_points)} points")
    
    def webcam_to_laser_coords(self, webcam_x: int, webcam_y: int, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """
        Convert webcam coordinates to laser coordinates.
        
        Args:
            webcam_x: X coordinate in webcam frame
            webcam_y: Y coordinate in webcam frame
            frame_width: Width of webcam frame
            frame_height: Height of webcam frame
            
        Returns:
            Tuple of (laser_x, laser_y) coordinates
        """
        if self.calibration_matrix is not None:
            # Use calibration matrix for accurate mapping
            point = np.array([[[webcam_x, webcam_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.calibration_matrix)
            laser_x = int(transformed[0][0][0])
            laser_y = int(transformed[0][0][1])
        else:
            # Simple proportional mapping
            x_ratio = webcam_x / frame_width
            y_ratio = webcam_y / frame_height
            
            # Map to projection area
            area_width = self.projection_area[2] - self.projection_area[0]
            area_height = self.projection_area[3] - self.projection_area[1]
            
            laser_x = int(self.projection_area[0] + x_ratio * area_width)
            laser_y = int(self.projection_area[1] + y_ratio * area_height)
        
        # Clamp to valid range
        laser_x = max(0, min(4095, laser_x))
        laser_y = max(0, min(4095, laser_y))
        
        return laser_x, laser_y
    
    def recognize_gesture(self, hand_landmarks: List[Tuple[int, int]]) -> GestureType:
        """
        Recognize hand gesture from landmarks.
        
        Args:
            hand_landmarks: List of (x, y) landmark coordinates
            
        Returns:
            Recognized gesture type
        """
        if not self.enable_gestures or len(hand_landmarks) < 21:
            return GestureType.UNKNOWN
        
        # Extract key landmarks
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        index_tip = hand_landmarks[8]
        index_pip = hand_landmarks[6]
        middle_tip = hand_landmarks[12]
        middle_pip = hand_landmarks[10]
        ring_tip = hand_landmarks[16] 
        ring_pip = hand_landmarks[14]
        pinky_tip = hand_landmarks[20]
        pinky_pip = hand_landmarks[18]
        wrist = hand_landmarks[0]
        
        # Calculate finger extensions
        fingers_up = []
        
        # Thumb (different logic due to orientation)
        if thumb_tip[0] > thumb_ip[0]:  # Right hand
            fingers_up.append(thumb_tip[0] > thumb_ip[0])
        else:  # Left hand
            fingers_up.append(thumb_tip[0] < thumb_ip[0])
        
        # Other fingers
        fingers_up.append(index_tip[1] < index_pip[1])  # Index
        fingers_up.append(middle_tip[1] < middle_pip[1])  # Middle
        fingers_up.append(ring_tip[1] < ring_pip[1])     # Ring
        fingers_up.append(pinky_tip[1] < pinky_pip[1])   # Pinky
        
        # Gesture recognition
        total_fingers = sum(fingers_up)
        
        if total_fingers == 0:
            return GestureType.FIST
        elif total_fingers == 1 and fingers_up[1]:  # Only index finger
            return GestureType.POINT
        elif total_fingers == 2 and fingers_up[1] and fingers_up[2]:  # Index and middle
            return GestureType.PEACE
        elif total_fingers >= 4:  # Most fingers up
            return GestureType.OPEN_HAND
        elif total_fingers == 1 and fingers_up[0]:  # Only thumb
            # Check if thumb is up or down
            if thumb_tip[1] < wrist[1]:
                return GestureType.THUMBS_UP
            else:
                return GestureType.THUMBS_DOWN
        
        return GestureType.UNKNOWN
    
    def get_stable_gesture(self, current_gesture: GestureType) -> GestureType:
        """Get stable gesture by filtering out noise."""
        self.gesture_history.append(current_gesture)
        
        # Count occurrences of each gesture in recent history
        gesture_counts = {}
        for gesture in self.gesture_history:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Return most common gesture if it appears frequently enough
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        if most_common[1] >= 3:  # Gesture must appear at least 3 times
            return most_common[0]
        
        return GestureType.UNKNOWN
    
    def process_gesture_command(self, gesture: GestureType):
        """Process gesture as a control command."""
        if gesture == GestureType.FIST:
            self.laser_enabled = False
        elif gesture == GestureType.POINT:
            self.projection_mode = ProjectionMode.POINT
            self.laser_enabled = True
        elif gesture == GestureType.OPEN_HAND:
            self.projection_mode = ProjectionMode.OUTLINE
            self.laser_enabled = True
        elif gesture == GestureType.PEACE:
            self.projection_mode = ProjectionMode.DRAWING
            self.laser_enabled = True
        elif gesture == GestureType.THUMBS_UP:
            self.laser_power = min(30.0, self.laser_power + 5.0)
            print(f"⚡ Power increased to {self.laser_power:.1f}%")
        elif gesture == GestureType.THUMBS_DOWN:
            self.laser_power = max(0.0, self.laser_power - 5.0)
            print(f"⚡ Power decreased to {self.laser_power:.1f}%")
    
    def generate_laser_frame(self) -> List[bytes]:
        """
        Generate laser frame data based on current hand tracking.
        
        Returns:
            List of binary laser point data
        """
        frame = []
        
        if not self.laser_enabled:
            # Return center point with no power when disabled
            return [struct.pack('<HHHHH', 2048, 2048, 0, 0, 0)]
        
        # Convert power percentage to 12-bit value
        power_12bit = int((self.laser_power / 100.0) * 4095)
        
        if self.projection_mode == ProjectionMode.POINT:
            # Single point mode - reduced points to prevent buffer overflow
            num_points = min(10, max(5, power_12bit // 500))  # Much fewer points
            for _ in range(num_points):
                frame.append(struct.pack('<HHHHH', 
                                        self.current_laser_x, 
                                        self.current_laser_y,
                                        power_12bit, power_12bit, power_12bit))
        
        elif self.projection_mode == ProjectionMode.DRAWING:
            # Drawing mode with trail - limit trail length to prevent buffer overflow
            max_trail_points = 20  # Limit trail length
            if len(self.drawing_trail) > 1:
                # Use only the most recent points
                recent_trail = list(self.drawing_trail)[-max_trail_points:]
                for i, (x, y) in enumerate(recent_trail):
                    # Fade older points
                    fade_factor = (i + 1) / len(recent_trail)
                    faded_power = int(power_12bit * fade_factor)
                    
                    frame.append(struct.pack('<HHHHH', x, y, 
                                           faded_power, faded_power, faded_power))
            else:
                # Current point only
                frame.append(struct.pack('<HHHHH',
                                       self.current_laser_x,
                                       self.current_laser_y,
                                       power_12bit, power_12bit, power_12bit))
        
        return frame if frame else [struct.pack('<HHHHH', 2048, 2048, 0, 0, 0)]
    
    def process_hands(self, hand_landmarks_list: List[List[Tuple[int, int]]], 
                     frame_width: int, frame_height: int) -> List[HandPoint]:
        """
        Process detected hands and update laser coordinates.
        
        Args:
            hand_landmarks_list: List of hand landmark lists
            frame_width: Width of camera frame
            frame_height: Height of camera frame
            
        Returns:
            List of processed hand points
        """
        processed_points = []
        
        if not hand_landmarks_list:
            self.laser_enabled = False
            return processed_points
        
        # Process first hand (primary)
        primary_hand = hand_landmarks_list[0]
        
        # Recognize gesture
        gesture = self.recognize_gesture(primary_hand)
        stable_gesture = self.get_stable_gesture(gesture)
        
        # Process gesture commands
        if stable_gesture != GestureType.UNKNOWN:
            self.process_gesture_command(stable_gesture)
        
        # Get target coordinates based on mode
        if self.projection_mode == ProjectionMode.POINT:
            # Use index finger tip (landmark 8)
            if len(primary_hand) > 8:
                finger_tip = primary_hand[8]
                target_x, target_y = self.webcam_to_laser_coords(
                    finger_tip[0], finger_tip[1], frame_width, frame_height)
                
                # Smooth movement
                self.current_laser_x = int(self.current_laser_x * self.smooth_factor + 
                                         target_x * (1 - self.smooth_factor))
                self.current_laser_y = int(self.current_laser_y * self.smooth_factor + 
                                         target_y * (1 - self.smooth_factor))
                
                processed_points.append(HandPoint(self.current_laser_x, self.current_laser_y, 8, 0))
        
        elif self.projection_mode == ProjectionMode.DRAWING:
            # Drawing mode - use index finger with trail
            if len(primary_hand) > 8:
                finger_tip = primary_hand[8]
                target_x, target_y = self.webcam_to_laser_coords(
                    finger_tip[0], finger_tip[1], frame_width, frame_height)
                
                # Add to drawing trail
                if stable_gesture != GestureType.FIST:  # Don't draw when fist is closed
                    self.drawing_trail.append((target_x, target_y))
                    self.current_laser_x = target_x
                    self.current_laser_y = target_y
                else:
                    # Clear trail when fist is made
                    self.drawing_trail.clear()
                
                processed_points.append(HandPoint(target_x, target_y, 8, 0))
        
        elif self.projection_mode == ProjectionMode.OUTLINE:
            # Hand outline mode - use key landmarks
            key_landmarks = [4, 8, 12, 16, 20]  # Fingertips
            for landmark_id in key_landmarks:
                if len(primary_hand) > landmark_id:
                    point = primary_hand[landmark_id]
                    laser_x, laser_y = self.webcam_to_laser_coords(
                        point[0], point[1], frame_width, frame_height)
                    processed_points.append(HandPoint(laser_x, laser_y, landmark_id, 0))
        
        # Record session if enabled
        if self.recording:
            self.recorded_session.append({
                'timestamp': time.time(),
                'frame': self.frame_count,
                'hands': hand_landmarks_list,
                'gesture': stable_gesture.value,
                'laser_coords': (self.current_laser_x, self.current_laser_y),
                'laser_enabled': self.laser_enabled,
                'projection_mode': self.projection_mode.value
            })
        
        return processed_points
    
    def run_calibration_mode(self):
        """Run interactive calibration mode."""
        print("\n🎯 Starting Calibration Mode")
        print("Click on 4 corners of your projection area in this order:")
        print("1. Top-left")
        print("2. Top-right") 
        print("3. Bottom-right")
        print("4. Bottom-left")
        print("Press 's' to save calibration, 'r' to reset, 'q' to quit")
        
        self.calibration_points = []
        
        def mouse_callback(event, x, y, flags, param):
            _ = flags, param  # Suppress unused variable warnings
            if event == cv2.EVENT_LBUTTONDOWN and len(self.calibration_points) < 4:
                # Define corresponding laser coordinates for corners
                laser_coords = [
                    (0, 0),       # Top-left
                    (4095, 0),    # Top-right
                    (4095, 4095), # Bottom-right
                    (0, 4095)     # Bottom-left
                ]
                
                laser_x, laser_y = laser_coords[len(self.calibration_points)]
                self.calibration_points.append({
                    'webcam_x': x,
                    'webcam_y': y,
                    'laser_x': laser_x,
                    'laser_y': laser_y
                })
                
                print(f"Point {len(self.calibration_points)}: Webcam({x}, {y}) -> Laser({laser_x}, {laser_y})")
                
                if len(self.calibration_points) == 4:
                    self.calculate_calibration_matrix()
                    print("✅ All calibration points collected!")
        
        # Start hand tracking
        self.hand_tracker.start_tracking(self.camera_index)
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        try:
            while True:
                # Get debug frame from hand tracker
                frame = self.hand_tracker.get_debug_frame()
                if frame is not None:
                    # Draw calibration points
                    for i, point in enumerate(self.calibration_points):
                        cv2.circle(frame, (point['webcam_x'], point['webcam_y']), 10, (0, 0, 255), -1)
                        cv2.putText(frame, f"{i+1}", (point['webcam_x']+15, point['webcam_y']), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Draw instruction
                    if len(self.calibration_points) < 4:
                        corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
                        instruction = f"Click {corner_names[len(self.calibration_points)]}"
                        cv2.putText(frame, instruction, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Press 's' to save, 'r' to reset", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Calibration', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and len(self.calibration_points) == 4:
                    self.save_calibration()
                    break
                elif key == ord('r'):
                    self.calibration_points = []
                    print("🔄 Calibration reset")
                
                time.sleep(0.01)
        
        finally:
            self.hand_tracker.stop_tracking()
            cv2.destroyAllWindows()
    
    def run_main_loop(self):
        """Run the main hand tracking and laser projection loop."""
        print(f"\n🚀 Starting Hand-to-Laser Projection")
        print(f"Mode: {self.projection_mode.value}")
        print("Controls:")
        print("  👊 Fist: Laser off")
        print("  👆 Point: Single point mode")
        print("  ✋ Open hand: Hand outline mode")
        print("  ✌️  Peace: Drawing mode")
        print("  👍 Thumbs up: Increase power")
        print("  👎 Thumbs down: Decrease power")
        print("Press 'q' to quit, 'r' to start/stop recording")
        
        # Start hand tracking
        self.hand_tracker.start_tracking(self.camera_index)
        
        # Start laser controller in background thread
        laser_thread = threading.Thread(target=self.laser_controller.run, daemon=True)
        laser_thread.start()
        
        # Create display window
        cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                start_time = time.time()
                
                # Get hand tracking results
                hand_landmarks_list = self.hand_tracker.get_current_landmarks()
                debug_frame = self.hand_tracker.get_debug_frame()
                
                if debug_frame is not None:
                    frame_height, frame_width = debug_frame.shape[:2]
                    
                    # Process hands
                    processed_points = self.process_hands(hand_landmarks_list, frame_width, frame_height)
                    
                    # Draw laser points on frame
                    for point in processed_points:
                        # Convert laser coords back to webcam coords for visualization
                        webcam_x = int((point.x / 4095.0) * frame_width)
                        webcam_y = int((point.y / 4095.0) * frame_height)
                        
                        cv2.circle(debug_frame, (webcam_x, webcam_y), 8, (0, 0, 255), -1)
                        cv2.circle(debug_frame, (webcam_x, webcam_y), 12, (255, 255, 255), 2)
                    
                    # Draw UI overlay
                    overlay_color = (0, 255, 0) if self.laser_enabled else (0, 0, 255)
                    
                    # Status information
                    status_lines = [
                        f"Mode: {self.projection_mode.value}",
                        f"Laser: {'ON' if self.laser_enabled else 'OFF'}",
                        f"Power: {self.laser_power:.1f}%",
                        f"Hands: {len(hand_landmarks_list)}",
                        f"Position: ({self.current_laser_x}, {self.current_laser_y})"
                    ]
                    
                    for i, line in enumerate(status_lines):
                        cv2.putText(debug_frame, line, (10, 30 + i * 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
                    
                    # Recording indicator
                    if self.recording:
                        cv2.circle(debug_frame, (frame_width - 30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(debug_frame, "REC", (frame_width - 80, 38),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Performance info 
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    
                    cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, frame_height - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Hand Tracking', debug_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.recording = not self.recording
                    if self.recording:
                        self.recorded_session = []
                        print("🔴 Recording started")
                    else:
                        self.save_recorded_session()
                        print("⏹️  Recording stopped")
                elif key == ord('1'):
                    self.projection_mode = ProjectionMode.POINT
                    print(f"Mode changed to: {self.projection_mode.value}")
                elif key == ord('2'):
                    self.projection_mode = ProjectionMode.OUTLINE
                    print(f"Mode changed to: {self.projection_mode.value}")
                elif key == ord('3'):
                    self.projection_mode = ProjectionMode.DRAWING
                    print(f"Mode changed to: {self.projection_mode.value}")
                elif key == ord('+') or key == ord('='):
                    self.laser_power = min(100.0, self.laser_power + 5.0)
                    print(f"Power: {self.laser_power:.1f}%")
                elif key == ord('-'):
                    self.laser_power = max(0.0, self.laser_power - 5.0)
                    print(f"Power: {self.laser_power:.1f}%")
                elif key == ord(' '):
                    self.laser_enabled = not self.laser_enabled
                    print(f"Laser: {'ON' if self.laser_enabled else 'OFF'}")
                
                self.frame_count += 1
                time.sleep(0.01)  # Small delay for stability
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            # Cleanup
            self.hand_tracker.stop_tracking()
            cv2.destroyAllWindows()
            self.laser_controller.stop()
            
            # Print session statistics
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print(f"\n📊 Session Statistics:")
            print(f"   Total frames: {self.frame_count}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Projection mode: {self.projection_mode.value}")
            
            if self.recording:
                self.save_recorded_session()
    
    def save_recorded_session(self):
        """Save recorded session to file."""
        if not self.recorded_session:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_tracking_session_{timestamp}.json"
        
        try:
            session_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'duration_seconds': time.time() - self.start_time,
                    'frame_count': len(self.recorded_session),
                    'projection_mode': self.projection_mode.value,
                    'laser_power': self.laser_power
                },
                'frames': self.recorded_session
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            print(f"💾 Session saved: {filename}")
        except Exception as e:
            print(f"❌ Could not save session: {e}")


def main():
    """Main function with comprehensive command-line interface."""
    parser = argparse.ArgumentParser(
        description='Hand Tracking to Laser Projection via Webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Default mode
  %(prog)s --camera 1 --mode outline          # Camera 1, outline mode
  %(prog)s --calibrate                        # Calibration mode
  %(prog)s --record session1 --power 25       # Recording at 25%% power
  %(prog)s --no-gestures --mode drawing       # Drawing mode without gestures

Hand Gestures:
  👊 Fist          - Laser off
  👆 Point         - Single point mode  
  ✋ Open hand     - Hand outline mode
  ✌️  Peace        - Drawing mode
  👍 Thumbs up     - Increase laser power
  👎 Thumbs down   - Decrease laser power

Keyboard Controls:
  q    - Quit
  r    - Start/stop recording
  1-3  - Change projection mode
  +/-  - Adjust laser power
  Space - Toggle laser on/off
        """)
    
    # Camera and tracking
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--max-hands', type=int, default=2,
                       help='Maximum number of hands to track (default: 2)')
    parser.add_argument('--detection-confidence', type=float, default=0.7,
                       help='Hand detection confidence threshold (default: 0.7)')
    parser.add_argument('--tracking-confidence', type=float, default=0.5,
                       help='Hand tracking confidence threshold (default: 0.5)')
    
    # Projection settings
    parser.add_argument('--mode', type=str, 
                       choices=['point', 'outline', 'drawing', 'gestures'],
                       default='point', help='Projection mode (default: point)')
    parser.add_argument('--power', type=float, default=15.0,
                       help='Initial laser power percentage (default: 15%%)')
    parser.add_argument('--smooth', type=float, default=0.7,
                       help='Movement smoothing factor 0-1 (default: 0.7)')
    parser.add_argument('--projection-area', type=int, nargs=4,
                       metavar=('X1', 'Y1', 'X2', 'Y2'),
                       default=[0, 0, 4095, 4095],
                       help='Laser projection area bounds (default: full range)')
    
    # Features
    parser.add_argument('--no-gestures', action='store_true',
                       help='Disable gesture recognition')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run calibration mode')
    parser.add_argument('--record', type=str, metavar='SESSION_NAME',
                       help='Start recording session with given name')
    
    args = parser.parse_args()
    
    print("🖐️  Hand Tracking to Laser Projection")
    print("=" * 50)
    
    try:
        # Initialize controller
        controller = HandToLaserController(
            camera_index=args.camera,
            projection_mode=ProjectionMode(args.mode),
            max_hands=args.max_hands,
            min_detection_confidence=args.detection_confidence,
            min_tracking_confidence=args.tracking_confidence,
            laser_power=args.power,
            smooth_factor=args.smooth,
            projection_area=tuple(args.projection_area),
            enable_gestures=not args.no_gestures
        )
        
        # Handle different modes
        if args.calibrate:
            controller.run_calibration_mode()
        else:
            if args.record:
                controller.recording = True
                print(f"🔴 Recording enabled: {args.record}")
            
            controller.run_main_loop()
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Application interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())