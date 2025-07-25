"""
Hand tracking module using MediaPipe.
"""

import threading
import time
from typing import List, Tuple, Optional, Callable
import cv2
import mediapipe as mp
import struct


class HandTracker:
    """MediaPipe-based hand tracker."""
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize hand tracker.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Thread-safe state
        self.landmark_lock = threading.Lock()
        self.current_landmarks: List[List[Tuple[int, int]]] = []
        self.running = False
        self.tracking_thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()
        self.debug_frame: Optional[cv2.Mat] = None
    
    def start_tracking(self, camera_index: int = 0):
        """Start hand tracking in a separate thread."""
        if self.running:
            return
            
        self.running = True
        self.camera_index = camera_index
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
    
    def stop_tracking(self):
        """Stop hand tracking thread."""
        self.running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
    
    def get_current_landmarks(self) -> List[List[Tuple[int, int]]]:
        """Get current hand landmarks in thread-safe manner."""
        with self.landmark_lock:
            return self.current_landmarks.copy()
    
    def get_debug_frame(self) -> Optional[cv2.Mat]:
        """Get current debug frame with hand annotations."""
        with self.frame_lock:
            return self.debug_frame.copy() if self.debug_frame is not None else None
    
    def _tracking_loop(self):
        """Main hand tracking loop running in separate thread."""
        cap = cv2.VideoCapture(self.camera_index)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            hand_coords = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm_list = []
                    h, w, _ = frame.shape
                    for lm in hand_landmarks.landmark:
                        # Convert normalized coordinates to pixel values
                        x, y = int(lm.x * w), int(lm.y * h)
                        lm_list.append((x, y))
                    hand_coords.append(lm_list)
                    
                    # Draw landmarks on frame for debugging
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Update shared state
            with self.landmark_lock:
                self.current_landmarks = hand_coords
            
            with self.frame_lock:
                self.debug_frame = frame
            
            time.sleep(0.01)  # Small delay
        
        cap.release()


class HandFrameGenerator:
    """Frame generator that uses hand tracking results."""
    
    def __init__(self, hand_tracker: HandTracker, screen_width: int = None, screen_height: int = None):
        self.hand_tracker = hand_tracker
        # Get screen size if not provided
        if screen_width is None or screen_height is None:
            try:
                import pyautogui
                self.screen_width, self.screen_height = pyautogui.size()
            except ImportError:
                self.screen_width, self.screen_height = 1920, 1080
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height
    
    def __call__(self):
        """Generate frame data based on current hand landmarks.""" 
        landmarks = self.hand_tracker.get_current_landmarks()
        return self._create_frame_from_landmarks(landmarks)
    
    def _create_frame_from_landmarks(self, landmarks: List[List[Tuple[int, int]]]) -> List[bytes]:
        """Convert hand landmarks to laser frame data."""
        if not landmarks:
            # Center point when no hands detected
            return [struct.pack('<HHHHH', 2048, 2048, 4095, 4095, 4095)]
        
        frame = []
        
        # Process each detected hand
        for hand_landmarks in landmarks:
            if not hand_landmarks:
                continue
                
            # Convert landmarks to laser coordinates
            laser_points = []
            for x, y in hand_landmarks:
                # Normalize to screen coordinates
                nx = x / self.screen_width
                ny = y / self.screen_height
                
                # Convert to laser coordinates (0-4095)
                lx = int(nx * 4095)
                ly = int((1 - ny) * 4095)  # Flip Y axis
                laser_points.append((lx, ly))
            
            # Create smooth lines between key landmarks
            key_connections = [
                # Thumb
                (4, 3), (3, 2), (2, 1), (1, 0),
                # Index finger
                (8, 7), (7, 6), (6, 5), (5, 0),
                # Middle finger
                (12, 11), (11, 10), (10, 9), (9, 0),
                # Ring finger
                (16, 15), (15, 14), (14, 13), (13, 0),
                # Pinky
                (20, 19), (19, 18), (18, 17), (17, 0)
            ]
            
            for start_idx, end_idx in key_connections:
                if start_idx < len(laser_points) and end_idx < len(laser_points):
                    start_x, start_y = laser_points[start_idx]
                    end_x, end_y = laser_points[end_idx]
                    
                    # Interpolate between points
                    steps = 50
                    for i in range(steps):
                        t = i / steps
                        x = int(start_x + (end_x - start_x) * t)
                        y = int(start_y + (end_y - start_y) * t)
                        frame.append(struct.pack('<HHHHH', x, y, 4095, 0, 0))  # Red color
        
        return frame if frame else [struct.pack('<HHHHH', 2048, 2048, 4095, 4095, 4095)]


def create_hand_tracker_with_camera(camera_index: int = 0) -> Tuple[HandTracker, Callable]:
    """
    Convenience function to create hand tracker and frame generator for camera input.
    
    Returns:
        Tuple of (hand_tracker, frame_generator_function)
    """
    tracker = HandTracker()
    
    # Start tracking on the camera
    tracker.start_tracking(camera_index)
    
    # Create frame generator
    frame_generator = HandFrameGenerator(tracker)
    
    return tracker, frame_generator