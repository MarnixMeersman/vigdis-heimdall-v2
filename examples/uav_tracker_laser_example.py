#!/usr/bin/env python3
"""
Real-time UAV tracking with laser projection example.
Combines object detection, tracking, and laser cube projection.
"""

import sys
import argparse
import cv2
import numpy as np
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from heimdall.core.detection import ObjectDetector
from heimdall.core.laser import LaserController, create_frame_from_bbox


class UAVTrackerLaserExample:
    """Combined UAV tracker with laser projection and real-time display."""
    
    def __init__(self, video_path: str, model_name: str = "models/yolo11n-seg", 
                 confidence_threshold: float = 0.1):
        self.video_path = video_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Initialize detector
        self.detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            target_class_ids=[4, 14, 33]  # airplane, bird, kite
        )
        
        # Shared state for laser projection
        self.current_bbox = None
        self.bbox_lock = threading.Lock()
        self.running = True
        
        # Video properties
        self.cap = None
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
        
        # Setup laser controller
        self.laser_controller = LaserController(self.laser_frame_generator)
        
    def laser_frame_generator(self):
        """Generate laser frame data from current bounding box."""
        with self.bbox_lock:
            bbox = self.current_bbox
        return create_frame_from_bbox(bbox, self.frame_width, self.frame_height)
    
    def update_bbox(self, bbox: Optional[Tuple[float, float, float, float]]):
        """Thread-safe bbox update."""
        with self.bbox_lock:
            self.current_bbox = bbox
    
    def setup_video(self):
        """Initialize video capture."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
    
    def run_tracking_loop(self):
        """Main tracking loop with real-time display."""
        print("Starting UAV tracking with laser projection...")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        # Create window for display
        cv2.namedWindow('UAV Tracker + Laser', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        detection_count = 0
        paused = False
        
        # Start laser controller in separate thread
        laser_thread = threading.Thread(target=self.laser_controller.run, daemon=True)
        laser_thread.start()
        
        try:
            while self.running:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # Run detection
                    detections = self.detector._run_inference(frame)
                    detections = detections.with_nms(self.detector.nms_threshold)
                    
                    # Filter by target class IDs and confidence
                    class_mask = np.isin(detections.class_id, self.detector.target_class_ids)
                    detections = detections[class_mask]
                    detections = detections[detections.confidence >= self.detector.confidence_threshold]
                    
                    # Create annotated frame
                    annotated_frame = frame.copy()
                    
                    # Process detections
                    best_bbox = None
                    max_area = 0
                    
                    for i, bbox in enumerate(detections.xyxy):
                        x1, y1, x2, y2 = map(int, bbox)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Track largest detection for laser projection
                        if area > max_area:
                            max_area = area
                            best_bbox = (float(x1), float(y1), float(x2), float(y2))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw center point
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        
                        # Add confidence score
                        confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                        label = f"UAV {i+1} ({confidence:.2f})"
                        coord_text = f"({center_x}, {center_y})"
                        
                        # Label background
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        (coord_w, coord_h), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        
                        bg_width = max(label_w, coord_w) + 10
                        bg_height = label_h + coord_h + 15
                        
                        cv2.rectangle(annotated_frame, (x1, y1 - bg_height), 
                                    (x1 + bg_width, y1), (0, 255, 0), -1)
                        
                        # Draw text
                        cv2.putText(annotated_frame, label, (x1 + 5, y1 - coord_h - 8), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(annotated_frame, coord_text, (x1 + 5, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        
                        detection_count += 1
                    
                    # Update laser projection with best detection
                    self.update_bbox(best_bbox)
                    
                    # Add status overlay
                    status_text = f"Frame: {frame_count} | Detections: {len(detections.xyxy)}"
                    laser_status = "LASER: Active" if best_bbox else "LASER: Idle"
                    
                    cv2.putText(annotated_frame, status_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, laser_status, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 255, 0) if best_bbox else (128, 128, 128), 2)
                    
                    # Display frame
                    cv2.imshow('UAV Tracker + Laser', annotated_frame)
                    
                    frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                
                # Control frame rate
                if not paused:
                    time.sleep(1.0 / self.fps)
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            self.cleanup()
            print(f"\nProcessed {frame_count} frames with {detection_count} total detections")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.laser_controller.stop()
    
    def run(self):
        """Main entry point."""
        try:
            self.setup_video()
            self.run_tracking_loop()
        except Exception as e:
            print(f"Error: {e}")
            self.cleanup()


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='UAV Tracker with Laser Projection')
    parser.add_argument('--video', type=str, default='test_data/videoplayback.mp4',
                       help='Input video path')
    parser.add_argument('--model', type=str, default='models/yolo11n-seg',
                       help='YOLO model to use')
    parser.add_argument('--confidence', type=float, default=0.1,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Verify video file exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    print("UAV Tracker + Laser Projection")
    print("==============================")
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.confidence}")
    print()
    
    # Create and run tracker
    tracker = UAVTrackerLaserExample(
        video_path=args.video,
        model_name=args.model,
        confidence_threshold=args.confidence
    )
    
    tracker.run()


if __name__ == "__main__":
    main()