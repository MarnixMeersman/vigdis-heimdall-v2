#!/usr/bin/env python3
"""
Test version of UAV tracker with display only (no laser hardware required).
"""

import sys
import argparse
import cv2
import numpy as np
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from heimdall.core.detection import ObjectDetector


def main():
    """Test UAV tracking with real-time display."""
    parser = argparse.ArgumentParser(description='Test UAV Tracker Display')
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
    
    print("UAV Tracker Display Test")
    print("========================")
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.confidence}")
    print("Press 'q' to quit, 'p' to pause/resume")
    print()
    
    # Initialize detector
    detector = ObjectDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
        target_class_ids=[4, 14, 33]  # airplane, bird, kite
    )
    
    # Setup video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    
    # Display window
    cv2.namedWindow('UAV Tracker Test', cv2.WINDOW_RESIZABLE)
    
    frame_count = 0
    detection_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Run detection
                detections = detector._run_inference(frame)
                detections = detections.with_nms(detector.nms_threshold)
                
                # Filter by target class IDs and confidence
                class_mask = np.isin(detections.class_id, detector.target_class_ids)
                detections = detections[class_mask]
                detections = detections[detections.confidence >= detector.confidence_threshold]
                
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
                    
                    # Track largest detection
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
                
                # Add status overlay
                status_text = f"Frame: {frame_count} | Detections: {len(detections.xyxy)}"
                laser_status = "LASER TARGET: Active" if best_bbox else "LASER TARGET: None"
                
                cv2.putText(annotated_frame, status_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, laser_status, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          (0, 255, 0) if best_bbox else (128, 128, 128), 2)
                
                # Show laser target coordinates if available
                if best_bbox:
                    x1, y1, x2, y2 = best_bbox
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    # Convert to laser coordinates (normalized)
                    laser_x = center_x / frame_width
                    laser_y = center_y / frame_height
                    laser_coord_text = f"Laser: ({laser_x:.3f}, {laser_y:.3f})"
                    cv2.putText(annotated_frame, laser_coord_text, (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display frame
                cv2.imshow('UAV Tracker Test', annotated_frame)
                
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
                time.sleep(1.0 / fps)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames with {detection_count} total detections")


if __name__ == "__main__":
    main()