#!/usr/bin/env python3
"""
Test script to run only the object tracker without laser controller.
"""

import sys
import argparse
import cv2
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Test the tracker without laser."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test object tracker with video annotation')
    parser.add_argument('--start', type=float, default=0, help='Start time in minutes')
    parser.add_argument('--end', type=float, help='End time in minutes (optional)')
    parser.add_argument('--video', type=str, default='test_data/videoplayback.mp4', help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path (optional, defaults to input_annotated.mp4)')
    args = parser.parse_args()
    
    print("Testing Object Tracker Only")
    print("===========================")
    print(f"Video: {args.video}")
    
    # Fix: Handle the conditional formatting properly
    end_time_str = f"{args.end:.1f}" if args.end else "end"
    print(f"Time range: {args.start:.1f} - {end_time_str} minutes")
    
    # Setup video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Calculate frame range
    start_frame = int(args.start * 60 * fps)
    end_frame = int(args.end * 60 * fps) if args.end else total_frames
    end_frame = min(end_frame, total_frames)
    
    print(f"Processing frames {start_frame} to {end_frame} ({(end_frame-start_frame)/fps/60:.1f} minutes)")
    
    # Setup output video
    output_path = args.output or args.video.replace('.mp4', '_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create detector
    print("Creating detector...")
    from heimdall.core.detection import ObjectDetector
    detector = ObjectDetector(
        model_name="models/yolo11n-seg",
        confidence_threshold=0.1,
        target_class_ids= [4, 14, 33]
    )
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print("Starting detection and annotation...")
    print("Press Ctrl+C to stop")
    
    try:
        frame_count = 0
        detection_count = 0
        
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detections = detector._run_inference(frame)
            detections = detections.with_nms(detector.nms_threshold)
            # Filter by target class IDs
            import numpy as np
            class_mask = np.isin(detections.class_id, detector.target_class_ids)
            detections = detections[class_mask]
            detections = detections[detections.confidence >= detector.confidence_threshold]
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw detections
            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Add UAV label and coordinates
                label = f"UAV {i+1}"
                coord_text = f"({center_x}, {center_y})"
                
                # Label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                (coord_w, coord_h), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv2.rectangle(annotated_frame, (x1, y1-label_h-coord_h-10), (x1+max(label_w, coord_w)+10, y1), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(annotated_frame, label, (x1+5, y1-coord_h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(annotated_frame, coord_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                detection_count += 1
            
            # Write frame to output video
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps / 60
                print(f"Processed {frame_count} frames, {detection_count} detections, time: {current_time:.1f}m")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        out.release()
        print(f"\nProcessed {frame_count} frames with {detection_count} detections")
        print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    main()