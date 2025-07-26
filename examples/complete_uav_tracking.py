#!/usr/bin/env python3
"""
Complete UAV Tracking Pipeline with Live Plotting and Video Annotation

This comprehensive example demonstrates the full Heimdall UAV tracking pipeline:
1. Load and configure machine learning models for object detection
2. Apply SORT tracker for multi-object tracking with persistent IDs
3. Live plot UAV coordinates in real-time
4. Save annotated video with tracking information
5. Export tracking data for analysis

Features:
- Configurable YOLO models (YOLOv8n, YOLOv8m, YOLOv8x, YOLOv11)
- SORT tracker parameter tuning
- Real-time coordinate plotting
- Video annotation
- Data export (CSV, JSON)
- Performance metrics

Usage:
    python complete_uav_tracking.py --video test_data/videoplayback.mp4 --model models/yolov8n
    python complete_uav_tracking.py --video test_data/videoplayback.mp4 --model models/yolov8m --confidence 0.5
    python complete_uav_tracking.py --help  # Show all configuration options

Requirements:
    - Input video file
    - YOLO model file in models/ directory
    - matplotlib for live plotting
    - All heimdall.core dependencies
"""

import sys
import argparse
import csv
import json
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from threading import Thread, Event
import queue

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall.core.detection import ObjectDetector
from heimdall.core.tracker import Sort


class UAVTrackingPipeline:
    """
    Complete UAV tracking pipeline with visualization and data export.
    
    This class encapsulates the entire tracking workflow from video input
    to annotated output, including real-time visualization and data logging.
    """
    
    def __init__(self, 
                 video_path: str,
                 output_path: str = None,
                 model_name: str = "models/yolov8n",
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.5,
                 target_classes: List[int] = None,
                 tracker_max_age: int = 30,
                 tracker_min_hits: int = 3,
                 tracker_iou_threshold: float = 0.3,
                 enable_gpu: bool = True,
                 live_plot: bool = True,
                 plot_window_size: int = 100):
        """
        Initialize the UAV tracking pipeline.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output video (auto-generated if None)
            model_name: YOLO model to use (e.g., 'models/yolov8n', 'models/yolov8m')
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            nms_threshold: Non-maximum suppression threshold (0.0-1.0)
            target_classes: List of COCO class IDs to track (None for auto-detect)
            tracker_max_age: SORT max age parameter (frames to keep inactive tracks)
            tracker_min_hits: SORT min hits parameter (detections needed to confirm track)
            tracker_iou_threshold: SORT IoU threshold for track association
            enable_gpu: Enable GPU acceleration (Apple Metal/CUDA)
            live_plot: Show live coordinate plotting
            plot_window_size: Number of recent points to show in live plot
        """
        
        # Store configuration
        self.video_path = Path(video_path)
        self.output_path = Path(output_path) if output_path else self.video_path.with_suffix('_annotated.mp4')
        self.model_name = model_name
        self.live_plot = live_plot
        self.plot_window_size = plot_window_size
        
        # Validate input
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps
        
        print(f"📹 Video Properties:")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.1f}")
        print(f"   Duration: {self.duration:.1f}s ({self.total_frames} frames)")
        
        # Initialize object detector with detailed configuration
        print(f"\n🤖 Initializing Object Detector:")
        print(f"   Model: {model_name}")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   NMS threshold: {nms_threshold}")
        print(f"   GPU acceleration: {enable_gpu}")
        
        self.detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            target_class_ids=target_classes,
            use_gpu=enable_gpu,
            prefer_ultralytics=True
        )
        
        # Initialize SORT tracker with detailed configuration
        print(f"\n🎯 Initializing SORT Tracker:")
        print(f"   Max age: {tracker_max_age} frames")
        print(f"   Min hits: {tracker_min_hits}")
        print(f"   IoU threshold: {tracker_iou_threshold}")
        
        self.tracker = Sort(
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
            iou_threshold=tracker_iou_threshold
        )
        
        # Initialize data storage
        self.tracking_data = defaultdict(list)  # track_id -> list of (frame, x, y, w, h, confidence)
        self.frame_data = []  # frame-by-frame statistics
        self.track_colors = {}  # consistent colors for each track ID
        
        # Initialize live plotting if enabled
        if self.live_plot:
            self.setup_live_plot()
            
        # Performance tracking
        self.processing_times = []
        self.detection_counts = []
        self.track_counts = []
        
    def setup_live_plot(self):
        """Initialize matplotlib for live plotting of UAV coordinates."""
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coordinate plot
        self.ax1.set_title('UAV Coordinates (Live)')
        self.ax1.set_xlabel('X Coordinate (pixels)')
        self.ax1.set_ylabel('Y Coordinate (pixels)')
        self.ax1.set_xlim(0, self.width)
        self.ax1.set_ylim(self.height, 0)  # Flip Y axis for image coordinates
        self.ax1.grid(True, alpha=0.3)
        
        # Statistics plot
        self.ax2.set_title('Tracking Statistics')
        self.ax2.set_xlabel('Frame')
        self.ax2.set_ylabel('Count')
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize plot data structures
        self.plot_data = defaultdict(lambda: {'x': deque(maxlen=self.plot_window_size), 
                                            'y': deque(maxlen=self.plot_window_size)})
        self.stats_frames = deque(maxlen=self.plot_window_size)
        self.stats_detections = deque(maxlen=self.plot_window_size)
        self.stats_tracks = deque(maxlen=self.plot_window_size)
        
    def update_live_plot(self, frame_num: int, tracks: np.ndarray, detections_count: int):
        """Update the live plot with new tracking data."""
        if not self.live_plot:
            return
            
        # Update coordinate data
        for track in tracks:
            track_id = int(track[4])
            center_x = (track[0] + track[2]) / 2
            center_y = (track[1] + track[3]) / 2
            
            self.plot_data[track_id]['x'].append(center_x)
            self.plot_data[track_id]['y'].append(center_y)
        
        # Update statistics
        self.stats_frames.append(frame_num)
        self.stats_detections.append(detections_count)
        self.stats_tracks.append(len(tracks))
        
        # Clear and redraw plots
        if frame_num % 5 == 0:  # Update every 5 frames to reduce overhead
            self.ax1.clear()
            self.ax2.clear()
            
            # Redraw coordinate plot
            self.ax1.set_title('UAV Coordinates (Live)')
            self.ax1.set_xlabel('X Coordinate (pixels)')
            self.ax1.set_ylabel('Y Coordinate (pixels)')
            self.ax1.set_xlim(0, self.width)
            self.ax1.set_ylim(self.height, 0)
            self.ax1.grid(True, alpha=0.3)
            
            for track_id, data in self.plot_data.items():
                if len(data['x']) > 0:
                    color = self.get_track_color(track_id)
                    self.ax1.plot(list(data['x']), list(data['y']), 
                                color=color, marker='o', markersize=3, alpha=0.7, 
                                label=f'Track {track_id}')
                    # Mark current position
                    if len(data['x']) > 0:
                        self.ax1.plot(data['x'][-1], data['y'][-1], 
                                    color=color, marker='o', markersize=8, 
                                    markeredgecolor='white', markeredgewidth=2)
            
            if self.plot_data:
                self.ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Redraw statistics plot
            self.ax2.set_title('Tracking Statistics')
            self.ax2.set_xlabel('Frame')
            self.ax2.set_ylabel('Count')
            self.ax2.grid(True, alpha=0.3)
            
            if len(self.stats_frames) > 0:
                self.ax2.plot(list(self.stats_frames), list(self.stats_detections), 
                            'b-', label='Detections', alpha=0.7)
                self.ax2.plot(list(self.stats_frames), list(self.stats_tracks), 
                            'r-', label='Active Tracks', alpha=0.7)
                self.ax2.legend()
            
            plt.tight_layout()
            plt.pause(0.001)
    
    def get_track_color(self, track_id: int) -> Tuple[float, float, float]:
        """Get consistent color for a track ID."""
        if track_id not in self.track_colors:
            # Generate a distinctive color based on track ID
            np.random.seed(track_id)  # Consistent color for same ID
            self.track_colors[track_id] = tuple(np.random.rand(3))
        return self.track_colors[track_id]
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Process a single frame through the detection and tracking pipeline.
        
        Args:
            frame: Input video frame
            frame_num: Frame number for tracking
            
        Returns:
            Tuple of (annotated_frame, tracks, processing_time)
        """
        start_time = time.time()
        
        # Run object detection
        detections = self.detector._run_inference(frame)
        detections = detections.with_nms(self.detector.nms_threshold)
        
        # Filter detections by target classes and confidence
        if len(detections.xyxy) > 0:
            class_mask = np.isin(detections.class_id, self.detector.target_class_ids)
            detections = detections[class_mask]
            detections = detections[detections.confidence >= self.detector.confidence_threshold]
        
        # Prepare detections for SORT tracker
        if len(detections.xyxy) > 0:
            detection_boxes = detections.xyxy
            confidence_scores = detections.confidence if detections.confidence is not None else np.array([0.5] * len(detection_boxes))
            sort_detections = np.column_stack([detection_boxes, confidence_scores])
        else:
            sort_detections = np.empty((0, 5))
        
        # Update tracker
        tracks = self.tracker.update(sort_detections)
        
        processing_time = time.time() - start_time
        
        # Store tracking data
        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = track[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            confidence = np.mean(confidence_scores) if len(confidence_scores) > 0 else 0.0
            
            self.tracking_data[track_id].append({
                'frame': frame_num,
                'timestamp': frame_num / self.fps,
                'center_x': float(center_x),
                'center_y': float(center_y),
                'width': float(width),
                'height': float(height),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(confidence)
            })
        
        # Store frame statistics
        self.frame_data.append({
            'frame': frame_num,
            'timestamp': frame_num / self.fps,
            'detections': len(detections.xyxy),
            'tracks': len(tracks),
            'processing_time': processing_time
        })
        
        # Create annotated frame
        annotated_frame = self.annotate_frame(frame, tracks, detections, frame_num, processing_time)
        
        # Update live plot
        self.update_live_plot(frame_num, tracks, len(detections.xyxy))
        
        return annotated_frame, tracks, processing_time
    
    def annotate_frame(self, frame: np.ndarray, tracks: np.ndarray, detections, 
                      frame_num: int, processing_time: float) -> np.ndarray:
        """
        Create professionally annotated frame with tracking information.
        
        Args:
            frame: Original video frame
            tracks: SORT tracking results
            detections: Raw detection results
            frame_num: Current frame number
            processing_time: Time taken to process this frame
            
        Returns:
            Annotated frame with all visualization elements
        """
        annotated = frame.copy()
        
        # Draw tracks with consistent colors and IDs
        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Get consistent color for this track
            color_normalized = self.get_track_color(track_id)
            color_bgr = tuple(int(c * 255) for c in reversed(color_normalized))  # Convert to BGR
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Draw center point
            cv2.circle(annotated, (center_x, center_y), 6, color_bgr, -1)
            cv2.circle(annotated, (center_x, center_y), 8, (255, 255, 255), 2)
            
            # Draw track trail (last few positions)
            if track_id in self.tracking_data and len(self.tracking_data[track_id]) > 1:
                trail_points = []
                for i in range(max(0, len(self.tracking_data[track_id]) - 10), 
                             len(self.tracking_data[track_id])):
                    data_point = self.tracking_data[track_id][i]
                    trail_points.append((int(data_point['center_x']), int(data_point['center_y'])))
                
                # Draw trail
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)  # Fade effect
                    color_faded = tuple(int(c * alpha) for c in color_bgr)
                    cv2.line(annotated, trail_points[i-1], trail_points[i], color_faded, 2)
            
            # Prepare label information
            track_length = len(self.tracking_data[track_id]) if track_id in self.tracking_data else 0
            label_lines = [
                f"UAV #{track_id}",
                f"Pos: ({center_x}, {center_y})",
                f"Frames: {track_length}"
            ]
            
            # Calculate label background size
            label_height = 25
            max_width = 0
            for line in label_lines:
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                max_width = max(max_width, text_width)
            
            # Draw label background
            label_bg_y = max(0, y1 - len(label_lines) * label_height - 10)
            cv2.rectangle(annotated, 
                         (x1, label_bg_y), 
                         (x1 + max_width + 20, y1), 
                         color_bgr, -1)
            cv2.rectangle(annotated, 
                         (x1, label_bg_y), 
                         (x1 + max_width + 20, y1), 
                         (255, 255, 255), 2)
            
            # Draw label text
            for i, line in enumerate(label_lines):
                text_y = label_bg_y + (i + 1) * 20
                cv2.putText(annotated, line, (x1 + 10, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw frame information overlay
        info_bg_height = 120
        cv2.rectangle(annotated, (10, 10), (400, info_bg_height), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (400, info_bg_height), (255, 255, 255), 2)
        
        info_lines = [
            f"Frame: {frame_num}/{self.total_frames}",
            f"Time: {frame_num/self.fps:.1f}s / {self.duration:.1f}s",
            f"Detections: {len(detections.xyxy) if hasattr(detections, 'xyxy') else 0}",
            f"Active Tracks: {len(tracks)}",
            f"Processing: {processing_time*1000:.1f}ms"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(annotated, line, (20, 35 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def run_tracking(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Dict:
        """
        Run the complete tracking pipeline on the video.
        
        Args:
            start_frame: Frame to start processing from
            end_frame: Frame to stop processing at (None for end of video)
            
        Returns:
            Dictionary with processing statistics and results
        """
        if end_frame is None:
            end_frame = self.total_frames
        
        # Setup video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        
        # Setup progress tracking
        total_frames_to_process = end_frame - start_frame
        
        print(f"\n🚀 Starting UAV Tracking Pipeline:")
        print(f"   Processing frames {start_frame} to {end_frame}")
        print(f"   Output video: {self.output_path}")
        print(f"   Live plotting: {'Enabled' if self.live_plot else 'Disabled'}")
        print(f"\n📊 Progress:")
        
        # Seek to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        start_time = time.time()
        frames_processed = 0
        
        try:
            for frame_num in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if not ret:
                    print(f"⚠️  Warning: Could not read frame {frame_num}")
                    break
                
                # Process frame
                annotated_frame, tracks, processing_time = self.process_frame(frame, frame_num)
                
                # Write to output video
                video_writer.write(annotated_frame)
                
                # Update progress
                frames_processed += 1
                if frames_processed % 30 == 0 or frames_processed == total_frames_to_process:
                    progress = frames_processed / total_frames_to_process * 100
                    elapsed = time.time() - start_time
                    fps_actual = frames_processed / elapsed if elapsed > 0 else 0
                    eta = (total_frames_to_process - frames_processed) / fps_actual if fps_actual > 0 else 0
                    
                    print(f"   {progress:5.1f}% | Frame {frame_num:6d} | "
                          f"Tracks: {len(tracks):2d} | "
                          f"Speed: {fps_actual:5.1f} FPS | "
                          f"ETA: {eta:6.1f}s")
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Processing interrupted by user at frame {frame_num}")
        
        finally:
            # Cleanup
            video_writer.release()
            self.cap.release()
            if self.live_plot:
                plt.ioff()
            
            total_time = time.time() - start_time
            
            # Generate final statistics
            stats = self.generate_statistics(frames_processed, total_time)
            
            print(f"\n✅ Processing Complete!")
            print(f"   Frames processed: {frames_processed}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Average FPS: {frames_processed/total_time:.1f}")
            print(f"   Output saved: {self.output_path}")
            
            return stats
    
    def generate_statistics(self, frames_processed: int, total_time: float) -> Dict:
        """Generate comprehensive statistics from the tracking session."""
        stats = {
            'session_info': {
                'video_path': str(self.video_path),
                'output_path': str(self.output_path),
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat(),
                'frames_processed': frames_processed,
                'total_processing_time': total_time,
                'average_fps': frames_processed / total_time if total_time > 0 else 0
            },
            'tracking_summary': {
                'unique_tracks': len(self.tracking_data),
                'total_detections': sum(fd['detections'] for fd in self.frame_data),
                'average_detections_per_frame': np.mean([fd['detections'] for fd in self.frame_data]) if self.frame_data else 0,
                'average_tracks_per_frame': np.mean([fd['tracks'] for fd in self.frame_data]) if self.frame_data else 0,
                'average_processing_time': np.mean([fd['processing_time'] for fd in self.frame_data]) if self.frame_data else 0
            },
            'track_details': {}
        }
        
        # Individual track statistics
        for track_id, track_data in self.tracking_data.items():
            if len(track_data) > 0:
                positions = [(d['center_x'], d['center_y']) for d in track_data]
                distances = []
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    distances.append(np.sqrt(dx*dx + dy*dy))
                
                stats['track_details'][track_id] = {
                    'duration_frames': len(track_data),
                    'duration_seconds': len(track_data) / self.fps,
                    'first_seen_frame': track_data[0]['frame'],
                    'last_seen_frame': track_data[-1]['frame'],
                    'average_confidence': np.mean([d['confidence'] for d in track_data]),
                    'total_distance_pixels': sum(distances) if distances else 0,
                    'average_speed_pixels_per_frame': np.mean(distances) if distances else 0,
                    'bounding_box': {
                        'min_x': min(d['center_x'] for d in track_data),
                        'max_x': max(d['center_x'] for d in track_data),
                        'min_y': min(d['center_y'] for d in track_data),
                        'max_y': max(d['center_y'] for d in track_data)
                    }
                }
        
        return stats
    
    def export_data(self, csv_path: Optional[str] = None, json_path: Optional[str] = None):
        """
        Export tracking data to CSV and/or JSON formats.
        
        Args:
            csv_path: Path for CSV export (auto-generated if None)
            json_path: Path for JSON export (auto-generated if None)
        """
        base_name = self.video_path.stem
        
        # Export to CSV
        if csv_path is None:
            csv_path = self.video_path.parent / f"{base_name}_tracking_data.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['track_id', 'frame', 'timestamp', 'center_x', 'center_y', 
                         'width', 'height', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for track_id, track_data in self.tracking_data.items():
                for data_point in track_data:
                    row = {
                        'track_id': track_id,
                        'frame': data_point['frame'],
                        'timestamp': data_point['timestamp'],
                        'center_x': data_point['center_x'],
                        'center_y': data_point['center_y'],
                        'width': data_point['width'],
                        'height': data_point['height'],
                        'bbox_x1': data_point['bbox'][0],
                        'bbox_y1': data_point['bbox'][1],
                        'bbox_x2': data_point['bbox'][2],
                        'bbox_y2': data_point['bbox'][3],
                        'confidence': data_point['confidence']
                    }
                    writer.writerow(row)
        
        # Export to JSON
        if json_path is None:
            json_path = self.video_path.parent / f"{base_name}_tracking_data.json"
        
        export_data = {
            'tracking_data': dict(self.tracking_data),
            'frame_statistics': self.frame_data,
            'video_info': {
                'path': str(self.video_path),
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'duration': self.duration
            }
        }
        
        with open(json_path, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, default=str)
        
        print(f"📁 Data exported:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")


def main():
    """Main function with comprehensive command-line interface."""
    parser = argparse.ArgumentParser(
        description='Complete UAV Tracking Pipeline with Live Plotting and Video Annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video test_data/videoplayback.mp4
  %(prog)s --video test_data/videoplayback.mp4 --model models/yolov8m --confidence 0.5
  %(prog)s --video test_data/videoplayback.mp4 --no-live-plot --tracker-max-age 50
  %(prog)s --video test_data/videoplayback.mp4 --start 30 --end 120 --export-data

Available YOLO Models:
  models/yolov8n     - Fastest, least accurate
  models/yolov8m     - Balanced speed/accuracy  
  models/yolov8x     - Slowest, most accurate
  models/yolo11n-seg - Latest YOLOv11 nano with segmentation

Target Classes (COCO):
  4  - Airplane
  14 - Bird  
  33 - Kite
        """)
    
    # Input/Output arguments
    parser.add_argument('--video', type=str, required=True,
                       help='Input video file path')
    parser.add_argument('--output', type=str,
                       help='Output video path (default: input_annotated.mp4)')
    parser.add_argument('--start', type=float, default=0,
                       help='Start time in seconds (default: 0)')
    parser.add_argument('--end', type=float,
                       help='End time in seconds (default: end of video)')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='models/yolov8n',
                       help='YOLO model to use (default: models/yolov8n)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Detection confidence threshold 0.0-1.0 (default: 0.3)')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                       help='Non-maximum suppression threshold (default: 0.5)')
    parser.add_argument('--target-classes', type=int, nargs='+',
                       help='COCO class IDs to track (default: auto-detect)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    
    # Tracker configuration
    parser.add_argument('--tracker-max-age', type=int, default=30,
                       help='SORT max age - frames to keep inactive tracks (default: 30)')
    parser.add_argument('--tracker-min-hits', type=int, default=3,
                       help='SORT min hits - detections needed to confirm track (default: 3)')
    parser.add_argument('--tracker-iou-threshold', type=float, default=0.3,
                       help='SORT IoU threshold for track association (default: 0.3)')
    
    # Visualization
    parser.add_argument('--no-live-plot', action='store_true',
                       help='Disable live coordinate plotting')
    parser.add_argument('--plot-window', type=int, default=100,
                       help='Number of recent points in live plot (default: 100)')
    
    # Data export
    parser.add_argument('--export-data', action='store_true',
                       help='Export tracking data to CSV and JSON')
    parser.add_argument('--csv-output', type=str,
                       help='CSV export path (default: auto-generated)')
    parser.add_argument('--json-output', type=str,
                       help='JSON export path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate arguments
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return 1
    
    print("🎬 Complete UAV Tracking Pipeline")
    print("=" * 50)
    print(f"Input video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    
    try:
        # Initialize pipeline
        pipeline = UAVTrackingPipeline(
            video_path=args.video,
            output_path=args.output,
            model_name=args.model,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms_threshold,
            target_classes=args.target_classes,
            tracker_max_age=args.tracker_max_age,
            tracker_min_hits=args.tracker_min_hits,
            tracker_iou_threshold=args.tracker_iou_threshold,
            enable_gpu=not args.no_gpu,
            live_plot=not args.no_live_plot,
            plot_window_size=args.plot_window
        )
        
        # Calculate frame range
        start_frame = int(args.start * pipeline.fps) if args.start else 0
        end_frame = int(args.end * pipeline.fps) if args.end else None
        
        # Run tracking
        stats = pipeline.run_tracking(start_frame, end_frame)
        
        # Export data if requested
        if args.export_data:
            pipeline.export_data(args.csv_output, args.json_output)
        
        # Print final summary
        print(f"\n📋 Final Summary:")
        print(f"   Unique tracks found: {stats['tracking_summary']['unique_tracks']}")
        print(f"   Total detections: {stats['tracking_summary']['total_detections']}")
        print(f"   Average processing time: {stats['tracking_summary']['average_processing_time']*1000:.1f}ms per frame")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())