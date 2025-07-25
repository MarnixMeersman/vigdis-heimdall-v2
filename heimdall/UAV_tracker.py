import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
import time
from collections import defaultdict
import os

# SORT Tracker Implementation
class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        """
        # Define constant velocity model
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0],
                                             [0,0,1,0,0,0,0],
                                             [0,0,0,1,0,0,0]], np.float32)
        
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0],
                                           [0,1,0,0,0,1,0],
                                           [0,0,1,0,0,0,1],
                                           [0,0,0,1,0,0,0],
                                           [0,0,0,0,1,0,0],
                                           [0,0,0,0,0,1,0],
                                           [0,0,0,0,0,0,1]], np.float32)
        
        self.kf.measurementNoiseCov = 0.01 * np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov = 0.01 * np.eye(7, dtype=np.float32)
        self.kf.errorCovPost = 0.1 * np.eye(7, dtype=np.float32)
        
        self.kf.statePre = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0], np.float32)
        self.kf.statePost = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0], np.float32)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        measurement = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], np.float32)
        self.kf.correct(measurement)
        
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.statePost[6] + self.kf.statePost[2]) <= 0:
            self.kf.statePost[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.statePost.copy())
        return self.kf.statePost[:4].flatten()
        
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.statePost[:4].flatten()

def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem using Hungarian algorithm
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
      return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
      return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))
        
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold = 0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
            
        iou_matrix = iou_batch(detections, trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))
            
        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)
                
        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]]<iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class UAVTracker:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize UAV tracker with YOLO model and SORT tracker
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.track_history = defaultdict(list)
        
        # Colors for different tracks
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 128), (128, 128, 0), (255, 192, 203), (50, 205, 50)
        ]
        
    def detect_objects(self, frame):
        """
        Detect objects in frame using YOLO
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Filter by confidence
                    if conf >= self.confidence_threshold:
                        detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def draw_tracks(self, frame, tracks):
        """
        Draw bounding boxes, center points, and track histories
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            
            # Calculate center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Get color for this track
            color = self.colors[int(track_id) % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw track ID
            cv2.putText(frame, f'ID: {int(track_id)}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw coordinates
            coord_text = f'({center_x}, {center_y})'
            cv2.putText(frame, coord_text, 
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Store track history
            self.track_history[track_id].append((center_x, center_y))
            
            # Draw track trail (last 20 points)
            if len(self.track_history[track_id]) > 1:
                points = self.track_history[track_id][-20:]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
            
            # Print coordinates to terminal
            print(f"Track ID {int(track_id)}: Center=({center_x}, {center_y}), "
                  f"BBox=({x1}, {y1}, {x2}, {y2})")
        
        return frame
    
    def process_video(self, video_path, output_path=None, display_live=True):
        """
        Process video with UAV tracking
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("\nStarting UAV tracking...")
        print("Press 'q' to quit early\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Draw tracks and annotations
            annotated_frame = self.draw_tracks(frame.copy(), tracks)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(tracks)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display live video
            if display_live:
                cv2.imshow('UAV Tracker', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write to output video
            if out:
                out.write(annotated_frame)
            
            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({fps_current:.1f} FPS)")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {frame_count/elapsed_time:.2f}")
        
        if output_path:
            print(f"Output video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='UAV Coordinates Tracker with YOLO and SORT')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', default=None, 
                       help='Path to output video file (optional)')
    parser.add_argument('--model', '-m', default='yolov8n.pt',
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable live video display')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        return
    
    # Initialize tracker
    tracker = UAVTracker(model_path=args.model, confidence_threshold=args.confidence)
    
    # Set output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        args.output = f"{base_name}_tracked.mp4"
    
    # Process video
    tracker.process_video(
        video_path=args.video_path,
        output_path=args.output,
        display_live=not args.no_display
    )

if __name__ == "__main__":
    main