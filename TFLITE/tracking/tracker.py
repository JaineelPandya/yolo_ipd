"""
Tracking module - handles object tracking across frames using ByteTrack
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import config

logger = logging.getLogger(__name__)

@dataclass
class TrackedObject:
    """Represents a tracked object across frames"""
    track_id: int
    class_id: int
    class_name: str
    detections: List[Dict] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)  # Center points
    last_seen_frame: int = 0
    first_seen_frame: int = 0
    confidence_history: List[float] = field(default_factory=list)
    is_active: bool = True
    
    def add_detection(self, detection: Dict, frame_id: int):
        """Add new detection to track"""
        self.detections.append(detection)
        self.positions.append(self._get_center(detection['bbox']))
        self.confidence_history.append(detection['confidence'])
        self.last_seen_frame = frame_id
        
        if self.first_seen_frame == 0:
            self.first_seen_frame = frame_id
    
    def _get_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Get center coordinates of bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_last_detection(self) -> Dict:
        """Get most recent detection"""
        return self.detections[-1] if self.detections else None
    
    def get_last_position(self) -> Tuple[float, float]:
        """Get most recent position"""
        return self.positions[-1] if self.positions else (0, 0)
    
    def get_average_confidence(self) -> float:
        """Get average confidence across detections"""
        if not self.confidence_history:
            return 0.0
        return float(np.mean(self.confidence_history))
    
    def get_movement_distance(self) -> float:
        """Get total movement distance"""
        if len(self.positions) < 2:
            return 0.0
        
        total_distance = 0
        for i in range(1, len(self.positions)):
            x1, y1 = self.positions[i-1]
            x2, y2 = self.positions[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        return float(total_distance)


class ByteTracker:
    """ByteTrack-based tracker implementation"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, max_tracks: int = 100):
        """
        Initialize ByteTrack tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections to keep track
            max_tracks: Maximum simultaneous tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_tracks = max_tracks
        
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> Dict[int, TrackedObject]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dictionaries from detector
        
        Returns:
            Dictionary of active tracks
        """
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, detection in matched_tracks:
            if track_id in self.tracks:
                self.tracks[track_id].add_detection(detection, self.frame_count)
        
        # Create new tracks for unmatched detections
        for detection in detections:
            if not any(d[1] == detection for _, d in matched_tracks):
                self._create_new_track(detection)
        
        # Remove inactive tracks
        self._cleanup_old_tracks()
        
        return self._get_active_tracks()
    
    def _match_detections(self, detections: List[Dict]) -> List[Tuple[int, Dict]]:
        """Match detections to existing tracks using IoU"""
        matched = []
        used_detection_indices = set()
        
        # Try to match existing tracks with detections
        for track_id, track in list(self.tracks.items()):
            if not track.is_active:
                continue
            
            best_iou = 0
            best_detection_idx = -1
            last_bbox = track.get_last_detection()['bbox']
            
            # Find best matching detection
            for det_idx, detection in enumerate(detections):
                if det_idx in used_detection_indices:
                    continue
                
                # Only match same class
                if detection['class_id'] != track.class_id:
                    continue
                
                iou = self._calculate_iou(last_bbox, detection['bbox'])
                if iou > best_iou and iou > 0.3:  # IoU threshold
                    best_iou = iou
                    best_detection_idx = det_idx
            
            if best_detection_idx >= 0:
                matched.append((track_id, detections[best_detection_idx]))
                used_detection_indices.add(best_detection_idx)
        
        return matched
    
    def _create_new_track(self, detection: Dict):
        """Create new track for detection"""
        if len(self.tracks) >= self.max_tracks:
            return
        
        track_id = self.next_id
        self.next_id += 1
        
        track = TrackedObject(
            track_id=track_id,
            class_id=detection['class_id'],
            class_name=detection['class_name']
        )
        track.add_detection(detection, self.frame_count)
        self.tracks[track_id] = track
        
        logger.debug(f"Created new track: {track_id} ({detection['class_name']})")
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been detected recently"""
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            age = self.frame_count - track.last_seen_frame
            
            if age > self.max_age or len(track.detections) < self.min_hits:
                if age > self.max_age:
                    track.is_active = False
                del self.tracks[track_id]
    
    def _get_active_tracks(self) -> Dict[int, TrackedObject]:
        """Get only active tracks"""
        active = {}
        for track_id, track in self.tracks.items():
            if track.is_active and len(track.detections) >= self.min_hits:
                active[track_id] = track
        return active
    
    @staticmethod
    def _calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def get_tracks_by_class(self, class_name: str) -> Dict[int, TrackedObject]:
        """Get all tracks of a specific class"""
        return {
            tid: track for tid, track in self._get_active_tracks().items()
            if track.class_name == class_name
        }
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        active_tracks = self._get_active_tracks()
        return {
            "total_tracks": len(self.tracks),
            "active_tracks": len(active_tracks),
            "frame_count": self.frame_count,
            "avg_track_length": np.mean([len(t.detections) for t in active_tracks.values()]) 
                               if active_tracks else 0
        }


def create_tracker(tracker_type: str = "bytetrack", **kwargs) -> ByteTracker:
    """Factory function to create tracker"""
    if tracker_type == "bytetrack":
        return ByteTracker(
            max_age=kwargs.get('max_age', config.MAX_AGE),
            min_hits=kwargs.get('min_hits', config.MIN_HITS),
            max_tracks=kwargs.get('max_tracks', config.MAX_TRACKS)
        )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
