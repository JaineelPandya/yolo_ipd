"""
Frame deduplication module - determines if a frame should be stored
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import config

logger = logging.getLogger(__name__)

@dataclass
class FrameEvaluationResult:
    """Result of frame evaluation for storage"""
    should_store: bool
    reason: str
    importance_score: float  # 0-1, higher = more important
    changes: List[str]  # List of what changed


class FrameDeduplicator:
    """Determines if a frame should be stored based on changes"""
    
    def __init__(self):
        """Initialize frame deduplicator"""
        self.last_frame = None
        self.last_detections = []
        self.frame_embedding = None
        self.embedding_model = None
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """Initialize embedding model for similarity comparison"""
        try:
            from torchvision import models, transforms
            self.embedding_model = models.resnet50(pretrained=True)
            self.embedding_model = self.embedding_model.eval()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            logger.info("✓ Embedding model loaded for frame deduplication")
        except ImportError:
            logger.warning("torchvision not available - frame deduplication will be basic")
            self.embedding_model = None
    
    def evaluate(self, frame: np.ndarray, detections: List[Dict], 
                 tracked_objects: Dict) -> FrameEvaluationResult:
        """
        Evaluate if frame should be stored
        
        Args:
            frame: Current frame
            detections: Detections in current frame
            tracked_objects: Currently tracked objects
        
        Returns:
            FrameEvaluationResult with decision and reason
        """
        changes = []
        importance_score = 0.0
        
        # Check 1: New detections or objects disappeared
        new_detections = self._check_new_detections(detections)
        if new_detections:
            changes.append("New detections")
            importance_score += 0.3
        
        disappeared_objects = self._check_disappeared_objects(tracked_objects)
        if disappeared_objects:
            changes.append("Objects disappeared")
            importance_score += 0.2
        
        # Check 2: Significant object movement
        position_changed = self._check_position_change(detections)
        if position_changed:
            changes.append("Objects moved significantly")
            importance_score += 0.25
        
        # Check 3: Confidence improvement
        confidence_improved = self._check_confidence_improvement(detections)
        if confidence_improved:
            changes.append("Detection confidence improved")
            importance_score += 0.15
        
        # Check 4: Small objects with good confidence (high importance for assistive use)
        small_objects = self._check_small_objects(detections)
        if small_objects:
            changes.append(f"Found small object: {small_objects[0]['class_name']}")
            importance_score += 0.3
        
        # Check 5: Scene/lighting change (using embedding)
        scene_changed = self._check_scene_change(frame)
        if scene_changed:
            changes.append("Scene/lighting changed")
            importance_score += 0.25
        
        # Update state
        self.last_detections = detections
        self.last_frame = frame.copy() if frame is not None else None
        
        # Decision: store if any significant change detected
        should_store = importance_score >= 0.25  # 25% confidence threshold
        
        reason = ", ".join(changes) if changes else "No significant changes"
        
        return FrameEvaluationResult(
            should_store=should_store,
            reason=reason,
            importance_score=min(importance_score, 1.0),
            changes=changes
        )
    
    def _check_new_detections(self, current_detections: List[Dict]) -> bool:
        """Check if there are new detections in this frame"""
        if not self.last_detections:
            return len(current_detections) > 0
        
        last_classes = {d['class_id'] for d in self.last_detections}
        current_classes = {d['class_id'] for d in current_detections}
        
        new_classes = current_classes - last_classes
        return len(new_classes) > 0
    
    def _check_disappeared_objects(self, tracked_objects: Dict) -> bool:
        """Check if tracked objects disappeared"""
        if not self.last_detections:
            return False
        
        # This is handled by tracking module mainly
        # Here we just check if count decreased significantly
        return len(tracked_objects) < len(self.last_detections)
    
    def _check_position_change(self, detections: List[Dict]) -> bool:
        """Check if objects moved significantly"""
        if not self.last_detections or not detections:
            return False
        
        if len(detections) != len(self.last_detections):
            return True
        
        # Compare positions for same class objects
        for curr_det in detections:
            for last_det in self.last_detections:
                if curr_det['class_id'] == last_det['class_id']:
                    # Calculate center position change
                    curr_center = self._get_bbox_center(curr_det['bbox'])
                    last_center = self._get_bbox_center(last_det['bbox'])
                    
                    distance = np.sqrt(
                        (curr_center[0] - last_center[0])**2 + 
                        (curr_center[1] - last_center[1])**2
                    )
                    
                    if distance > config.LOCATION_CHANGE_THRESHOLD:
                        return True
        
        return False
    
    def _check_confidence_improvement(self, detections: List[Dict]) -> bool:
        """Check if detection confidence improved"""
        if not self.last_detections or not detections:
            return False
        
        for curr_det in detections:
            for last_det in self.last_detections:
                if curr_det['class_id'] == last_det['class_id']:
                    confidence_gain = curr_det['confidence'] - last_det['confidence']
                    if confidence_gain > config.CONFIDENCE_IMPROVEMENT_THRESHOLD:
                        return True
        
        return False
    
    def _check_small_objects(self, detections: List[Dict]) -> List[Dict]:
        """Check for important small objects"""
        small_objects = []
        
        for det in detections:
            # Check if object is in tracking list
            if det['class_name'].lower() in config.SMALL_OBJECTS_TO_TRACK:
                if det['confidence'] > 0.3:  # Lowered from 0.6 for better coverage
                    small_objects.append(det)
        
        return small_objects
    
    def _check_scene_change(self, frame: np.ndarray) -> bool:
        """Check if scene changed using frame embedding"""
        if self.last_frame is None:
            return True
        
        if self.embedding_model is None:
            # Fallback: use histogram comparison
            return self._compare_histograms(self.last_frame, frame)
        
        try:
            import torch
            
            # Get embeddings
            emb1 = self._get_frame_embedding(self.last_frame)
            emb2 = self._get_frame_embedding(frame)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
            
            # Store for future comparison
            self.frame_embedding = emb2
            
            # If similarity is low, scene changed
            return similarity < config.SIMILARITY_THRESHOLD
        except Exception as e:
            logger.warning(f"Embedding comparison error: {e}")
            return False
    
    def _get_frame_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Get frame embedding using ResNet50"""
        if self.embedding_model is None:
            return None
        
        try:
            import torch
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Transform
            img_tensor = self.transform(frame_rgb).unsqueeze(0)
            
            # Get embedding
            with torch.no_grad():
                # Remove last layer and get feature vector
                embedding = self.embedding_model(img_tensor)
            
            return embedding
        except Exception as e:
            logger.warning(f"Embedding error: {e}")
            return None
    
    @staticmethod
    def _compare_histograms(frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Compare frames using histogram similarity (fallback method)"""
        # Convert to HSV for better color comparison
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 50], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 50], [0, 180, 0, 256])
        
        # Normalize
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Compare using correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return similarity < config.SIMILARITY_THRESHOLD
    
    @staticmethod
    def _get_bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
        """Get center of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def create_deduplicator() -> FrameDeduplicator:
    """Factory function to create deduplicator"""
    return FrameDeduplicator()
