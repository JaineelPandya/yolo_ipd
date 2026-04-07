"""
Frame deduplication — lightweight histogram-based approach.
No ResNet50 or torchvision required; works on Raspberry Pi.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import config

logger = logging.getLogger(__name__)


@dataclass
class FrameEvaluationResult:
    """Result of frame evaluation for storage decision"""
    should_store: bool
    reason: str
    importance_score: float   # 0–1
    changes: List[str] = field(default_factory=list)


class FrameDeduplicator:
    """
    Decides whether a frame is worth storing by checking:
      1. New object classes detected
      2. Significant object displacement
      3. Confidence jump
      4. High-value small object present
      5. Scene change (HSV histogram comparison — no ML needed)
    """

    def __init__(self):
        self.last_frame: np.ndarray | None = None
        self.last_detections: List[Dict] = []
        self.last_hist: np.ndarray | None = None   # cached histogram

    # ------------------------------------------------------------------
    def evaluate(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        tracked_objects: Dict,
    ) -> FrameEvaluationResult:
        changes: List[str] = []
        score = 0.0

        # 1. New object class appeared
        if self._new_class(detections):
            changes.append("New object class detected")
            score += 0.35

        # 2. Tracked count dropped (something disappeared)
        if self._disappeared(tracked_objects):
            changes.append("Object disappeared from view")
            score += 0.20

        # 3. Object moved significantly
        if self._moved(detections):
            changes.append("Object moved significantly")
            score += 0.25

        # 4. Confidence improved
        if self._conf_improved(detections):
            changes.append("Detection confidence improved")
            score += 0.15

        # 5. High-value small object present
        small = self._small_objects(detections)
        if small:
            changes.append(f"Small object: {small[0]['class_name']}")
            score += 0.30

        # 6. Scene changed (histogram)
        if self._scene_changed(frame):
            changes.append("Scene/lighting changed")
            score += 0.25

        # Update state
        self.last_detections = list(detections)
        self.last_frame = frame.copy()
        self.last_hist = self._compute_hist(frame)

        should_store = score >= 0.25
        reason = ", ".join(changes) if changes else "No significant change"

        return FrameEvaluationResult(
            should_store=should_store,
            reason=reason,
            importance_score=min(score, 1.0),
            changes=changes,
        )

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _new_class(self, detections: List[Dict]) -> bool:
        last_ids = {d["class_id"] for d in self.last_detections}
        curr_ids = {d["class_id"] for d in detections}
        if not self.last_detections:
            return len(detections) > 0
        return bool(curr_ids - last_ids)

    def _disappeared(self, tracked_objects: Dict) -> bool:
        return len(tracked_objects) < len(self.last_detections)

    def _moved(self, detections: List[Dict]) -> bool:
        if not self.last_detections or not detections:
            return False
        if len(detections) != len(self.last_detections):
            return True
        for curr in detections:
            for last in self.last_detections:
                if curr["class_id"] == last["class_id"]:
                    c1 = self._centre(curr["bbox"])
                    c2 = self._centre(last["bbox"])
                    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    if dist > config.LOCATION_CHANGE_THRESHOLD:
                        return True
        return False

    def _conf_improved(self, detections: List[Dict]) -> bool:
        if not self.last_detections or not detections:
            return False
        for curr in detections:
            for last in self.last_detections:
                if curr["class_id"] == last["class_id"]:
                    if curr["confidence"] - last["confidence"] > config.CONFIDENCE_IMPROVEMENT_THRES:
                        return True
        return False

    def _small_objects(self, detections: List[Dict]) -> List[Dict]:
        return [
            d for d in detections
            if d["class_name"].lower() in config.SMALL_OBJECTS_TO_TRACK
            and d["confidence"] > 0.30
        ]

    def _scene_changed(self, frame: np.ndarray) -> bool:
        if self.last_hist is None:
            return True
        curr_hist = self._compute_hist(frame)
        similarity = cv2.compareHist(self.last_hist, curr_hist, cv2.HISTCMP_CORREL)
        return similarity < config.SIMILARITY_THRESHOLD

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hist(frame: np.ndarray) -> np.ndarray:
        """Compute normalised HSV histogram (H+S channels)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    @staticmethod
    def _centre(bbox: np.ndarray) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


def create_deduplicator() -> FrameDeduplicator:
    """Factory function"""
    return FrameDeduplicator()
