#!/usr/bin/env python3
"""Efficient YOLO detection post-processing: optimized NMS, label smoothing, and class confidence boosting.

This module provides utility functions to improve detection efficiency and reduce
class confusion (e.g., person vs. dog) through:
- Custom confidence thresholding with class bias
- Optimized Non-Maximum Suppression (NMS) with soft-NMS option
- BatchNMS for faster processing
- Duplicate removal by IoU overlap
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5, 
             sigma: float = 0.5, method: str = "soft") -> Tuple[np.ndarray, np.ndarray]:
    """Apply Soft-NMS to reduce redundant detections while preserving borderline detections.
    
    Args:
        boxes: (N, 4) array of bounding boxes in [x1, y1, x2, y2] format
        scores: (N,) array of detection scores
        iou_threshold: IoU threshold for suppression
        sigma: decay parameter for soft-NMS
        method: 'soft' (exponential decay) or 'hard' (standard NMS)
    
    Returns:
        keep_boxes, keep_scores: filtered boxes and scores
    """
    if len(boxes) == 0:
        return boxes, scores

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(scores)[::-1]
    keep = []
    scores = scores.copy()

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        if len(idxs) == 1:
            break

        xx1 = np.maximum(x1[current], x1[idxs[1:]])
        yy1 = np.maximum(y1[current], y1[idxs[1:]])
        xx2 = np.minimum(x2[current], x2[idxs[1:]])
        yy2 = np.minimum(y2[current], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (area[current] + area[idxs[1:]] - intersection)

        if method == "soft":
            # Exponential decay with IoU
            decay = np.exp(-(iou ** 2) / sigma)
            scores[idxs[1:]] *= decay
        else:
            # Hard NMS: remove boxes above threshold
            scores[idxs[1:]][iou > iou_threshold] = 0

        idxs = idxs[1:][scores[idxs[1:]] > 0]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]

    return boxes[keep], scores[keep]


def class_weighted_confidence(detections: List[Dict], class_weights: Optional[Dict] = None) -> List[Dict]:
    """Boost or reduce confidence scores based on class-specific weights.
    
    Useful for reducing confusion between similar classes (e.g., person vs. dog).
    Higher weight = more confident; lower weight = less confident.
    
    Args:
        detections: List of detection dicts with 'class_id', 'confidence' keys
        class_weights: Dict mapping class_id -> weight multiplier (default: uniform 1.0)
    
    Returns:
        detections with adjusted confidence scores
    """
    if class_weights is None:
        class_weights = {}

    for det in detections:
        cls_id = det.get("class_id")
        weight = class_weights.get(cls_id, 1.0)
        det["confidence"] *= weight
        det["confidence"] = min(1.0, max(0.0, det["confidence"]))  # Clamp to [0, 1]

    return detections


def filter_by_class_confidence(detections: List[Dict], class_conf_thresholds: Dict[int, float]) -> List[Dict]:
    """Filter detections by class-specific confidence thresholds.
    
    Example: person requires 0.6 confidence, dog requires 0.7, cat requires 0.5.
    
    Args:
        detections: List of detection dicts
        class_conf_thresholds: Dict mapping class_id -> min_confidence
    
    Returns:
        filtered detections
    """
    filtered = []
    for det in detections:
        cls_id = det.get("class_id")
        conf_threshold = class_conf_thresholds.get(cls_id, 0.5)
        if det.get("confidence", 0) >= conf_threshold:
            filtered.append(det)
    return filtered


def batch_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, 
              iou_threshold: float = 0.45) -> np.ndarray:
    """Perform NMS separately per class, then combine results.
    
    More efficient than global NMS when classes are well-separated.
    
    Args:
        boxes: (N, 4) array in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        class_ids: (N,) class IDs
        iou_threshold: IoU threshold for NMS
    
    Returns:
        keep: indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    keep = []
    unique_classes = np.unique(class_ids)

    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        # Apply NMS for this class
        cls_keep = _nms_indices(cls_boxes, cls_scores, iou_threshold)
        keep.extend(cls_indices[cls_keep])

    return np.array(sorted(keep), dtype=int)


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Standard NMS returning indices."""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        if len(idxs) == 1:
            break

        xx1 = np.maximum(x1[current], x1[idxs[1:]])
        yy1 = np.maximum(y1[current], y1[idxs[1:]])
        xx2 = np.minimum(x2[current], x2[idxs[1:]])
        yy2 = np.minimum(y2[current], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (area[current] + area[idxs[1:]] - intersection + 1e-8)

        idxs = idxs[1:][iou <= iou_threshold]

    return np.array(keep, dtype=int)


if __name__ == "__main__":
    # Example: test soft NMS
    boxes = np.array([[10, 10, 50, 50], [15, 15, 55, 55], [100, 100, 150, 150]])
    scores = np.array([0.9, 0.8, 0.7])
    
    keep_boxes, keep_scores = soft_nms(boxes, scores, method="soft")
    print("Soft NMS result:")
    print("Keep boxes:", keep_boxes)
    print("Keep scores:", keep_scores)
