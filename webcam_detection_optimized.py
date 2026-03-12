#!/usr/bin/env python3
"""Webcam detection with optimized real-time inference and efficient post-processing.

This script improves upon the basic webcam_detection.py by:
- Using soft-NMS to reduce false detections
- Applying class-specific confidence thresholds to reduce person/dog confusion
- Batch-processing frames when possible
- Displaying inference time and class statistics
"""

import argparse
import cv2
import sys
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from efficient_detection import soft_nms, batch_nms, filter_by_class_confidence


def main():
    parser = argparse.ArgumentParser(description="Optimized real-time webcam detection with YOLOv8")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Base confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference: 'cpu' or GPU ID")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--save-dir", type=str, default="runs/webcam_optimized", help="Save directory for frames")
    parser.add_argument("--use-soft-nms", action="store_true", help="Use soft-NMS instead of standard NMS")
    parser.add_argument("--use-batch-nms", action="store_true", help="Use batch NMS (per-class)")
    args = parser.parse_args()

    # Class-specific confidence thresholds to reduce person/dog confusion
    class_conf_thresholds = {#
        # Example: adjust these per your trained model
        # 0: 0.6,  # person
        # 1: 0.7,  # dog
        # 2: 0.5,  # cat
    }

    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open webcam device {args.camera}")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    print(f"Soft-NMS: {args.use_soft_nms}, Batch-NMS: {args.use_batch_nms}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    paused = False

    # Stats
    frame_times = []
    inference_times = []

    print("Press 'q' to quit, 's' to save, 'p' to pause, 'r' to reset stats")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        if paused:
            cv2.putText(frame, "PAUSED (press 'p' to resume)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("YOLOv8 Optimized Webcam Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                paused = False
            elif key == ord('q'):
                break
            continue

        frame_start = time.time()

        # Run inference
        inf_start = time.time()
        results = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, 
                       device=args.device, verbose=False, half=False)
        inf_time = time.time() - inf_start
        inference_times.append(inf_time)

        # Process results with optional optimizations
        annotated_frame = frame.copy()
        detected_classes = {}
        box_count = 0

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Extract detections
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Apply soft-NMS or batch-NMS if requested
            if args.use_soft_nms:
                boxes, confs = soft_nms(boxes, confs, iou_threshold=args.iou, method="soft")
                cls_ids = cls_ids[:len(boxes)]  # Align class IDs
            elif args.use_batch_nms:
                keep_idx = batch_nms(boxes, confs, cls_ids, iou_threshold=args.iou)
                boxes = boxes[keep_idx]
                confs = confs[keep_idx]
                cls_ids = cls_ids[keep_idx]

            # Apply class-specific confidence thresholds
            if class_conf_thresholds:
                valid_mask = []
                for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
                    threshold = class_conf_thresholds.get(cls_id, args.conf)
                    valid_mask.append(conf >= threshold)
                valid_mask = np.array(valid_mask)
                boxes = boxes[valid_mask]
                confs = confs[valid_mask]
                cls_ids = cls_ids[valid_mask]

            # Draw detections
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_name = model.names[cls_id]
                box_count += 1

                # Track class counts
                detected_classes[cls_name] = detected_classes.get(cls_name, 0) + 1

                # Draw box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{cls_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Compute stats
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        avg_inf = np.mean(inference_times[-30:]) if inference_times else 0
        avg_frame = np.mean(frame_times[-30:]) if frame_times else 0
        current_fps = 1.0 / (frame_time + 1e-6)

        # Draw stats
        y_offset = 30
        cv2.putText(annotated_frame, f"Boxes: {box_count} | Inf: {inf_time*1000:.1f}ms | FPS: {current_fps:.1f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(annotated_frame, f"Avg Inf: {avg_inf*1000:.1f}ms | Avg Frame: {avg_frame*1000:.1f}ms",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw class counts
        y_offset += 25
        for cls_name, count in sorted(detected_classes.items()):
            cv2.putText(annotated_frame, f"{cls_name}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
            y_offset += 20

        cv2.imshow("YOLOv8 Optimized Webcam Detection", annotated_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            frame_path = save_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
            print(f"Saved: {frame_path}")
            frame_count += 1
        elif key == ord('p'):
            paused = True
        elif key == ord('r'):
            frame_times.clear()
            inference_times.clear()
            print("Stats reset")

    cap.release()
    cv2.destroyAllWindows()

    # Final stats
    if frame_times:
        print(f"\n=== Performance Summary ===")
        print(f"Avg Inference: {np.mean(inference_times)*1000:.2f}ms")
        print(f"Avg Frame Time: {np.mean(frame_times)*1000:.2f}ms")
        print(f"Avg FPS: {1.0 / np.mean(frame_times):.1f}")
    print(f"Frames saved to: {save_dir}")


if __name__ == "__main__":
    main()
