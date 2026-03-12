#!/usr/bin/env python3
"""Real-time object detection using webcam and YOLOv8 model.

This script captures video frames from the webcam and performs real-time
object detection using a trained YOLOv8 model. Detections are displayed
with bounding boxes and class labels.

Usage:
  python webcam_detection.py --model yolov8n.pt
  python webcam_detection.py --model runs/detect/train/weights/best.pt --conf 0.5 --device 0
  python webcam_detection.py --model best.pt --classes 0 1 5  # Only detect specific classes

Press 'q' to quit, 's' to save frame, 'p' to pause/resume.
"""

import argparse
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Real-time webcam detection with YOLOv8")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS (0-1)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference: 'cpu' or GPU ID (default: cpu)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Filter by class IDs (e.g., --classes 0 1 5)")
    parser.add_argument("--save-dir", type=str, default="runs/webcam", help="Directory to save frames")
    parser.add_argument("--fps-limit", type=int, default=30, help="Target FPS (limits processing speed)")
    args = parser.parse_args()

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

    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam opened: {frame_width}x{frame_height} @ {fps:.1f} FPS")

    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    paused = False

    print("Press 'q' to quit, 's' to save frame, 'p' to pause/resume")
    print(f"Confidence: {args.conf}, IOU: {args.iou}, Classes filter: {args.classes}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from webcam")
            break

        if paused:
            # Display paused message
            cv2.putText(frame, "PAUSED (press 'p' to resume)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("YOLOv8 Webcam Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                paused = False
            elif key == ord('q'):
                break
            continue

        # Run inference
        results = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False, half=False)

        # Process results
        annotated_frame = frame.copy()
        detected_classes = set()
        box_count = 0

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Filter by class if specified
                if args.classes is not None and cls_id not in args.classes:
                    continue

                detected_classes.add(cls_id)
                box_count += 1

                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class name
                class_name = model.names[cls_id]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label with background
                label = f"{class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add info overlay
        info_text = f"Boxes: {box_count} | Classes: {len(detected_classes)} | FPS: {fps:.1f}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display frame
        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit requested")
            break
        elif key == ord('s'):
            frame_path = save_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
            print(f"Frame saved: {frame_path}")
            frame_count += 1
        elif key == ord('p'):
            paused = True
            print("Paused (press 'p' to resume)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Webcam detection completed. Saved frames in: {save_dir}")


if __name__ == "__main__":
    main()
