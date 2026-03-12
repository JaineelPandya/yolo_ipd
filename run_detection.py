#!/usr/bin/env python3
"""
Simplified YOLOv8 Webcam Detection Script
Works with standard webcam access on Windows
"""

import cv2
import sys
from pathlib import Path

print("Initializing YOLOv8 webcam detection...\n")

# Step 1: Import dependencies
try:
    from ultralytics import YOLO
    print("✓ Ultralytics YOLO imported successfully")
except ImportError as e:
    print(f"✗ Failed to import YOLO: {e}")
    sys.exit(1)

# Step 2: Load model
model_path = "yolov8n.pt"
print(f"\nLoading model: {model_path}")
try:
    model = YOLO(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  - Classes available: {len(model.names)}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Step 3: Setup webcam
print("\nInitializing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ ERROR: Could not open webcam.")
    print("  - Make sure your webcam is connected and not in use by another application")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"✓ Webcam opened: {frame_width}x{frame_height} @ {fps:.1f} FPS")

# Create output directory
output_dir = Path("runs/webcam")
output_dir.mkdir(parents=True, exist_ok=True)
frame_count = 0

print("\n" + "=" * 60)
print("WEBCAM DETECTION STARTED")
print("=" * 60)
print("Controls:")
print("  'q' - Quit detection")
print("  's' - Save current frame")
print("  'p' - Pause/Resume")
print("=" * 60 + "\n")

paused = False
frame_num = 0

# Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam")
        break

    frame_num += 1

    if paused:
        cv2.putText(frame, "PAUSED - Press 'p' to resume", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Run inference
        try:
            results = model(frame, conf=0.25, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        
                        detections.append({
                            'class': cls_name,
                            'conf': conf,
                            'box': (x1, y1, x2, y2)
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{cls_name} {conf:.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Display frame info
            info_text = f"Detections: {len(detections)} | Frame: {frame_num}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if len(detections) > 0:
                print(f"[Frame {frame_num}] Detected {len(detections)} objects: {', '.join(set([d['class'] for d in detections]))}")
                
        except Exception as e:
            print(f"Error during inference: {e}")
            break

    # Display the frame
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n[QUIT] Stopping detection...")
        break
    elif key == ord('s'):
        frame_path = output_dir / f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        print(f"✓ Frame saved: {frame_path}")
        frame_count += 1
    elif key == ord('p'):
        paused = not paused
        status = "PAUSED" if paused else "RESUMED"
        print(f"[{status}] Detection paused/resumed")

# Cleanup
print("\nCleaning up...")
cap.release()
cv2.destroyAllWindows()
print("✓ Webcam detection completed")
if frame_count > 0:
    print(f"✓ Saved {frame_count} frames to: {output_dir}")
