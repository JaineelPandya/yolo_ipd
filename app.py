"""
YOLOv8 Live Object Detection using Laptop Camera
Supports COCO, Mapillary, and WOTR datasets.
Author: Your Name
"""

import cv2
import torch
from ultralytics import YOLO
import argparse
import os

# ------------------------------
# ⚙️ Argument Parser
# ------------------------------
parser = argparse.ArgumentParser(description="YOLOv8 Live Object Detection")
parser.add_argument("--dataset", type=str, default="coco",
                    choices=["coco", "mapillary", "wotr"],
                    help="Choose between 'coco', 'mapillary', or 'wotr'")
parser.add_argument("--train", action="store_true",
                    help="Enable training mode for the WOTR dataset.")
args = parser.parse_args()

# ------------------------------
# 🔧 Model Configuration
# ------------------------------
if torch.cuda.is_available():
    print("✅ GPU detected — high-performance mode.")
else:
    print("⚙️ CPU detected — running in low-power mode.")

# Model paths
COCO_MODEL_PATH = "yolov8n.pt"  # pretrained on COCO (80 classes)
MAPILLARY_MODEL_PATH = "runs/detect/mapillary/weights/best.pt"  # your custom Mapillary model
WOTR_MODEL_PATH = "runs/detect/train/weights/best.pt"  # custom WOTR model
WOTR_YAML_PATH = "wotr.yaml"

# ------------------------------
# 🚀 Training Logic (Optional)
# ------------------------------
if args.train:
    if args.dataset != "wotr":
        print("⚠️ Training is only supported for the 'wotr' dataset. Please specify --dataset wotr.")
        exit(1)
    
    print("✅ Training mode enabled for WOTR dataset.")
    print("📦 Loading base model for training...")
    model = YOLO(COCO_MODEL_PATH)
    print("🚀 Starting training... This may take a while.")
    
    try:
        model.train(
            data=WOTR_YAML_PATH,
            epochs=25,  # A reasonable number for initial training
            imgsz=640,
            patience=5
        )
        print("🎉 Training complete! Model saved to runs/detect/train/")
        # After training, switch to inference mode with the new model
        print("📦 Loading newly trained WOTR model for live detection...")
        model = YOLO(WOTR_MODEL_PATH)
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        exit(1)

else:
    # Load appropriate model for inference
    if args.dataset == "mapillary":
        if os.path.exists(MAPILLARY_MODEL_PATH):
            print("📦 Loading Mapillary-trained model...")
            model = YOLO(MAPILLARY_MODEL_PATH)
            print("✅ Mapillary model loaded successfully.")
        else:
            print("⚠️ Mapillary weights not found! Falling back to COCO model.")
            model = YOLO(COCO_MODEL_PATH)
    elif args.dataset == "wotr":
        if os.path.exists(WOTR_MODEL_PATH):
            print("📦 Loading custom WOTR-trained model...")
            model = YOLO(WOTR_MODEL_PATH)
            print("✅ WOTR model loaded successfully.")
        else:
            print("⚠️ WOTR weights not found! Please train the model first using the --train flag.")
            print("Falling back to COCO model for now.")
            model = YOLO(COCO_MODEL_PATH)
    else:
        print("📦 Loading COCO pretrained model...")
        model = YOLO(COCO_MODEL_PATH)
        print("✅ COCO model loaded successfully.")

# ------------------------------
# 🎥 Open Laptop Camera
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Could not access laptop camera.")
    exit(1)

font = cv2.FONT_HERSHEY_SIMPLEX
print(f"🚀 Starting live detection on {args.dataset.upper()} dataset... Press 'q' to quit.")

# ------------------------------
# 🔍 Real-time Detection Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Empty frame. Retrying...")
        continue

    # Run inference
    results = model.predict(
        source=frame,
        conf=0.4,    # Confidence threshold
        imgsz=640,   # Inference size
        verbose=False
    )

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Draw detections and issue alerts
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names.get(cls, f"class_{cls}")

            # --- Hazard Alert Logic ---
            if args.dataset == 'wotr':
                # Alert for poles that are too close
                if name == 'pole':
                    box_height = y2 - y1
                    if box_height > frame_height * 0.6: # If pole takes up >60% of screen height
                        alert_text = "ALERT: Pole very close!"
                        print(alert_text)
                        cv2.putText(frame, alert_text, (50, 50), font, 1, (0, 0, 255), 3)

                # Alert for cars/trucks that are close
                if name in ['car', 'truck']:
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area > frame_width * frame_height * 0.2: # If vehicle takes up >20% of screen area
                        alert_text = f"ALERT: {name.capitalize()} is very close!"
                        print(alert_text)
                        cv2.putText(frame, alert_text, (50, 100), font, 1, (0, 0, 255), 3)

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8), font, 0.6, (255, 255, 255), 2)

            # Draw center coordinates
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: ({cx}, {cy})", (cx + 10, cy), font, 0.6, (255, 255, 255), 2)

    # Display the resulting frame with detections
    cv2.imshow("YOLOv8 Live Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# 🔚 Cleanup
# ------------------------------
cap.release()
cv2.destroyAllWindows()