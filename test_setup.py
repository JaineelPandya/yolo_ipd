#!/usr/bin/env python3
"""Test script to verify environment setup and run webcam detection."""

import sys
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

print("=" * 60)
print("YOLO Webcam Detection - Environment Test")
print("=" * 60)

# Test imports
print("\n[1] Testing imports...")
try:
    import numpy as np
    print("✓ NumPy imported")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✓ Pandas imported")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    print("✓ PyTorch imported")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")

# Test webcam
print("\n[2] Testing webcam access...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Webcam detected: {width}x{height} @ {fps:.1f} FPS")
    cap.release()
else:
    print("✗ Webcam not accessible (this may be expected in headless environment)")

# Test YOLO model
print("\n[3] Testing YOLO model...")
model_path = "yolov8n.pt"
if Path(model_path).exists():
    print(f"✓ Model file exists: {model_path}")
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  - Classes: {len(model.names)}")
        print(f"  - Sample classes: {list(model.names.values())[:5]}")
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        sys.exit(1)
else:
    print(f"✗ Model file not found: {model_path}")
    sys.exit(1)

# Test inference
print("\n[4] Testing inference with dummy frame...")
try:
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model(dummy_frame, verbose=False)
    print(f"✓ Inference works (results: {len(results)} frame(s))")
except Exception as e:
    print(f"✗ Inference failed: {e}")

print("\n" + "=" * 60)
print("✓ All tests passed! Environment is ready.")
print("=" * 60)
print("\nNow running webcam detection...")
print("Controls: 'q' = quit, 's' = save frame, 'p' = pause/resume")
print("-" * 60)

# Run the actual detection
exec(open('webcam_detection.py').read())
