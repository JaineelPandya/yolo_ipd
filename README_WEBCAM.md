# YOLO Webcam Detection & Training Setup

This project adds **Real-time Webcam Detection** with an efficient, trained YOLOv8 model to reduce class confusion.

## Quick Start

### 1. Setup Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & Prepare Open Images Dataset (100 classes, 20k images)

```bash
python scripts/prepare_openimages_v6.py --num-classes 100 --max-samples 20000 --output datasets/openimages_v6
```

Once download completes, verify:
```bash
ls datasets/openimages_v6/
```

### 3. Train YOLOv8 Model with Custom Hyperparameters

```bash
python -m ultralytics.train data=openimages_v6.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 hyp=hyp_openimages.yaml
```

Or for faster experimentation:
```bash
python -m ultralytics.train data=openimages_v6.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=32
```

### 4. Webcam Real-Time Detection

#### Basic Webcam Detection
```bash
python webcam_detection.py --model yolov8n.pt --conf 0.5
```

#### Optimized Webcam Detection (Soft-NMS, Class-Specific Thresholds)
```bash
python webcam_detection_optimized.py --model yolov8n.pt --conf 0.5 --use-soft-nms
```

#### With Trained Model
After training completes:
```bash
python webcam_detection_optimized.py --model runs/detect/train/weights/best.pt --conf 0.6 --use-soft-nms
```

#### Filter Detections by Class
```bash
python webcam_detection.py --model best.pt --classes 0 1 5 --conf 0.5
```

## Webcam Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current frame |
| `p` | Pause/Resume |
| `r` | Reset performance stats (optimized version only) |

## Key Features

### Efficient Detection
- **Soft-NMS**: Exponential decay of overlapping detections instead of hard removal (reduces false negatives)
- **Batch-NMS**: Per-class NMS for faster processing
- **Class-Specific Confidence**: Adjust per-class thresholds to reduce person/dog confusion

### Dataset
- **Open Images V6**: 20,000 images across 100 daily-use classes
- **Classes**: person, dog, cat, car, bus, truck, bicycle, motorcycle, cell phone, laptop, etc.

### Training Optimizations (`hyp_openimages.yaml`)
- **Focal Loss (FL)**: Better focus on hard examples
- **Class Weight**: Higher loss for minority classes
- **Mosaic Augmentation**: Increased context and diversity
- **Mixup**: Label smoothing reduces overfitting

## File Structure

```
.
├── webcam_detection.py              # Basic webcam detection
├── webcam_detection_optimized.py    # Optimized version with soft-NMS
├── efficient_detection.py           # Utility module for NMS optimizations
├── scripts/
│   └── prepare_openimages_v6.py    # Download & export Open Images dataset
├── openimages_v6.yaml              # Dataset config (10 classes)
├── hyp_openimages.yaml             # Custom training hyperparameters
└── README.md                        # This file
```

## Reducing Person/Dog Confusion

1. **Use Soft-NMS** instead of hard NMS:
   ```bash
   python webcam_detection_optimized.py --model best.pt --use-soft-nms
   ```

2. **Boost Class-Specific Confidence**:
   - Edit `webam_detection_optimized.py` and set:
     ```python
     class_conf_thresholds = {
         0: 0.6,  # person (higher threshold)
         1: 0.7,  # dog (even higher)
     }
     ```

3. **Use Trained Model**:
   - Train with the Open Images dataset to learn class distinctions:
     ```bash
     python -m ultralytics.train data=openimages_v6.yaml model=yolov8n.pt epochs=100
     ```

## Performance Tips

- **Reduce image size** for faster inference: `--imgsz 416` (default 640)
- **Lower confidence threshold** to catch more objects: `--conf 0.25`
- **Use GPU** (if available): `--device 0` (default)
- **Use lightweight model**: `yolov8n.pt` (nano, fast) vs. `yolov8m.pt` (medium, accurate)

## Example Outputs

- Webcam frames are saved to `runs/webcam/` or `runs/webcam_optimized/`
- Training results saved to `runs/detect/train/`
- Dataset exported to `datasets/openimages_v6/`

## References

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [Open Images V6](https://storage.googleapis.com/openimages/web/index.html)
- [FiftyOne Zoo](https://docs.voxel51.com/user_guide/dataset_zoo/index.html)
