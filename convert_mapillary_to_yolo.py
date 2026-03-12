import os
import cv2
import numpy as np
from tqdm import tqdm

MASKS_DIR = "data/mapillary/training/labels"
IMAGES_DIR = "data/mapillary/training/images"
YOLO_LABELS_DIR = "data/mapillary_yolo/train/labels"
YOLO_IMAGES_DIR = "data/mapillary_yolo/train/images"

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
os.makedirs(YOLO_IMAGES_DIR, exist_ok=True)

def mask_to_yolo(mask_path, label_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    h, w = mask.shape[:2]
    unique_classes = np.unique(mask)

    with open(label_path, "w") as f:
        for cls_id in unique_classes:
            if cls_id == 0:  # background
                continue

            ys, xs = np.where(mask == cls_id)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)

            # Normalize coordinates for YOLO format
            x_center = (x_min + x_max) / (2 * w)
            y_center = (y_min + y_max) / (2 * h)
            bbox_width = (x_max - x_min) / w
            bbox_height = (y_max - y_min) / h

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Loop through all masks
for mask_file in tqdm(os.listdir(MASKS_DIR)):
    if mask_file.endswith(".png"):
        mask_path = os.path.join(MASKS_DIR, mask_file)
        label_path = os.path.join(YOLO_LABELS_DIR, mask_file.replace(".png", ".txt"))
        mask_to_yolo(mask_path, label_path)
        # Copy image to new folder
        img_path = os.path.join(IMAGES_DIR, mask_file.replace(".png", ".jpg"))
        if os.path.exists(img_path):
            os.system(f"cp '{img_path}' '{YOLO_IMAGES_DIR}/'")
