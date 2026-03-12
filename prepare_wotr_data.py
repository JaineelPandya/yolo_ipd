import os
import xml.etree.ElementTree as ET
import random
import shutil
from tqdm import tqdm

# ------------------------------
# Configuration
# ------------------------------
ANNOTATIONS_DIR = "WOTR/Annotations"
JPEG_IMAGES_DIR = "WOTR/JPEGImages"
OUTPUT_DIR = "WOTR_YOLO"
TRAIN_RATIO = 0.8
SAMPLE_SIZE = 2000  # Number of images to sample

# Define the classes you want to include
CLASSES = ['ashcan', 'bicycle', 'blind_road', 'bus', 'car', 'crosswalk', 'dog', 'fire_hydrant', 'green_light', 'motorcycle', 'person', 'pole', 'red_light', 'reflective_cone', 'roadblock', 'sign', 'tree', 'tricycle', 'truck', 'warning_column']

# ------------------------------
# Helper Functions
# ------------------------------
def get_image_size(annotation_path):
    """Gets image size from an XML file."""
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        return width, height
    except (ET.ParseError, AttributeError, ValueError) as e:
        print(f"Error parsing {annotation_path}: {e}")
        return 0, 0

def convert_to_yolo_format(size, box):
    """Converts a single XML annotation to YOLO format."""
    width, height = size
    xmin, ymin, xmax, ymax = box

    # YOLO format: (x_center, y_center, width, height) normalized
    x_center = (xmin + xmax) / (2.0 * width)
    y_center = (ymin + ymax) / (2.0 * height)
    w = (xmax - xmin) / (1.0 * width)
    h = (ymax - ymin) / (1.0 * height)

    return x_center, y_center, w, h

# ------------------------------
# Main Processing
# ------------------------------
def main():
    """Main function to process the dataset."""
    print("🚀 Starting dataset preparation...")

    # Create output directories
    train_images_dir = os.path.join(OUTPUT_DIR, 'train', 'images')
    train_labels_dir = os.path.join(OUTPUT_DIR, 'train', 'labels')
    val_images_dir = os.path.join(OUTPUT_DIR, 'val', 'images')
    val_labels_dir = os.path.join(OUTPUT_DIR, 'val', 'labels')

    for path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created: {path}")

    # Get all annotation files and sample them
    all_annotations = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
    if len(all_annotations) > SAMPLE_SIZE:
        sampled_annotations = random.sample(all_annotations, SAMPLE_SIZE)
        print(f"✅ Sampled {SAMPLE_SIZE} annotations from {len(all_annotations)} total.")
    else:
        sampled_annotations = all_annotations
        print(f"⚠️ Using all {len(all_annotations)} annotations as it's less than the sample size.")

    # Split into training and validation sets
    random.shuffle(sampled_annotations)
    split_index = int(len(sampled_annotations) * TRAIN_RATIO)
    train_annotations = sampled_annotations[:split_index]
    val_annotations = sampled_annotations[split_index:]

    print(f"Split: {len(train_annotations)} for training, {len(val_annotations)} for validation.")

    # Process each set
    for subset, output_img_dir, output_lbl_dir in [
        (train_annotations, train_images_dir, train_labels_dir),
        (val_annotations, val_images_dir, val_labels_dir)
    ]:
        count = 0
        for annotation_file in tqdm(subset, desc=f"Processing {subset} a"):
            annotation_path = os.path.join(ANNOTATIONS_DIR, annotation_file)

            # Derive image filename from annotation filename
            image_filename_base = os.path.splitext(annotation_file)[0]
            image_filename = image_filename_base + '.jpg'
            image_path = os.path.join(JPEG_IMAGES_DIR, image_filename)

            if not os.path.exists(image_path):
                print(f"⚠️ Image not found for {annotation_file} (expected {image_filename}), skipping.")
                continue

            # Get image size from the XML file
            width, height = get_image_size(annotation_path)
            if width == 0 or height == 0:
                print(f"⚠️ Could not get image size for {annotation_file}, skipping.")
                continue
            
            # Parse XML for bounding boxes
            try:
                tree = ET.parse(annotation_path)
                root = tree.getroot()
            except ET.ParseError as e:
                print(f"⚠️ Could not parse XML for {annotation_file}: {e}, skipping.")
                continue

            yolo_labels = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in CLASSES:
                    continue

                class_id = CLASSES.index(name)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Convert to YOLO format
                x_center, y_center, w, h = convert_to_yolo_format((width, height), (xmin, ymin, xmax, ymax))
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            if not yolo_labels:
                print(f"⚠️ No relevant objects found in {annotation_file}, skipping.")
                continue

            # Copy image and write YOLO label
            shutil.copy(image_path, os.path.join(output_img_dir, image_filename))
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            with open(os.path.join(output_lbl_dir, label_filename), 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            count += 1
        print(f"✅ Processed {count} files for the subset.")

    print("🎉 Dataset preparation complete!")

if __name__ == "__main__":
    main()
