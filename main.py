#!/usr/bin/env python3
"""
YOLOv8 Object Detection System
Supports webcam detection, image detection, video detection, and dataset analysis
"""

import os
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

def count_images(dataset_path):
    """Count images in dataset"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_count = 0
    
    for ext in image_extensions:
        image_count += len(list(Path(dataset_path).rglob(f'*{ext}')))
    
    return image_count

def load_dataset_info(yaml_file):
    """Load dataset info from YAML"""
    try:
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {yaml_file}: {e}")
        return None

def get_image_count_from_dataset(dataset_info):
    """Get image count from dataset YAML info"""
    if not dataset_info:
        return 0
    
    path = dataset_info.get('path', '')
    count = 0
    
    # Check for standard paths
    for key in ['train', 'val', 'test']:
        if key in dataset_info:
            subpath = os.path.join(path, dataset_info[key])
            if os.path.exists(subpath):
                count += count_images(subpath)
    
    return count

def start_webcam_detection(model_path='yolov8n.pt', conf=0.5):
    """Start webcam detection with FPS counter"""
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam")
            return
        
        print("\n✅ Webcam opened successfully!")
        print("🎥 Starting detection... Press 'q' to quit\n")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Resolution: {width}x{height} @ {fps} FPS")
        print(f"🎯 Confidence threshold: {conf}\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=conf, verbose=False)
            
            # Get detections info
            detections = results[0]
            num_detections = len(detections.boxes) if detections.boxes is not None else 0
            
            # Visualize results
            annotated_frame = detections.plot()
            
            # Add frame counter and detection count
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {num_detections}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
            
            # Display in terminal every 30 frames
            if frame_count % 30 == 0:
                print(f"📊 Frame {frame_count} | Detections: {num_detections}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n✅ Webcam detection completed!")
        print(f"📊 Total frames processed: {frame_count}")
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error during webcam detection: {e}")

def detect_image(image_path, model_path='yolov8n.pt', conf=0.5):
    """Detect objects in image"""
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        model = YOLO(model_path)
        results = model(image_path, conf=conf, verbose=False)
        
        detections = results[0]
        num_detections = len(detections.boxes) if detections.boxes is not None else 0
        
        print(f"\n✅ Detection completed!")
        print(f"📊 Objects detected: {num_detections}")
        
        # Display results
        annotated_frame = detections.plot()
        cv2.imshow('YOLOv8 Image Detection', annotated_frame)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error during image detection: {e}")

def detect_video(video_path, model_path='yolov8n.pt', conf=0.5):
    """Detect objects in video"""
    try:
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return
        
        model = YOLO(model_path)
        results = model(video_path, conf=conf, verbose=False)
        
        print(f"\n✅ Video detection completed!")
        print("Results saved to 'runs/detect' directory")
        
    except Exception as e:
        print(f"❌ Error during video detection: {e}")

def display_dataset_info():
    """Display information about available datasets"""
    datasets = {
        'OpenImages V6': 'openimages_v6.yaml',
        'COCO128': 'coco128.yaml',
        'Mapillary': 'mapillary.yaml',
        'WOTR': 'wotr.yaml'
    }
    
    print("\n" + "="*60)
    print("📊 DATASET INFORMATION")
    print("="*60)
    
    for name, yaml_file in datasets.items():
        if os.path.exists(yaml_file):
            print(f"\n📁 {name}:")
            info = load_dataset_info(yaml_file)
            if info:
                print(f"   ├─ Classes: {info.get('nc', 'N/A')}")
                
                # Count images
                img_count = get_image_count_from_dataset(info)
                print(f"   ├─ Total Images: {img_count}")
                
                # List class names
                if 'names' in info:
                    names = info['names']
                    if isinstance(names, list):
                        class_names = ', '.join(names[:5])
                        if len(names) > 5:
                            class_names += f", ... (+{len(names)-5} more)"
                    else:
                        class_names = str(names)
                    print(f"   └─ Classes: {class_names}")
        else:
            print(f"\n❌ {name}: {yaml_file} not found")

def main():
    print("\n" + "="*60)
    print("🚀 YOLOv8 OBJECT DETECTION SYSTEM")
    print("="*60)
    
    # Display dataset information
    display_dataset_info()
    
    # Check model file
    print("\n" + "="*60)
    print("🔍 MODEL STATUS")
    print("="*60)
    
    if os.path.exists('yolov8n.pt'):
        print("✅ YOLOv8 Nano model found (yolov8n.pt)")
    else:
        print("⚠️  YOLOv8 Nano model not found - will be downloaded on first use")
    
    # Menu loop
    while True:
        print("\n" + "="*60)
        print("🎯 SELECT MODE:")
        print("="*60)
        print("1. 🎥 Start Webcam Detection")
        print("2. 🖼️  Detect from Image")
        print("3. 🎬 Detect from Video")
        print("4. 📊 Show Dataset Info")
        print("0. ❌ Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == '1':
            conf = input("Enter confidence threshold (0.0-1.0, default 0.5): ").strip()
            try:
                conf = float(conf) if conf else 0.5
            except:
                conf = 0.5
            print("\n🎥 Starting Webcam Detection...")
            start_webcam_detection(conf=conf)
            
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            conf = input("Enter confidence threshold (0.0-1.0, default 0.5): ").strip()
            try:
                conf = float(conf) if conf else 0.5
            except:
                conf = 0.5
            if os.path.exists(image_path):
                detect_image(image_path, conf=conf)
            else:
                print(f"❌ Image not found: {image_path}")
                
        elif choice == '3':
            video_path = input("Enter video path: ").strip()
            conf = input("Enter confidence threshold (0.0-1.0, default 0.5): ").strip()
            try:
                conf = float(conf) if conf else 0.5
            except:
                conf = 0.5
            if os.path.exists(video_path):
                detect_video(video_path, conf=conf)
            else:
                print(f"❌ Video not found: {video_path}")
                
        elif choice == '4':
            display_dataset_info()
            
        elif choice == '0':
            print("\n👋 Exiting... Goodbye!")
            break
        else:
            print("\n❌ Invalid choice! Please enter 0-4")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
