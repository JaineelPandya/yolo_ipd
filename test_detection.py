#!/usr/bin/env python3
"""
🔍 Test Detection System - Diagnose detection issues
"""

import cv2
import sys
import logging
from pathlib import Path

import config
from detection.detector import create_detector
from tracking.tracker import create_tracker
from utils.helpers import FrameProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_webcam():
    """Test if webcam is working"""
    print("\n" + "="*60)
    print("🎥 TESTING WEBCAM")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ FAILED: Cannot open webcam!")
        print("   - Check USB connection")
        print("   - Try different camera ID (0, 1, 2...)")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ FAILED: Cannot read from webcam!")
        return False
    
    print(f"✓ Webcam OK - Frame size: {frame.shape}")
    cap.release()
    return True

def test_model_loading():
    """Test if model loads correctly"""
    print("\n" + "="*60)
    print("🧠 TESTING MODEL LOADING")
    print("="*60)
    
    try:
        detector = create_detector(use_tflite=config.USE_TFLITE)
        if detector is None:
            print("❌ FAILED: Detector is None!")
            return False, None
        
        model_type = "TensorFlow Lite" if config.USE_TFLITE else "PyTorch"
        print(f"✓ Model loaded: {model_type}")
        return True, detector
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, None

def test_detection(detector, test_image=None):
    """Test detection on sample frame"""
    print("\n" + "="*60)
    print("🎯 TESTING DETECTION")
    print("="*60)
    
    if test_image is None:
        # Use webcam for testing
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("❌ FAILED: Cannot get frame from webcam!")
            return False
    else:
        frame = cv2.imread(test_image)
        if frame is None:
            print(f"❌ FAILED: Cannot load test image: {test_image}")
            return False
    
    print(f"Testing with frame size: {frame.shape}")
    
    # Test different confidence thresholds
    thresholds = [0.1, 0.25, 0.5, 0.75]
    
    for conf_threshold in thresholds:
        result = detector.detect(frame, conf_threshold=conf_threshold)
        detections = result.get('detections', [])
        
        print(f"\nConfidence {conf_threshold:.2f}: {len(detections)} detections")
        
        if detections:
            for i, det in enumerate(detections[:5]):  # Show first 5
                print(f"  {i+1}. {det['class_name']}: {det['confidence']:.2f}")
            if len(detections) > 5:
                print(f"  ... and {len(detections)-5} more")
        else:
            print(f"  ⚠️  No detections at this threshold")
    
    # Show recommended threshold
    result_025 = detector.detect(frame, conf_threshold=0.25)
    result_050 = detector.detect(frame, conf_threshold=0.5)
    
    det_025 = len(result_025.get('detections', []))
    det_050 = len(result_050.get('detections', []))
    
    print(f"\n📊 Recommendation:")
    print(f"  - At conf=0.25: {det_025} detections")
    print(f"  - At conf=0.50: {det_050} detections")
    
    if det_025 == 0:
        print("  ⚠️  Model may not be working or objects not visible")
        return False
    elif det_025 > det_050:
        print(f"  → Try lower confidence (0.25) in Streamlit for better detection")
    
    return True

def test_tracking(detector):
    """Test tracking on video"""
    print("\n" + "="*60)
    print("📍 TESTING TRACKING")
    print("="*60)
    
    try:
        tracker = create_tracker("bytetrack")
        print("✓ Tracker loaded successfully")
        
        # Test with a few frames
        cap = cv2.VideoCapture(0)
        
        tracked_count = 0
        for frame_num in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            
            result = detector.detect(frame, conf_threshold=0.25)
            detections = result.get('detections', [])
            
            if detections:
                tracked_objects = tracker.update(detections)
                tracked_count += len(tracked_objects)
        
        cap.release()
        
        print(f"✓ Processed 30 frames, tracked {tracked_count} objects")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_phone_detection():
    """Specific test for phone detection"""
    print("\n" + "="*60)
    print("📱 TESTING PHONE DETECTION")
    print("="*60)
    
    detector = create_detector(use_tflite=config.USE_TFLITE)
    if detector is None:
        print("❌ Cannot initialize detector")
        return
    
    cap = cv2.VideoCapture(0)
    print("💡 Hold a phone in front of camera and press SPACE when ready...")
    
    frame_count = 0
    phone_found = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Test every 5 frames
        if frame_count % 5 == 0:
            result = detector.detect(frame, conf_threshold=0.25)
            detections = result.get('detections', [])
            
            # Check for phone
            for det in detections:
                if 'phone' in det['class_name'].lower():
                    print(f"✓ PHONE DETECTED! Confidence: {det['confidence']:.2f}")
                    phone_found = True
                    break
            
            if frame_count % 30 == 0:
                print(f"  Scanned {frame_count} frames, {len(detections)} total objects detected")
        
        # Display frame with detections
        if detections:
            annotated = FrameProcessor.draw_detections(frame, detections)
            cv2.imshow("Phone Detection Test", annotated)
        else:
            cv2.imshow("Phone Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if not phone_found:
        print("\n⚠️  Phone not detected!")
        print("   Possible causes:")
        print("   - Phone not visible in frame")
        print("   - Model not detecting phones")
        print("   - Lighting too dark/bright")
        print("   - Try lower confidence threshold in Streamlit")

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  🔍 YOLO DETECTION SYSTEM DIAGNOSTIC TEST".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Run tests
    webcam_ok = test_webcam()
    if not webcam_ok:
        print("\n❌ Webcam test failed. Fix this first!")
        sys.exit(1)
    
    model_ok, detector = test_model_loading()
    if not model_ok or detector is None:
        print("\n❌ Model loading failed!")
        sys.exit(1)
    
    detection_ok = test_detection(detector)
    
    if detection_ok:
        test_tracking(detector)
    
    # Test specific phone detection
    print("\n" + "="*60)
    response = input("Do you want to test phone detection? (y/n): ").lower()
    if response == 'y':
        test_phone_detection()
    
    print("\n" + "="*60)
    print("✓ DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\n📝 SUMMARY:")
    print("  - Webcam: ✓" if webcam_ok else "  - Webcam: ❌")
    print("  - Model: ✓" if model_ok else "  - Model: ❌")
    print("  - Detection: ✓" if detection_ok else "  - Detection: ❌")
    print("\n💡 NEXT STEPS:")
    print("  1. Run Streamlit app: streamlit run app_streamlit.py")
    print("  2. Set confidence threshold to 0.25 (in sidebar)")
    print("  3. Click 'Start Camera' and observe detections")

if __name__ == "__main__":
    main()
