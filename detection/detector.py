"""
Detection module - handles YOLO inference with PT and TFLite support
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import config

logger = logging.getLogger(__name__)

class YOLODetector:
    """Detection class supporting both PyTorch and TensorFlow Lite models"""
    
    def __init__(self, model_path: str = None, use_tflite: bool = False, device: str = "cpu"):
        """
        Initialize detector
        
        Args:
            model_path: Path to model file
            use_tflite: Use TensorFlow Lite model
            device: 'cpu' or GPU device ID
        """
        self.use_tflite = use_tflite or config.USE_TFLITE
        self.device = device
        self.model = None
        self.interpreter = None
        
        if model_path is None:
            model_path = config.YOLOV8_TFLITE_MODEL if self.use_tflite else config.YOLOV8_PT_MODEL
        
        self.model_path = model_path
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load model from file"""
        if not Path(model_path).exists():
            logger.warning(f"Model not found at {model_path}. Make sure to download it first.")
            logger.info(f"Download models using: python scripts/download_models.py")
            return
        
        try:
            if self.use_tflite:
                self._load_tflite_model(model_path)
            else:
                self._load_pt_model(model_path)
            logger.info(f"✓ Model loaded: {model_path}")
        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            raise
    
    def _load_pt_model(self, model_path: str):
        """Load PyTorch YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"✓ Loaded PyTorch model on device: {self.device}")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics torch")
            raise
    
    def _load_tflite_model(self, model_path: str):
        """Load TensorFlow Lite model"""
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            logger.info(f"✓ Loaded TensorFlow Lite model")
        except ImportError:
            logger.error("tensorflow not installed. Install with: pip install tensorflow")
            raise
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Dict:
        """
        Run inference on frame
        
        Args:
            frame: Input image (BGR)
            conf_threshold: Confidence threshold
        
        Returns:
            Dictionary with detections
        """
        if self.model is None and self.interpreter is None:
            logger.error("Model not loaded")
            return {"detections": [], "time": 0}
        
        try:
            if self.use_tflite:
                return self._detect_tflite(frame, conf_threshold)
            else:
                return self._detect_pt(frame, conf_threshold)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"detections": [], "time": 0, "error": str(e)}
    
    def _detect_pt(self, frame: np.ndarray, conf_threshold: float) -> Dict:
        """Inference using PyTorch YOLO"""
        import time
        
        start_time = time.time()
        
        # Run inference
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            for detection in result.boxes:
                detections.append({
                    'bbox': detection.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                    'confidence': float(detection.conf[0]),
                    'class_id': int(detection.cls[0]),
                    'class_name': result.names[int(detection.cls[0])],
                    'area': self._calculate_area(detection.xyxy[0].cpu().numpy())
                })
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "detections": detections,
            "time": inference_time,
            "frame_shape": frame.shape
        }
    
    def _detect_tflite(self, frame: np.ndarray, conf_threshold: float) -> Dict:
        """Inference using TensorFlow Lite"""
        import time
        
        start_time = time.time()
        
        # Preprocess
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Resize frame to model input size
        input_shape = input_details[0]['shape']
        resized_frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
        
        # Normalize
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            resized_frame = (resized_frame / input_scale + input_zero_point).astype(np.uint8)
        else:
            resized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(resized_frame, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        
        # Parse detections (YOLOv8 TFLite output format)
        detections = self._parse_tflite_output(output_data, frame.shape, conf_threshold)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "detections": detections,
            "time": inference_time,
            "frame_shape": frame.shape
        }
    
    def _parse_tflite_output(self, output_data: np.ndarray, frame_shape: Tuple, 
                            conf_threshold: float) -> List[Dict]:
        """Parse TFLite model output"""
        # This is a placeholder - actual parsing depends on model output format
        # For YOLOv8 TFLite, typically shape is (1, 25200, 85)
        detections = []
        
        if len(output_data.shape) == 3:
            # Shape: (batch, num_detections, 85)
            # Format: [x, y, w, h, conf, class_probs...]
            predictions = output_data[0]  # Take first batch
            
            for pred in predictions:
                conf = pred[4]
                if conf > conf_threshold:
                    # Get class
                    class_probs = pred[5:]
                    class_id = np.argmax(class_probs)
                    class_conf = class_probs[class_id]
                    
                    # Convert to xyxy format
                    x, y, w, h = pred[:4]
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2
                    
                    # Scale to frame size
                    h_frame, w_frame = frame_shape[:2]
                    x1, x2 = x1 * w_frame / 640, x2 * w_frame / 640
                    y1, y2 = y1 * h_frame / 640, y2 * h_frame / 640
                    
                    detections.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'confidence': float(conf * class_conf),
                        'class_id': int(class_id),
                        'class_name': self._get_class_name(int(class_id)),
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        return detections
    
    def _calculate_area(self, bbox: np.ndarray) -> float:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = bbox
        return float((x2 - x1) * (y2 - y1))
    
    def _get_class_name(self, class_id: int) -> str:
        """Get COCO class name"""
        coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'phone', 'keys', 'wallet', 'glasses', 'headphones',
            'watch', 'earbuds', 'pen', 'notebook'
        ]
        
        if 0 <= class_id < len(coco_names):
            return coco_names[class_id]
        return f"class_{class_id}"
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "use_tflite": self.use_tflite,
            "device": self.device,
            "input_size": config.MODEL_INPUT_SIZE,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "iou_threshold": config.IOU_THRESHOLD
        }


def create_detector(use_tflite: bool = None, device: str = None) -> YOLODetector:
    """Factory function to create detector"""
    if use_tflite is None:
        use_tflite = config.USE_TFLITE
    if device is None:
        device = config.INFERENCE_DEVICE
    
    return YOLODetector(use_tflite=use_tflite, device=device)
