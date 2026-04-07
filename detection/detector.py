"""
Detection module — YOLOv8 inference via TFLite (RPi) or PyTorch (desktop).

TFLite model expected: models/yolov8n_float32.tflite
PyTorch model expected: models/yolov8n.pt
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import config

logger = logging.getLogger(__name__)

# COCO 80-class names (standard YOLOv8n)
COCO_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class YOLODetector:
    """
    Unified detector that supports:
      - TensorFlow Lite (yolov8n_float32.tflite) — for Raspberry Pi
      - PyTorch / ultralytics (yolov8n.pt)        — for desktop / CUDA
    """

    def __init__(
        self,
        model_path: str = None,
        use_tflite: bool = None,
        device: str = None,
    ):
        self.use_tflite = config.USE_TFLITE if use_tflite is None else use_tflite
        self.device = config.INFERENCE_DEVICE if device is None else device

        if model_path is None:
            model_path = (
                config.YOLOV8_TFLITE_MODEL if self.use_tflite
                else config.YOLOV8_PT_MODEL
            )

        self.model_path = model_path
        self.model = None          # ultralytics YOLO object
        self.interpreter = None    # TFLite interpreter
        self._input_shape: Optional[Tuple[int, int]] = None

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if not Path(self.model_path).exists():
            logger.error(
                f"Model not found: {self.model_path}\n"
                "Download YOLOv8n TFLite from: "
                "https://github.com/ultralytics/assets/releases"
            )
            return

        try:
            if self.use_tflite:
                self._load_tflite()
            else:
                self._load_pt()
            logger.info(f"✓ Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"✗ Model load failed: {e}")
            raise

    def _load_pt(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"  PyTorch model on {self.device}")
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics torch")

    def _load_tflite(self):
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            except ImportError:
                raise ImportError(
                    "No TFLite runtime found. Install: pip install tflite-runtime"
                )

        self.interpreter = Interpreter(
            model_path=self.model_path,
            num_threads=config.NUM_THREADS,
        )
        self.interpreter.allocate_tensors()

        in_det = self.interpreter.get_input_details()[0]
        h, w = in_det["shape"][1], in_det["shape"][2]
        self._input_shape = (w, h)       # (width, height) for cv2.resize
        logger.info(f"  TFLite model — input shape: {in_det['shape']}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> Dict:
        """
        Run inference on a BGR frame.

        Returns:
            {
                "detections": List[{bbox, confidence, class_id, class_name, area}],
                "time_ms": float,
            }
        """
        if conf_threshold is None:
            conf_threshold = config.CONFIDENCE_THRESHOLD

        if self.model is None and self.interpreter is None:
            logger.warning("No model loaded — returning empty detections")
            return {"detections": [], "time_ms": 0}

        try:
            if self.use_tflite:
                return self._detect_tflite(frame, conf_threshold)
            else:
                return self._detect_pt(frame, conf_threshold)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"detections": [], "time_ms": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # PyTorch inference
    # ------------------------------------------------------------------

    def _detect_pt(self, frame: np.ndarray, conf_threshold: float) -> Dict:
        t0 = time.time()
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                detections.append({
                    "bbox":       bbox,
                    "confidence": float(box.conf[0]),
                    "class_id":   int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "area":       self._area(bbox),
                })
        return {"detections": detections, "time_ms": (time.time() - t0) * 1000}

    # ------------------------------------------------------------------
    # TFLite inference
    # ------------------------------------------------------------------

    def _detect_tflite(self, frame: np.ndarray, conf_threshold: float) -> Dict:
        t0 = time.time()

        in_details  = self.interpreter.get_input_details()
        out_details = self.interpreter.get_output_details()

        # Pre-process
        w, h = self._input_shape
        resized = cv2.resize(frame, (w, h))

        if in_details[0]["dtype"] == np.uint8:
            scale, zero = in_details[0]["quantization"]
            inp = ((resized / scale) + zero).astype(np.uint8)
        else:
            inp = (resized.astype(np.float32) / 255.0)

        # YOLOv8 TFLite: channels-last, single batch
        inp = np.expand_dims(inp, axis=0)

        self.interpreter.set_tensor(in_details[0]["index"], inp)
        self.interpreter.invoke()

        # Output shape for YOLOv8n float32: (1, 84, 8400)
        # 84 = [cx, cy, w, h, class_0…class_79]
        output = self.interpreter.get_tensor(out_details[0]["index"])

        detections = self._parse_yolov8_tflite(output, frame.shape, conf_threshold)
        return {"detections": detections, "time_ms": (time.time() - t0) * 1000}

    def _parse_yolov8_tflite(
        self,
        output: np.ndarray,
        frame_shape: Tuple,
        conf_threshold: float,
    ) -> List[Dict]:
        """
        Parse YOLOv8 TFLite output.
        Expected shape: (1, 84, 8400)  — 84 = 4 box coords + 80 classes
        """
        detections: List[Dict] = []

        # Squeeze batch dim → (84, 8400)
        preds = output[0]   # shape (84, 8400)

        if preds.ndim != 2:
            logger.warning(f"Unexpected TFLite output shape: {output.shape}")
            return detections

        num_coords = 4
        num_classes = preds.shape[0] - num_coords   # 80

        fh, fw = frame_shape[:2]

        # Transpose to (8400, 84) for easier iteration
        preds = preds.T   # (8400, 84)

        for pred in preds:
            class_scores = pred[num_coords : num_coords + num_classes]
            class_id     = int(np.argmax(class_scores))
            confidence   = float(class_scores[class_id])

            if confidence < conf_threshold:
                continue

            cx, cy, bw, bh = pred[:4]

            # Coords are normalised to [0,1] based on input size
            x1 = (cx - bw / 2) * fw
            y1 = (cy - bh / 2) * fh
            x2 = (cx + bw / 2) * fw
            y2 = (cy + bh / 2) * fh

            # Clip to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)

            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            detections.append({
                "bbox":       bbox,
                "confidence": confidence,
                "class_id":   class_id,
                "class_name": COCO_NAMES[class_id] if class_id < len(COCO_NAMES)
                              else f"class_{class_id}",
                "area":       self._area(bbox),
            })

        # Non-maximum suppression
        if detections:
            detections = self._nms(detections, config.IOU_THRESHOLD)

        return detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _area(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox
        return float((x2 - x1) * (y2 - y1))

    @staticmethod
    def _nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Simple class-agnostic NMS."""
        if not detections:
            return []

        boxes      = np.array([d["bbox"] for d in detections])
        scores     = np.array([d["confidence"] for d in detections])
        x1, y1     = boxes[:, 0], boxes[:, 1]
        x2, y2     = boxes[:, 2], boxes[:, 3]
        areas      = (x2 - x1) * (y2 - y1)
        order      = scores.argsort()[::-1]
        kept: List[int] = []

        while order.size > 0:
            i = order[0]
            kept.append(i)
            if order.size == 1:
                break

            inter_x1 = np.maximum(x1[i], x1[order[1:]])
            inter_y1 = np.maximum(y1[i], y1[order[1:]])
            inter_x2 = np.minimum(x2[i], x2[order[1:]])
            inter_y2 = np.minimum(y2[i], y2[order[1:]])

            inter_w  = np.maximum(0.0, inter_x2 - inter_x1)
            inter_h  = np.maximum(0.0, inter_y2 - inter_y1)
            inter    = inter_w * inter_h

            union    = areas[i] + areas[order[1:]] - inter
            iou      = inter / (union + 1e-6)

            order    = order[1:][iou <= iou_threshold]

        return [detections[i] for i in kept]

    def get_model_info(self) -> Dict:
        return {
            "model_path":       self.model_path,
            "use_tflite":       self.use_tflite,
            "device":           self.device,
            "input_shape":      self._input_shape,
            "conf_threshold":   config.CONFIDENCE_THRESHOLD,
            "iou_threshold":    config.IOU_THRESHOLD,
        }


def create_detector(use_tflite: bool = None, device: str = None) -> YOLODetector:
    """Factory function"""
    return YOLODetector(use_tflite=use_tflite, device=device)
