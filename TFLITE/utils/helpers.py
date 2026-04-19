"""
Utility functions for frame processing and management
"""

import cv2
import numpy as np
import hashlib
import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import config

logger = logging.getLogger(__name__)

class FrameProcessor:
    """Process and manage frames"""
    
    @staticmethod
    def resize_frame(frame: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """Resize frame while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        
        if width:
            aspect = width / w
            height = int(h * aspect)
        elif height:
            aspect = height / h
            width = int(w * aspect)
        else:
            return frame
        
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def compress_frame(frame: np.ndarray, quality: int = 85, 
                      size_tuple: Tuple = None) -> np.ndarray:
        """
        Compress frame for storage
        
        Args:
            frame: Input frame
            quality: JPEG quality (1-100)
            size_tuple: Optional (width, height) to resize
        
        Returns:
            Compressed frame
        """
        compressed = frame.copy()
        
        # Resize if specified
        if size_tuple:
            compressed = cv2.resize(compressed, size_tuple, interpolation=cv2.INTER_LINEAR)
        
        return compressed
    
    @staticmethod
    def calculate_frame_hash(frame: np.ndarray) -> str:
        """Calculate hash of frame for duplicate detection"""
        # Resize to small size for faster hashing
        small_frame = cv2.resize(frame, (100, 100))
        
        # Calculate hash
        frame_bytes = small_frame.tobytes()
        hash_obj = hashlib.sha256(frame_bytes)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list, 
                       tracked_objects: dict = None,
                       show_track_id: bool = True,
                       highlight_small_objects: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            tracked_objects: Dictionary of tracked objects
            show_track_id: Show tracking ID
            highlight_small_objects: Highlight important small objects
        
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox'].astype(int)
            x1, y1, x2, y2 = bbox
            
            # Determine color based on object type
            class_name = detection['class_name']
            is_small_object = class_name.lower() in config.SMALL_OBJECTS_TO_TRACK
            
            if is_small_object and highlight_small_objects:
                color = (0, 255, 0)  # Green for important objects
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue for regular objects
                thickness = 2
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            confidence = detection.get('confidence', 0)
            label = f"{class_name} {confidence:.2f}"
            
            if show_track_id and 'track_id' in detection:
                label = f"ID:{detection['track_id']} {label}"
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            label_y = max(20, y1 - 5)
            
            cv2.rectangle(output, 
                         (x1, label_y - text_size[1] - 5),
                         (x1 + text_size[0] + 5, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(output, label, (x1 + 2, label_y - 2),
                       font, font_scale, (255, 255, 255), font_thickness)
        
        return output
    
    @staticmethod
    def save_frame(frame: np.ndarray, output_dir: str = None, 
                  quality: int = 95, prefix: str = "frame") -> Optional[str]:
        """
        Save frame to disk
        
        Args:
            frame: Frame to save
            output_dir: Output directory
            quality: JPEG quality
            prefix: Filename prefix
        
        Returns:
            Path to saved file or None
        """
        if output_dir is None:
            output_dir = str(config.FRAMES_DIR)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save with quality setting
            success = cv2.imwrite(filepath, frame, 
                                 [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if success:
                logger.debug(f"Saved frame: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save frame: {filepath}")
                return None
        
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return None
    
    @staticmethod
    def save_compressed_frame(frame: np.ndarray, output_dir: str = None,
                             quality: int = 70, 
                             size: Tuple = None) -> Optional[str]:
        """
        Save compressed frame for storage optimization
        
        Args:
            frame: Frame to save
            output_dir: Output directory
            quality: JPEG quality (lower = smaller)
            size: Optional resize tuple (width, height)
        
        Returns:
            Path to saved file or None
        """
        if size is None:
            size = config.COMPRESSED_FRAME_SIZE
        
        if quality is None:
            quality = config.FRAME_COMPRESSION_QUALITY
        
        # Resize
        compressed = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        
        # Save
        return FrameProcessor.save_frame(compressed, output_dir, quality, "compressed")


class RaspberryPiOptimizer:
    """Raspberry Pi specific optimizations"""
    
    @staticmethod
    def optimize_for_rpi():
        """Apply RPi optimizations"""
        if not config.IS_RASPBERRY_PI:
            return
        
        try:
            # Disable OpenCV GPU operations
            import cv2
            cv2.ocl.setUseOpenCL(False)
            
            # Set thread count
            cv2.setNumThreads(config.NUM_THREADS)
            
            logger.info("✓ Raspberry Pi optimizations applied")
        except Exception as e:
            logger.warning(f"Could not apply RPi optimizations: {e}")
    
    @staticmethod
    def get_device_info() -> dict:
        """Get Raspberry Pi device information"""
        info = {
            "is_raspberry_pi": config.IS_RASPBERRY_PI,
            "use_tflite": config.USE_TFLITE,
            "inference_device": config.INFERENCE_DEVICE,
            "num_threads": config.NUM_THREADS,
            "batch_size": config.BATCH_SIZE
        }
        
        # Try to get hardware info
        try:
            with open('/proc/device-tree/model', 'r') as f:
                info["model"] = f.read().strip()
        except:
            info["model"] = "Unknown"
        
        # Get memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["total_memory_gb"] = memory.total / (1024**3)
            info["available_memory_gb"] = memory.available / (1024**3)
        except:
            pass
        
        return info


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.frame_times = []
        self.inference_times = []
        self.nms_times = []
        self.max_history = 100
    
    def add_frame_time(self, elapsed_ms: float):
        """Record frame processing time"""
        self.frame_times.append(elapsed_ms)
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)
    
    def add_inference_time(self, elapsed_ms: float):
        """Record inference time"""
        self.inference_times.append(elapsed_ms)
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)
    
    def get_fps(self) -> float:
        """Get average FPS"""
        if not self.frame_times:
            return 0.0
        avg_frame_time = np.mean(self.frame_times)
        return 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "fps": self.get_fps(),
            "avg_frame_time_ms": np.mean(self.frame_times) if self.frame_times else 0,
            "avg_inference_time_ms": np.mean(self.inference_times) if self.inference_times else 0,
            "frame_time_std": np.std(self.frame_times) if self.frame_times else 0
        }


def setup_logging(log_level: str = "INFO", log_dir: str = None):
    """Setup logging configuration"""
    if log_dir is None:
        log_dir = str(config.LOGS_DIR)
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = os.path.join(log_dir, "system.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging initialized: {log_file}")
