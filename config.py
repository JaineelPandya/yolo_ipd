"""
Configuration settings for the assistive memory system
"""

import os
import platform
from pathlib import Path

# ========================= PROJECT PATHS =========================
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "data" / "database"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [MODELS_DIR, DATABASE_DIR, FRAMES_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========================= HARDWARE DETECTION =========================
def detect_device():
    """Detect device type (Raspberry Pi, Desktop, etc.)"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            if 'Raspberry Pi' in model:
                return 'raspberry_pi'
    except:
        pass
    
    return 'desktop'

DEVICE_TYPE = detect_device()
IS_RASPBERRY_PI = DEVICE_TYPE == 'raspberry_pi'

# ========================= MODEL SETTINGS =========================
# Model paths
YOLOV8_PT_MODEL = str(MODELS_DIR / "yolov8n.pt")
YOLOV8_TFLITE_MODEL = str(MODELS_DIR / "yolov8n.tflite")

# Use TensorFlow Lite on Raspberry Pi for better performance
USE_TFLITE = IS_RASPBERRY_PI

# Device selection: check for CUDA availability on desktop
def _get_inference_device():
    if IS_RASPBERRY_PI:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"

INFERENCE_DEVICE = _get_inference_device()

# Model configuration
MODEL_INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.25  # Default 0.25 - lower for better detection coverage
IOU_THRESHOLD = 0.45

# Small objects to track (assistive tracking focus)
SMALL_OBJECTS_TO_TRACK = [
    'cell phone', 'phone', 'keys', 'wallet', 'glasses', 'watch',
    'earbuds', 'headphones', 'pen', 'pencil', 'notebook', 'book',
    'cup', 'glass', 'bottle', 'remote', 'controller', 'mouse',
    'keyboard', 'laptop', 'tablet', 'passport', 'credit card',
    'coin', 'ring', 'necklace', 'bracelet', 'hat', 'gloves',
    'shoe', 'sock', 'tie', 'scarf', 'bag', 'purse'
]

# ========================= TRACKING SETTINGS =========================
TRACKER_TYPE = "bytetrack"  # 'bytetrack' or 'deepsort'
TRACKER_CONFIDENCE = 0.5
MAX_AGE = 30  # Maximum frames to keep track without detection
MIN_HITS = 3  # Minimum detections to start tracking
MAX_TRACKS = 100  # Maximum simultaneous tracks

# ========================= FRAME DEDUPLICATION SETTINGS =========================
EMBEDDING_MODEL = "resnet50"  # For frame similarity comparison
SIMILARITY_THRESHOLD = 0.85  # Threshold for considering frames as duplicates (0-1)
VIDEO_FPS = 30  # Expected webcam FPS

# Store frame if:
# - Object location moved more than LOCATION_CHANGE_THRESHOLD pixels
# - Confidence improved by CONFIDENCE_IMPROVEMENT_THRESHOLD
# - Scene changed (detected via embedding similarity)
LOCATION_CHANGE_THRESHOLD = 50  # pixels
CONFIDENCE_IMPROVEMENT_THRESHOLD = 0.1  # 10% improvement

# ========================= MEMORY/DATABASE SETTINGS =========================
DATABASE_PATH = str(DATABASE_DIR / "object_memory.db")
MAX_STORED_FRAMES = 1000  # Limit stored frames for memory efficiency
FRAME_COMPRESSION_QUALITY = 85  # JPEG quality (0-100)
COMPRESSED_FRAME_SIZE = (320, 240)  # Resize for storage

# ========================= GEMINI API SETTINGS =========================
GEMINI_API_KEY = "AIzaSyAoE9gpy6QB2__OqNqfiXTeu6qFQ7_idjc"
GEMINI_MODEL = "gemini-1.5-vision"
GEMINI_SCENE_PROMPT = """Describe the environment and where objects are placed so a visually impaired person can understand the location. 
Be concise, specific, and focus on spatial relationships. 
Format: "The scene shows [main objects], with [small object] positioned at [location relative to main objects/furniture]."
"""

# ========================= QUERY SETTINGS =========================
# Recent search range (in minutes)
RECENT_SEARCH_RANGE = 60  # Last 1 hour by default

# ========================= STREAMLIT SETTINGS =========================
STREAMLIT_PAGE_TITLE = "Object Memory Assistant"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_INITIAL_SIDEBAR_STATE = "expanded"

# ========================= LOGGING SETTINGS =========================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ========================= PERFORMANCE SETTINGS =========================
# Memory optimization for Raspberry Pi
if IS_RASPBERRY_PI:
    # Reduce quality and size for RPi
    FRAME_COMPRESSION_QUALITY = 70
    COMPRESSED_FRAME_SIZE = (240, 180)
    MAX_STORED_FRAMES = 500
    BATCH_SIZE = 1
    NUM_THREADS = 2
else:
    BATCH_SIZE = 4
    NUM_THREADS = 4

# Inference skip frames for faster processing
INFERENCE_SKIP_FRAMES = 2 if IS_RASPBERRY_PI else 1

print(f"[CONFIG] Device Type: {DEVICE_TYPE}")
print(f"[CONFIG] Using TensorFlow Lite: {USE_TFLITE}")
print(f"[CONFIG] Inference Device: CPU" if USE_TFLITE else f"GPU {INFERENCE_DEVICE}")
