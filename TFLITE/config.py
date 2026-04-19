"""
Configuration settings for the Object Memory Assistant
Optimized for both Raspberry Pi (TFLite) and Desktop (PyTorch/CUDA)
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv optional; set GEMINI_API_KEY in your shell env

# ========================= PROJECT PATHS =========================
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "data" / "database"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create required directories on startup
for _dir in [MODELS_DIR, DATABASE_DIR, FRAMES_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ========================= HARDWARE DETECTION =========================
def detect_device() -> str:
    """Detect device type: 'raspberry_pi' or 'desktop'"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            if 'Raspberry Pi' in f.read():
                return 'raspberry_pi'
    except Exception:
        pass
    return 'desktop'

DEVICE_TYPE = detect_device()
IS_RASPBERRY_PI = DEVICE_TYPE == 'raspberry_pi'

# ========================= MODEL SETTINGS =========================
YOLOV8_PT_MODEL    = str(MODELS_DIR / "yolov8n.pt")
YOLOV8_TFLITE_MODEL = str(MODELS_DIR / "yolov8n_float32.tflite")

# Use TFLite on Raspberry Pi; PyTorch on desktop
USE_TFLITE = IS_RASPBERRY_PI

def _get_inference_device() -> str:
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

# Detection parameters
MODEL_INPUT_SIZE           = (640, 640)
CONFIDENCE_THRESHOLD       = 0.25
IOU_THRESHOLD              = 0.45
INFERENCE_SKIP_FRAMES      = 3 if IS_RASPBERRY_PI else 1

# ========================= TRACKING SETTINGS =========================
TRACKER_TYPE = "bytetrack"
MAX_AGE      = 30    # frames to keep a lost track alive
MIN_HITS     = 2     # detections needed to confirm a track
MAX_TRACKS   = 50

# ========================= FRAME DEDUPLICATION =========================
# Lightweight histogram-based similarity (no ResNet50 needed!)
SIMILARITY_THRESHOLD           = 0.92   # 0-1, higher = stricter dedup
LOCATION_CHANGE_THRESHOLD      = 60     # pixels — store if object moved this far
CONFIDENCE_IMPROVEMENT_THRES   = 0.10   # store if confidence jumped by 10%

# ========================= MEMORY / DATABASE =========================
DATABASE_PATH          = str(DATABASE_DIR / "object_memory.db")
MAX_STORED_FRAMES      = 500 if IS_RASPBERRY_PI else 1000
FRAME_COMPRESSION_QUALITY = 70 if IS_RASPBERRY_PI else 85
COMPRESSED_FRAME_SIZE  = (240, 180) if IS_RASPBERRY_PI else (320, 240)

# ========================= GEMINI API =========================
# Load from .env file or environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = "gemini-2.0-flash"          # fast + cheap model

# Rate-limiting: minimum seconds between Gemini calls (reduced for better coverage)
GEMINI_MIN_INTERVAL_SECONDS = 2  # Was 10, now 2 seconds for faster descriptions

GEMINI_OBJECT_PROMPT_TEMPLATE = """\
Look at this image. There is a '{object_name}' visible.

In 1-2 sentences, describe WHERE the {object_name} is located. Include:
- the room or environment type (bedroom, kitchen, desk area, etc.)
- the surface or furniture it's on/near
- its position (left side, top shelf, etc.)

Reply in second person as if telling the user where they left it.
Example: "You left your {object_name} on the wooden shelf in the bedroom, near the lamp."
Do NOT use bullet points. Just 1-2 natural sentences."""

# ========================= EMBEDDING & RAG SETTINGS =========================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        # Fast model (22MB) - good for RPi
# EMBEDDING_MODEL = "all-mpnet-base-v2"    # Better quality (420MB) - slower
EMBEDDING_DIM = 384                         # Output dimension of embeddings

# RAG Settings
ENABLE_RAG = True                           # Enable Retrieval-Augmented Generation
SEMANTIC_SIMILARITY_THRESHOLD = 0.65        # Min similarity (0-1) for RAG search
SEMANTIC_SEARCH_TOP_K = 5                   # Return top 5 similar results

# FAISS Vector Store
ENABLE_FAISS = True                         # Use FAISS for fast vector search
FAISS_INDEX_PATH = str(DATABASE_DIR / "faiss_index.bin")

# ========================= QUERY SETTINGS =========================
RECENT_SEARCH_RANGE = 60   # minutes to look back by default

# ========================= LOGGING =========================
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ========================= SMALL OBJECTS TO TRACK =========================
SMALL_OBJECTS_TO_TRACK = [
    'cell phone', 'phone', 'keys', 'wallet', 'glasses', 'watch',
    'earbuds', 'headphones', 'pen', 'pencil', 'notebook', 'book',
    'cup', 'bottle', 'remote', 'mouse', 'keyboard', 'laptop',
    'tablet', 'coin', 'ring', 'bag', 'purse', 'umbrella', 'scissors',
    'toothbrush', 'clock', 'vase', 'backpack', 'suitcase'
]

# ========================= PERFORMANCE (RPi) =========================
BATCH_SIZE   = 1 if IS_RASPBERRY_PI else 4
NUM_THREADS  = 2 if IS_RASPBERRY_PI else 4

# ========================= DEBUG =========================
print(f"[CONFIG] Device: {DEVICE_TYPE} | TFLite: {USE_TFLITE} | Inference: {INFERENCE_DEVICE}")
