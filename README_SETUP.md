# 🚀 AI Object Memory Assistant - Setup & Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Running Streamlit Web Interface](#running-streamlit-web-interface)
5. [Raspberry Pi Deployment](#raspberry-pi-deployment)
6. [Database Structure](#database-structure)
7. [API Configuration](#api-configuration)
8. [Troubleshooting](#troubleshooting)
9. [Project Modules](#project-modules)

---

## Overview

This is an **AI-powered Object Memory Assistant** that:
- ✓ Detects objects in real-time from webcam
- ✓ Tracks objects across frames (ByteTrack)
- ✓ Deduplicates frames using embedding similarity
- ✓ Generates scene descriptions via Gemini Vision API
- ✓ Stores object history in SQLite database
- ✓ Provides natural language query interface
- ✓ Runs on desktop (GPU) and Raspberry Pi (TensorFlow Lite)

**Use Case**: Help visually impaired users find misplaced objects by remembering where they were last seen with detailed location descriptions.

---

## System Requirements

### For Desktop/Development
- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 - 3.11
- **RAM**: 4GB minimum (8GB recommended for GPU models)
- **GPU**: Optional (NVIDIA GPU with CUDA support recommended)
- **Storage**: 2GB free space (+ storage for frames)
- **Webcam**: USB or built-in

### For Raspberry Pi
- **Model**: Raspberry Pi 4B (4GB+ RAM) or Pi 5
- **OS**: Raspberry Pi OS (64-bit recommended)
- **Storage**: 32GB SD card minimum
- **Memory**: 4GB+ RAM
- **Power**: Good quality 5V/3A power supply
- **Optional**: USB Accelerator (Google Coral TPU)

---

## Installation Guide

### Step 1: Clone Repository

```bash
cd ~/projects  # or your preferred location
git clone https://github.com/JaineelPandya/yolo_ipd.git
cd yolo_ipd
```

### Step 2: Create Virtual Environment

#### On Windows:
```batch
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install Streamlit
pip install streamlit

# Install Google Generative AI
pip install google-generativeai

# Install additional dependencies
pip install scikit-learn torchvision
```

### Step 4: Download Models

For desktop (PyTorch):
```bash
# Models will auto-download on first run, or manually:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

For Raspberry Pi (TensorFlow Lite):
```bash
# Download TensorFlow Lite model
pip install tensorflow

# Convert YOLOv8 to TFLite (if needed)
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='tflite')"
```

### Step 5: Configure API Key

Create `.env` file in project root:
```bash
cat > .env << EOF
GEMINI_API_KEY=AIzaSyAoE9gpy6QB2__OqNqfiXTeu6qFQ7_idjc
EOF
```

Or set environment variable:
```bash
export GEMINI_API_KEY=AIzaSyAoE9gpy6QB2__OqNqfiXTeu6qFQ7_idjc
```

### Step 6: Test Installation

```bash
# Test detector
python -c "from detection.detector import create_detector; d = create_detector(); print('✓ Detector OK')"

# Test database
python -c "from memory.storage import create_memory; m = create_memory(); print('✓ Database OK')"

# Test Streamlit
streamlit run app_streamlit.py
```

---

## Running Streamlit Web Interface

### Standard Launch

```bash
streamlit run app_streamlit.py
```

This opens the app at `http://localhost:8501`

### Advanced Configuration

Create `~/.streamlit/config.toml`:
```toml
[logger]
level = "info"

[client]
showErrorDetails = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#00AA88"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#333333"
font = "sans serif"

[server]
port = 8501
headless = false
runOnSave = false
```

### Using with GPU

```bash
# Set automatic GPU detection
CUDA_VISIBLE_DEVICES=0 streamlit run app_streamlit.py
```

### Alternative: Run with Python Script

```bash
python main_app.py --mode streamlit --camera 0 --conf 0.5 --enable-gemini
```

---

## Raspberry Pi Deployment

### Prerequisites

1. **Fresh Raspberry Pi OS Installation**
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.9+
sudo apt-get install -y python3 python3-venv python3-pip

# Install system dependencies
sudo apt-get install -y \
    libatlas-base-dev libjasper-dev libtiff5 libjasper1 \
    libharfbuzz0b libwebp6 libtiff5 \
    libopenjp2-7 libjasper1 libatlas-base-dev libblas3 liblapack3 libharfbuzz0b libwebp6
```

2. **Increase Swap (for model conversion)**
```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile swapon
```

### Installation on Raspberry Pi

```bash
# Clone and setup
cd ~
git clone https://github.com/JaineelPandya/yolo_ipd.git
cd yolo_ipd

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies (optimized for RPi)
pip install --upgrade pip
pip install -r requirements_rpi.txt  # Use RPi-specific requirements

# Download TensorFlow Lite model
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.tflite -O models/yolov8n.tflite
```

### requirements_rpi.txt

Create this file for Raspberry Pi:
```
ultralytics==8.0.0
opencv-python==4.8.0.74
numpy==1.21.0
torch==2.0.0
tensorflow-lite==2.13.0
google-generativeai==0.3.0
streamlit==1.28.0
Pillow==10.0.0
```

### Running on Raspberry Pi

**Option 1: Streamlit (Recommended for UI)**
```bash
# From project directory
streamlit run app_streamlit.py \
    --client.toolbarMode=minimal \
    --client.showErrorDetails=false
```

**Option 2: Direct Python (Headless)**
```bash
python main_app.py \
    --camera 0 \
    --mode headless \
    --enable-tflite \
    --conf 0.5
```

**Option 3: Docker (Easiest)**
```bash
# Build image
docker build -t yolo-memory:rpi -f Dockerfile.rpi .

# Run container
docker run -it \
    --device /dev/video0:/dev/video0 \
    -p 8501:8501 \
    yolo_memory:rpi
```

### Raspberry Pi Folder Structure

```
yolo_ipd/
├── models/
│   ├── yolov8n.tflite          # TensorFlow Lite model (for RPi)
│   └── yolov8n.pt              # PyTorch model (for desktop)
├── data/
│   ├── database/
│   │   └── object_memory.db    # SQLite database
│   └── frames/
│       ├── compressed/         # Small frames for storage
│       └── full/               # Full resolution frames
├── detection/
├── tracking/
├── memory/
├── gemini_api/
├── query/
├── utils/
├── app_streamlit.py            # Main Streamlit app
├── config.py                   # Configuration (auto-detects RPi)
└── README_SETUP.md             # This file
```

### Files to Copy to Raspberry Pi

Minimum files needed:
```bash
# Essential
detection/          → copy
tracking/           → copy
memory/             → copy
gemini_api/         → copy
query/              → copy
utils/              → copy
config.py           → copy (auto-detects RPi)
app_streamlit.py    → copy
requirements_rpi.txt → copy

# Optional
METHODOLOGY.md      → copy (documentation)
README.md           → copy (original readme)
```

### Performance Tips for Raspberry Pi

1. **Reduce Frame Rate**
   - Edit `config.py`:
   ```python
   INFERENCE_SKIP_FRAMES = 3  # Process every 3rd frame
   FRAME_COMPRESSION_QUALITY = 70  # Lower quality
   COMPRESSED_FRAME_SIZE = (240, 180)  # Smaller size
   ```

2. **Disable Unnecessary Features**
   ```python
   enable_gemini = False  # Disable API calls on weak connection
   USE_TFLITE = True  # Always use TFLite
   ```

3. **Monitor Resources**
   ```bash
   # Monitor CPU/Memory while running
   top -b -n 1 | head -20
   
   # Check temperature
   vcgencmd measure_temp
   ```

4. **Use Headless Mode**
   - Run without Streamlit GUI for faster inference
   - SSH into RPi and use command-line

5. **Enable GPU Acceleration** (if available)
   ```bash
   # Install Coral TPU support
   curl https://apt.coral.ai/apt.gpg | apt-key add -
   echo "deb https://apt.coral.ai/ bullseye main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
   sudo apt-get update
   sudo apt-get install -y libedgetpu1-std
   ```

---

## Database Structure

### SQLite Tables

#### objects
```sql
CREATE TABLE objects (
    id INTEGER PRIMARY KEY,
    object_name TEXT,           -- "phone", "keys", etc.
    class_id INTEGER,           -- COCO class ID
    track_id INTEGER,           -- Tracker ID
    timestamp DATETIME,         -- When detected
    bbox TEXT,                  -- JSON: [x1, y1, x2, y2]
    confidence REAL,            -- 0-1 score
    scene_description TEXT,     -- From Gemini API
    image_path TEXT,            -- Full resolution frame
    compressed_image_path TEXT, -- Compressed frame
    latitude REAL,              -- For future features
    longitude REAL,
    notes TEXT                  -- User notes
)
```

#### frames
```sql
CREATE TABLE frames (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    image_path TEXT UNIQUE,
    compressed_image_path TEXT,
    embedding_vector BLOB,      -- ResNet embeddings
    scene_description TEXT,
    num_objects INTEGER,
    frame_hash TEXT UNIQUE      -- SHA256 for dedup
)
```

### Querying the Database

```bash
# Open SQLite shell
sqlite3 data/database/object_memory.db

# Useful queries
.tables                                  # List tables
SELECT COUNT(*) FROM objects;            # Total detections
SELECT object_name, MAX(timestamp) FROM objects GROUP BY object_name;  # Latest per object
SELECT * FROM objects WHERE timestamp > datetime('now', '-1 hour');  # Last hour
```

---

## API Configuration

### Gemini Vision API Setup

1. **Get API Key**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create new API key
   - Copy key

2. **Set in Configuration**
   ```python
   # In config.py
   GEMINI_API_KEY = "Your_API_Key_Here"
   GEMINI_MODEL = "gemini-1.5-vision"
   ```

3. **Verify It Works**
   ```python
   from gemini_api.descriptor import create_scene_descriptor
   
   descriptor = create_scene_descriptor()
   description = descriptor.describe_scene(frame)
   print(description)
   ```

### Rate Limiting

Gemini API has free tier limits:
- 60 requests per minute
- Images under 20MB

The system automatically handles this with:
```python
# In config.py
GEMINI_SCENE_PROMPT = "..."  # System prompt
```

---

## Troubleshooting

### Webcam Issues

```bash
# Check if webcam is detected
ls /dev/video*

# Test with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# Try different camera ID
python app_streamlit.py --camera 1  # Try camera 1 instead of 0
```

### Memory/Performance Issues

```python
# In config.py, reduce:
MAX_STORED_FRAMES = 500      # From 1000
FRAME_COMPRESSION_QUALITY = 70  # From 85
COMPRESSED_FRAME_SIZE = (240, 180)  # From (320, 240)
```

### Gemini API Errors

```bash
# Check API key
python -c "import google.generativeai as genai; print('✓ API available')"

# Test with simple prompt
python -c "from gemini_api.descriptor import create_scene_descriptor; d = create_scene_descriptor(); print(d.client)"
```

### Database Locked Error

```python
# Clear zombie processes
import sqlite3
conn = sqlite3.connect('data/database/object_memory.db')
conn.execute('VACUUM')
conn.close()
```

### TensorFlow Lite Model Issues

```bash
# Convert YOLOv8 to TFLite
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='tflite')
"

# Check if conversion successful
ls models/yolov8n.tflite
```

---

## Project Modules

### detection/
Handles object detection with PyTorch and TensorFlow Lite support
- `detector.py`: YOLODetector class with PT/TFLite inference

### tracking/
Tracks objects across frames using ByteTrack
- `tracker.py`: ByteTracker class, TrackedObject dataclass

### memory/
SQLite database for storing object history
- `storage.py`: ObjectMemory class with CRUD operations

### gemini_api/
Generates scene descriptions using Gemini Vision API
- `descriptor.py`: GeminiSceneDescriptor class

### query/
Natural language query interface
- `engine.py`: ObjectQueryEngine class
- `get_last_seen(object_name)`: Main query function

### utils/
Utility functions and performance monitoring
- `deduplicator.py`: Frame deduplication using embeddings
- `helpers.py`: FrameProcessor, RaspberryPiOptimizer, PerformanceMonitor

### config.py
Central configuration file
- Auto-detects Raspberry Pi
- Selects TFLite or PyTorch automatically
- Configurable thresholds and parameters

---

## Configuration Quick Reference

### Key Config Parameters

```python
# Hardware
IS_RASPBERRY_PI              # Auto-detected
USE_TFLITE                   # Auto-selected based on hardware
INFERENCE_DEVICE             # "cpu" or GPU ID

# Detection
CONFIDENCE_THRESHOLD         # 0.5 (minimum 0.4 recommended)
IOU_THRESHOLD                # 0.45
MODEL_INPUT_SIZE             # (640, 640)

# Tracking
TRACKER_TYPE                 # "bytetrack"
MAX_AGE                       # 30 frames
MIN_HITS                      # 3 detections
MAX_TRACKS                    # 100

# Frame Deduplication
SIMILARITY_THRESHOLD          # 0.85 (0-1, higher = stricter)
LOCATION_CHANGE_THRESHOLD    # 50 pixels
CONFIDENCE_IMPROVEMENT_THRESHOLD  # 0.1 (10%)

# Memory
MAX_STORED_FRAMES            # 1000
FRAME_COMPRESSION_QUALITY    # 85 (1-100)

# API
GEMINI_API_KEY               # Your API key
GEMINI_MODEL                 # "gemini-1.5-vision"

# Objects to Track (Priority)
SMALL_OBJECTS_TO_TRACK       # ['phone', 'keys', 'wallet', ...]
```

---

## Running the Complete System

### Desktop (Development)

```bash
# Terminal 1: Start Streamlit UI
cd ~/projects/yolo_ipd
source venv/bin/activate
streamlit run app_streamlit.py

# Open browser to http://localhost:8501
# Click "Start Camera" in Live Detection tab
```

### Raspberry Pi (Production)

```bash
# SSH into RPi
ssh pi@raspberry.local

# Start Streamlit
cd ~/yolo_ipd
source venv/bin/activate
streamlit run app_streamlit.py \
    --client.toolbarMode=minimal \
    --logger.level=error

# Access from any device on network: http://raspberrypi.local:8501
```

---

## Next Steps

1. **Train Custom Model** (Optional)
   - Use your own images
   - Train on specific objects
   - Better accuracy for your use case

2. **Add Voice Input**
   - Integrate speech-to-text
   - Use with Google Speech-to-Text API

3. **Mobile App**
   - React Native or Flutter wrapper
   - Connect to same database

4. **Cloud Sync**
   - Sync database to cloud
   - Access from multiple devices

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [Project Modules](#project-modules) documentation
3. Check console logs: `tail -f logs/system.log`
4. Test individual modules with provided test scripts

---

## License & Attribution

Original YOLO Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

This assistive technology modification: AI Object Memory Assistant
