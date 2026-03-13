# 🚀 QUICKSTART - AI Object Memory Assistant

## For Now: Running on Windows/Mac/Linux (Desktop)

### ✨ LATEST UPDATE (March 14, 2026)

**🎉 NEW FEATURES:**
- ✅ **Shows ALL detections** with bounding boxes (not just small objects)
- ✅ **Detection confidence slider** now starts at 0.25 for better sensitivity
- ✅ **Real-time detection count** shows total, small, and other objects
- ✅ **Fixed Streamlit media cache issues** - no more "missing file" errors
- ✅ **Detection diagnostic tool** - run `python test_detection.py`

### 1️⃣ **Quick Setup (5 minutes)**

```bash
# Clone or download the project
cd yolo_ipd

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install everything
python setup_system.py
```

### 2️⃣ **Run Streamlit Web Interface** ⭐ Recommended

```bash
streamlit run app_streamlit.py
```

Then open your browser to: **http://localhost:8501**

**Streamlit Interface Tabs:**
- 🎥 **Live Detection**: See real-time detections from webcam
- 🔍 **Query Objects**: Ask "Where is my phone?" and get answers
- 📊 **Statistics**: View tracked objects and database stats
- 📚 **History**: See where objects were previously found

### 3️⃣ **Run Command-Line Interface**

```bash
# Start webcam detection
python main_app.py

# Query for an object
python main_app.py --query phone

# Show statistics
python main_app.py --stats

# See all options
python main_app.py --help
```

### 4️⃣ **What Happens When You Run It**

1. **Camera starts** → YOLOv8 detects objects
2. **Objects tracked** → ByteTrack follows them across frames
3. **Frame evaluated** → Deduplicator checks if worth saving
4. **Important frames** → Sent to Gemini Vision API
5. **Scene described** → "Your phone is on the wooden nightstand"
6. **Stored in database** → SQLite remembers for later
7. **You ask** → "Where's my phone?" → System responds with location

---

## 🍓 Future: For Raspberry Pi

### Folder Structure on Raspberry Pi

When deploying to Raspberry Pi, you need to transfer these folders/files:

**Essential (always needed):**
```
yolo_ipd/
├── detection/           ← Object detection
├── tracking/            ← Object tracking  
├── memory/              ← Database
├── gemini_api/          ← API calls
├── query/               ← Find objects
├── utils/               ← Helper functions
├── config.py            ← Configuration (auto-detects RPi)
└── app_streamlit.py     ← Web interface OR main_app.py for CLI
```

**Models folder (put in ~/yolo_ipd/models/):**
```
models/
├── yolov8n.pt          ← PyTorch model (90MB, for desktop)
└── yolov8n.tflite      ← TensorFlow Lite (30MB, for RPi - SMALLER!)
```

**Data folders (will be created automatically):**
```
data/
├── database/
│   └── object_memory.db ← SQLite database
└── frames/
    ├── compressed/     ← Small images (storage efficient)
    └── full/           ← Full resolution frames
```

### Raspberry Pi Installation Steps

1. **SSH into Pi:**
   ```bash
   ssh pi@raspberrypi.local  # or pi@192.168.x.x
   cd ~
   ```

2. **Clone project:**
   ```bash
   git clone https://github.com/JaineelPandya/yolo_ipd.git
   cd yolo_ipd
   ```

3. **Setup (same as above):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements_rpi.txt  # Note: requirements_RPI
   ```

4. **Run on Pi:**
   ```bash
   # Streamlit (Web UI, accessible from any device)
   streamlit run app_streamlit.py
   # Access from http://raspberrypi.local:8501
   
   # OR command-line (headless, faster)
   python main_app.py
   ```

### What Files to Load on Raspberry Pi

**Copy these files to your RPi:**
```bash
# From your computer:
scp -r detection/ pi@raspberrypi.local:~/yolo_ipd/
scp -r tracking/ pi@raspberrypi.local:~/yolo_ipd/
scp -r memory/ pi@raspberrypi.local:~/yolo_ipd/
scp -r gemini_api/ pi@raspberrypi.local:~/yolo_ipd/
scp -r query/ pi@raspberrypi.local:~/yolo_ipd/
scp -r utils/ pi@raspberrypi.local:~/yolo_ipd/
scp config.py pi@raspberrypi.local:~/yolo_ipd/
scp app_streamlit.py pi@raspberrypi.local:~/yolo_ipd/
scp main_app.py pi@raspberrypi.local:~/yolo_ipd/

# Optional documentation:
scp README_SETUP.md pi@raspberrypi.local:~/yolo_ipd/
```

**Or copy the entire folder:**
```bash
scp -r yolo_ipd/ pi@raspberrypi.local:~/
```

---

## 🎮 Using the System

### Example 1: Find Your Phone

1. **Leave your phone somewhere**
2. **Run the system** - it records where it's placed
3. **Later, ask**: "Where is my phone?"
4. **Get response**: "Your phone was last seen at 2:30 PM on the wooden nightstand near the lamp."

### Example 2: Find Your Keys

```bash
python main_app.py --query keys
```

Output:
```
Query Result:
Your keys were last seen at 10:15 AM inside the ceramic bowl on the kitchen counter.
```

### Example 3: Track Everything Today

```bash
python main_app.py --stats
```

Shows all objects detected today with their last locations.

---

## 🎯 Configuration

### What Objects Are Tracked

Edit `config.py`, change `SMALL_OBJECTS_TO_TRACK`:

```python
SMALL_OBJECTS_TO_TRACK = [
    'cell phone', 'phone', 'keys', 'wallet', 'glasses', 'watch',
    'earbuds', 'headphones', 'pen', 'pen', 'notebook', 'book',
    'cup', 'glass', 'bottle', 'remote', 'controller', 'mouse',
    # Add more objects here
]
```

### Adjust Detection Settings

In `config.py`:
```python
# Only detect confident objects
CONFIDENCE_THRESHOLD = 0.6  # default 0.5

# Sensitivity to object movement
LOCATION_CHANGE_THRESHOLD = 50  # pixels

# Store more/fewer frames
MAX_STORED_FRAMES = 500  # Raspberry Pi: reduce this
```

### Enable/Disable Gemini API

In Streamlit app:
- Toggle "Enable Scene Description" checkbox in left sidebar
- Or in code:
  ```python
  enable_gemini = False  # Set to False to disable API calls
  ```

---

## 🆘 Troubleshooting

### ⚠️ Model Not Detecting Objects (Phone, Keys, etc.)

**Quick fixes (in order):**

1. **Lower the confidence threshold:**
   - In Streamlit, set slider to **0.20-0.25** (left sidebar)
   - Default is now 0.25, lower = more detections

2. **Test the detection system:**
   ```bash
   python test_detection.py
   ```
   This shows you exactly what the model is detecting at different thresholds

3. **Check your webcam:**
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print('✓ Webcam OK' if cap.isOpened() else '❌ Webcam FAILED')"
   ```

4. **Improve lighting:**
   - Good lighting is essential for YOLOv8
   - Natural sunlight or bright room light works best
   - Avoid shadows and backlighting

5. **Position objects properly:**
   - Hold phone in hand (not in pocket)
   - Object should be clear and unobstructed
   - Keep object in center of view

6. **Check model file:**
   - Model should be in `models/yolov8n.pt`
   - If missing, run: `python setup_system.py`

### 📸 Improving Detection Accuracy

- **Increase exposure:** Better lighting helps significantly
- **Move closer:** Objects should be clearly visible
- **Try different angles:** Some angles work better than others
- **Adjust threshold:** Lower threshold catches more, higher is stricter
- **Train custom model:** For specific items, create custom YOLO model

### 🎯 Understanding Confidence Threshold

```
Confidence 0.10 = Very permissive (many false positives)
Confidence 0.25 = Good default for most use cases ← START HERE
Confidence 0.50 = Original strict threshold
Confidence 0.75 = Very strict (misses many real objects)
```

**Recommended:** Start at 0.25 and adjust up/down based on results

### Webcam Not Working
```bash
# Check if camera is detected
ls /dev/video*

# Test with Python
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# Try different camera ID
streamlit run app_streamlit.py
# Then in sidebar, change confidence threshold
```

### Database Locked
```bash
# Reset database (will lose history)
rm data/database/object_memory.db
```

### Frame Display Issues (Missing Media Files)
- **FIXED in this update!**
- App now displays frames directly from memory
- No more "missing file" errors

### ❌ Still Not Working?

1. Check logs:
   ```bash
   tail -f logs/system.log
   ```

2. Run diagnostics:
   ```bash
   python test_detection.py
   ```

3. Check model loading:
   ```bash
   python -c "from detection.detector import create_detector; d = create_detector(); print('✓ Model OK' if d else '❌ Model FAILED')"
   ```

---

## 📱 How It Works (Simplified)

```
Your Webcam
    ↓
[YOLOv8 Detection] → Detects objects
    ↓
[ByteTrack] → Tracks objects across frames
    ↓
[Frame Dedup] → Decides if frame is worth saving
    ↓
[✓ Important Frame] → Yes, save it
    ↓
[Gemini Vision API] → "Describe this scene"
    ↓
[SQLite Database] → Store: Object name, timestamp, location description
    ↓
[You ask] → "Where is my phone?"
    ↓
[Query Engine] → Searches database
    ↓
[Response] → "Last seen at 2:30 PM on your nightstand"
```

---

## 🚀 Next Steps

### To Improve Accuracy
1. Add more objects to `SMALL_OBJECTS_TO_TRACK` list
2. Collect frames of your items
3. Train custom YOLOv8 model on your data

### For Production Use
1. Run on Raspberry Pi with TensorFlow Lite
2. Set up automatic startup (systemd service)
3. Use better camera (USB camera for RPi)
4. Store database in cloud for sync

### For Voice Input
```bash
pip install SpeechRecognition pyaudio
# Then ask: "Hey, where's my phone?"
```

---

## 📚 Important Files Reference

| File | Purpose |
|------|---------|
| `app_streamlit.py` | Web interface (recommended for users) |
| `main_app.py` | Command-line interface |
| `config.py` | All configuration settings |
| `detection/detector.py` | YOLOv8 inference (PyTorch + TFLite) |
| `tracking/tracker.py` | ByteTrack object tracking |
| `memory/storage.py` | SQLite database operations |
| `gemini_api/descriptor.py` | Google Gemini Vision API calls |
| `query/engine.py` | Natural language search |
| `utils/deduplicator.py` | Frame deduplication |
| `utils/helpers.py` | Utilities and helpers |

---

## 🎓 Common Commands

```bash
# Setup
python setup_system.py

# Run Streamlit (web UI)
streamlit run app_streamlit.py

# Run command-line
python main_app.py

# Query for object
python main_app.py --query phone

# Show statistics
python main_app.py --stats

# Show object history
python main_app.py --history keys

# Clean old data
python main_app.py --cleanup

# Use TensorFlow Lite
python main_app.py --use-tflite

# Enable Gemini API
python main_app.py --enable-gemini

# Get help
python main_app.py --help
```

---

## 🤝 Support

- Check `README_SETUP.md` for detailed setup
- Check `METHODOLOGY.md` for technical details
- Look at console output for error messages
- Check `logs/system.log` for detailed logs

---

**Happy tracking! 🎯**
