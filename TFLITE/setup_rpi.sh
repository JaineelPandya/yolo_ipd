#!/bin/bash
# ============================================================
# Raspberry Pi Setup Script for TFLite Object Memory Assistant
# Run: chmod +x setup_rpi.sh && ./setup_rpi.sh
# ============================================================

set -e

echo "========================================"
echo "🍓 RPi TFLite Setup"
echo "========================================"

# 1. System dependencies
echo ""
echo "📦 Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    libopencv-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev

# 2. Create virtual environment
echo ""
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install Python dependencies
echo ""
echo "📥 Installing Python packages..."
pip install -r requirements_rpi.txt

# 5. Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p models data/database data/frames logs

# 6. Check model file
echo ""
if [ -f "models/yolov8n_float32.tflite" ]; then
    echo "✓ TFLite model found!"
else
    echo "⚠️  Model file NOT found at models/yolov8n_float32.tflite"
    echo "   Copy it from your PC:"
    echo "   scp yolov8n_float32.tflite pi@<PI_IP>:~/object_memory/models/"
fi

# 7. Test
echo ""
echo "🧪 Testing TFLite import..."
python3 -c "
try:
    import tflite_runtime.interpreter as tflite
    print('  ✓ tflite-runtime OK')
except ImportError:
    print('  ✗ tflite-runtime not found')
import cv2
print(f'  ✓ OpenCV {cv2.__version__}')
import numpy
print(f'  ✓ NumPy {numpy.__version__}')
"

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy model:  scp yolov8n_float32.tflite pi@<PI_IP>:~/object_memory/models/"
echo "  2. Activate:    source venv/bin/activate"
echo "  3. Run:         python main_app.py --use-tflite"
echo "  4. Headless:    python main_app.py --use-tflite --headless"
echo "========================================"
