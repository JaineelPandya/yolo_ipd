#!/usr/bin/env python3
"""
Verification script to check if Gemini API is working correctly
Run: python verify_setup.py
"""

import sys
import os

print("\n" + "="*70)
print("🔍 YOLO IPD - Gemini API Verification")
print("="*70)

# 1. Check Python version
print("\n1️⃣  Python Version")
version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"   Version: {version}")
if sys.version_info >= (3, 9):
    print("   ✅ OK (3.9+)")
else:
    print("   ❌ FAIL - Need Python 3.9 or higher")

# 2. Check required packages
print("\n2️⃣  Required Packages")
packages = {
    'cv2': 'OpenCV',
    'ultralytics': 'Ultralytics',
    'torch': 'PyTorch',
    'streamlit': 'Streamlit',
    'google': 'Google Generative AI',
    'numpy': 'NumPy',
    'sqlite3': 'SQLite3 (built-in)'
}

for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - Run: pip install {pkg}")

# 3. Check .env file
print("\n3️⃣  Environment Setup")
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("   ✅ python-dotenv available")
except ImportError:
    print("   ⚠️  python-dotenv not found (optional)")
    print("       Will load GEMINI_API_KEY from .env manually")

# 4. Check API Key
print("\n4️⃣  Gemini API Key")
import config
if config.GEMINI_API_KEY:
    key_preview = config.GEMINI_API_KEY[:20] + "..." + config.GEMINI_API_KEY[-4:]
    print(f"   ✅ Found: {key_preview}")
    print(f"      Length: {len(config.GEMINI_API_KEY)} chars")
else:
    print("   ❌ GEMINI_API_KEY not found in .env")
    print("      Add to .env: GEMINI_API_KEY=your_key_here")

# 5. Check model files
print("\n5️⃣  Model Files")
from pathlib import Path
model_path = Path("models/yolov8n.pt")
if model_path.exists():
    size_mb = model_path.stat().st_size / 1e6
    print(f"   ✅ yolov8n.pt ({size_mb:.1f} MB)")
else:
    print("   ❌ yolov8n.pt not found")
    print("      Run: python download_model.py --skip-export")

# 6. Check database
print("\n6️⃣  Database")
db_path = Path(config.DATABASE_PATH)
if db_path.exists():
    size_kb = db_path.stat().st_size / 1e3
    print(f"   ✅ Database exists ({size_kb:.1f} KB)")
else:
    print("   ℹ️  Database will be created on first run")

# 7. Check Gemini client initialization
print("\n7️⃣  Gemini Client")
try:
    from gemini_api.descriptor import create_scene_descriptor
    descriptor = create_scene_descriptor()
    
    if descriptor and descriptor.client:
        print(f"   ✅ Gemini client initialized")
        print(f"      Model: {descriptor.model_name}")
        print(f"      Rate limit: {config.GEMINI_MIN_INTERVAL_SECONDS}s")
    elif descriptor:
        print(f"   ⚠️  Descriptor created but client is None")
        if not config.GEMINI_API_KEY:
            print("      → Check GEMINI_API_KEY in .env")
        else:
            print("      → Check internet connection and API key validity")
    else:
        print("   ❌ Failed to create descriptor")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 8. Test YOLOv8 loading
print("\n8️⃣  YOLOv8 Model")
try:
    from ultralytics import YOLO
    model = YOLO(str(Path("models/yolov8n.pt")))
    print(f"   ✅ YOLOv8n model loaded")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")

# Summary
print("\n" + "="*70)
print("✅ Setup verification complete!")
print("="*70)
print("\n📋 NEXT STEPS:")
print("  1. If all checks passed: streamlit run app_streamlit.py")
print("  2. Enable Gemini in sidebar")
print("  3. Start camera and detect some objects")
print("  4. Search for them in Query Objects tab")
print("\n💡 For detailed guide, see: GEMINI_SETUP_GUIDE.md")
print("="*70 + "\n")
