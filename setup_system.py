"""
Quick setup script for Object Memory Assistant
Run: python setup_system.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command"""
    if description:
        print(f"\n{'='*60}")
        print(f"📦 {description}")
        print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    dirs = [
        "data/database",
        "data/frames/compressed",
        "data/frames/full",
        "logs",
        "models",
        "outputs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, but {version.major}.{version.minor} found")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if basic dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("  ✗ OpenCV not found)")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError:
        print("  ✗ NumPy not found")
        return False
    
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📥 Installing dependencies...")
    
    if platform.system() == "Linux" and Path("/proc/device-tree/model").exists():
        # Raspberry Pi
        print("  Detected Raspberry Pi - using optimized requirements")
        run_command(
            "pip install --upgrade pip",
            "Upgrading pip"
        )
        run_command(
            "pip install -r requirements_rpi.txt",
            "Installing Raspberry Pi requirements"
        )
    else:
        # Desktop/Development
        print("  Detected Desktop/Development environment")
        run_command(
            "pip install --upgrade pip",
            "Upgrading pip"
        )
        run_command(
            "pip install -r requirements.txt",
            "Installing requirements"
        )
        run_command(
            "pip install google-generativeai",
            "Installing Gemini API"
        )

def download_models():
    """Download YOLOv8 models"""
    print("\n🤖 Downloading models...")
    
    try:
        from ultralytics import YOLO
        
        # Download PyTorch model
        print("  Downloading YOLOv8 Nano (PyTorch)...")
        model = YOLO('yolov8n.pt')
        print("  ✓ yolov8n.pt downloaded")
        
        # Try to convert to TFLite (if TensorFlow available)
        try:
            print("  Converting to TensorFlow Lite...")
            model.export(format='tflite')
            print("  ✓ yolov8n.tflite created")
        except:
            print("  ⚠️  TensorFlow Lite conversion failed (optional)")
    
    except Exception as e:
        print(f"  ✗ Error downloading models: {e}")
        return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    print("\n⚙️  Setting up environment...")
    
    # Check for API key
    if not os.environ.get('GEMINI_API_KEY'):
        print("  ⚠️  GEMINI_API_KEY not set")
        print("  You can set it now or configure in config.py later")
        api_key = input("  Enter your Gemini API key (or press Enter to skip): ").strip()
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
            with open('.env', 'w') as f:
                f.write(f"GEMINI_API_KEY={api_key}\n")
            print("  ✓ Saved to .env file")
    else:
        print("  ✓ GEMINI_API_KEY found in environment")

def test_installation():
    """Test if installation was successful"""
    print("\n🧪 Testing installation...")
    
    try:
        print("  Testing detector...", end=" ")
        from detection import create_detector
        detector = create_detector()
        print("✓")
    except Exception as e:
        print(f"✗ {e}")
        return False
    
    try:
        print("  Testing tracker...", end=" ")
        from tracking import create_tracker
        tracker = create_tracker()
        print("✓")
    except Exception as e:
        print(f"✗ {e}")
        return False
    
    try:
        print("  Testing memory...", end=" ")
        from memory import create_memory
        memory = create_memory()
        memory.close()
        print("✓")
    except Exception as e:
        print(f"✗ {e}")
        return False
    
    print("\n✓ All components working!")
    return True

def main():
    """Main setup routine"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     🤖 AI Object Memory Assistant - Setup Script 🤖          ║
║                                                              ║
║  This script will set up the complete system for you.       ║
║  It will:                                                    ║
║  1. Check Python version                                    ║
║  2. Create necessary directories                            ║
║  3. Install Python dependencies                             ║
║  4. Download YOLOv8 models                                  ║
║  5. Test the installation                                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    input("Press Enter to continue...\n")
    
    # Step 1: Check Python
    if not check_python():
        print("❌ Setup failed!")
        return 1
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Check existing dependencies
    if not check_dependencies():
        print("⚠️  Some dependencies missing, installing...")
    
    # Step 4: Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return 1
    
    # Step 5: Download models
    try:
        download_models()
    except Exception as e:
        print(f"⚠️  Model download error (non-critical): {e}")
    
    # Step 6: Setup environment
    setup_environment()
    
    # Step 7: Test installation
    if not test_installation():
        print("❌ Installation tests failed!")
        return 1
    
    # Success!
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   ✅ SETUP COMPLETE! ✅                      ║
║                                                              ║
║  Next steps:                                                 ║
║                                                              ║
║  1. Run CLI interface:                                      ║
║     python main_app.py                                      ║
║                                                              ║
║  2. Run with TFLite (Raspberry Pi):                         ║
║     python main_app.py --use-tflite                         ║
║                                                              ║
║  3. Query for an object:                                    ║
║     python main_app.py --query phone                        ║
║                                                              ║
║  For Raspberry Pi deployment, see TFLITE/DEPLOY_GUIDE.md    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
