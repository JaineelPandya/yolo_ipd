#!/usr/bin/env python3
"""
YOLOv8n model downloader + TFLite exporter.

Usage:
    python download_model.py            # download .pt and export to TFLite
    python download_model.py --pt-only  # download .pt only (desktop/PyTorch)
    python download_model.py --skip-export  # download .pt, skip TFLite export
"""

import argparse
import urllib.request
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# YOLOv8n PyTorch weights (~6 MB) — official Ultralytics asset
PT_URLS = [
    "https://github.com/ultralytics/assets/releases/latest/download/yolov8n.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
]

PT_PATH     = MODELS_DIR / "yolov8n.pt"
TFLITE_PATH = MODELS_DIR / "yolov8n_float32.tflite"


# ──────────────────────────────────────────────────────────────────────────────

def progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(block_num * block_size / total_size * 100, 100)
    bar = int(pct // 2)
    print(f"\r  [{'█' * bar:<50s}] {pct:.1f}%", end="", flush=True)


def download_pt():
    """Download YOLOv8n.pt if not already present."""
    if PT_PATH.exists():
        print(f"✓ PyTorch model already present: {PT_PATH}")
        return

    last_err = None
    for url in PT_URLS:
        try:
            print(f"⬇  Downloading from: {url}")
            urllib.request.urlretrieve(url, PT_PATH, progress)
            print(f"\n✓ Saved: {PT_PATH}  ({PT_PATH.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as e:
            print(f"\n⚠  Failed ({e}), trying next URL...")
            last_err = e

    print(f"\n✗ All download attempts failed: {last_err}")
    sys.exit(1)


def export_tflite():
    """Export yolov8n.pt → yolov8n_float32.tflite using ultralytics."""
    if TFLITE_PATH.exists():
        print(f"✓ TFLite model already present: {TFLITE_PATH}")
        return

    print("\n🔄 Exporting .pt → TFLite (float32)...")
    print("   This requires: pip install ultralytics")

    try:
        from ultralytics import YOLO
        model = YOLO(str(PT_PATH))
        export_path = model.export(format="tflite")   # returns path string

        # ultralytics saves it next to the .pt; move to models/ if needed
        import shutil
        exported = Path(str(export_path))
        if exported.exists() and exported != TFLITE_PATH:
            shutil.move(str(exported), str(TFLITE_PATH))
            # also clean up any leftover *_saved_model/ directory
            saved_model_dir = exported.parent / (exported.stem + "_saved_model")
            if saved_model_dir.exists():
                shutil.rmtree(saved_model_dir, ignore_errors=True)

        if TFLITE_PATH.exists():
            print(f"✓ TFLite model saved: {TFLITE_PATH}  "
                  f"({TFLITE_PATH.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"⚠  Export succeeded but file not found at {TFLITE_PATH}")
            print(f"   Look for it at: {export_path}")

    except ImportError:
        print("\n✗ ultralytics is not installed.")
        print("  Install with:  pip install ultralytics")
        print("  Then re-run:   python download_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download YOLOv8n model and export to TFLite"
    )
    parser.add_argument("--pt-only", action="store_true",
                        help="Only download .pt (for desktop PyTorch use)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Download .pt but skip TFLite export")
    args = parser.parse_args()

    download_pt()

    if args.pt_only or args.skip_export:
        print("\n✅ PyTorch model ready.")
        print("   Run:  streamlit run app_streamlit.py")
        return

    export_tflite()

    print("\n✅ All models ready.")
    print("   TFLite (RPi): streamlit run app_streamlit.py  (select TFLite in sidebar)")
    print("   PyTorch (PC):  streamlit run app_streamlit.py  (select PyTorch in sidebar)")


if __name__ == "__main__":
    main()
