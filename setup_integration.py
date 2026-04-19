#!/usr/bin/env python3
"""
Setup script for Gemini + Database Integration
Installs all dependencies and verifies setup
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run command and report status"""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e.stderr[:200]}")
        return False

def main():
    print("\n" + "="*70)
    print("🚀 GEMINI + DATABASE INTEGRATION - SETUP")
    print("="*70)
    
    # Check Python
    print(f"\nPython version: {sys.version.split()[0]}")
    
    # Install dependencies
    print("\n📋 Installing dependencies...")
    
    packages = [
        "opencv-python",
        "numpy",
        "sqlite3",
        "google-generativeai",
        "sentence-transformers",
        "torch",
        "faiss-cpu",
    ]
    
    # Try installing each package
    for package in packages:
        if package == "sqlite3":
            # sqlite3 is built-in
            print(f"✓ {package} (built-in)")
            continue
        
        cmd = f"{sys.executable} -m pip install --quiet {package}"
        if run_command(cmd, f"Installing {package}"):
            pass
        else:
            print(f"   Continuing anyway...")
    
    # Set environment variable for testing (if not already set)
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  GEMINI_API_KEY not set!")
        print("   For full testing, set it with:")
        print("   export GEMINI_API_KEY=your_api_key")
        print("   Or create .env file with: GEMINI_API_KEY=your_api_key")
    else:
        print(f"\n✓ GEMINI_API_KEY is set ({len(os.getenv('GEMINI_API_KEY'))} chars)")
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Set GEMINI_API_KEY: export GEMINI_API_KEY=your_key")
    print("2. Run tests:         python test_integration.py")
    print("3. Run app:           python main_app.py --enable-gemini")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
