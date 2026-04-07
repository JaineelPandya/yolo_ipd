#!/usr/bin/env python3
"""
Quick test to check if Gemini API is working
"""

print("\n" + "="*70)
print("🔍 Quick Gemini Diagnostic")
print("="*70)

# Step 1: Load .env
print("\n1️⃣ Loading .env file...")
from dotenv import load_dotenv
import os
load_dotenv(override=True)

api_key_env = os.getenv("GEMINI_API_KEY", "")
print(f"   From .env: {api_key_env[:20] if api_key_env else 'NOT FOUND'}...")
print(f"   Length: {len(api_key_env)} chars")

# Step 2: Load config
print("\n2️⃣ Loading config...")
import config
print(f"   config.GEMINI_API_KEY: {config.GEMINI_API_KEY[:20] if config.GEMINI_API_KEY else 'NOT FOUND'}...")
print(f"   Length: {len(config.GEMINI_API_KEY)} chars")

# Step 3: Create descriptor
print("\n3️⃣ Creating GeminiSceneDescriptor...")
from gemini_api.descriptor import GeminiSceneDescriptor

# Try with explicit API key
api_key = os.getenv("GEMINI_API_KEY") or config.GEMINI_API_KEY
print(f"   Using API key: {api_key[:20]}...{api_key[-4:] if api_key else 'NONE'}")

descriptor = GeminiSceneDescriptor(api_key=api_key)

print(f"\n4️⃣ Result:")
print(f"   Client ready: {descriptor.client is not None}")
if descriptor.client:
    print(f"   ✅ SUCCESS - Gemini is initialized and ready!")
else:
    print(f"   ❌ FAILED - Gemini client is NOT initialized")
    print(f"   Check API key: {len(api_key)} chars")

print("\n" + "="*70)
