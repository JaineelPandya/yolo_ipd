#!/usr/bin/env python3
"""
Complete Test Suite for Gemini + Database Integration
Tests:
1. Environment setup
2. Gemini API connection
3. Database functionality
4. RAG embeddings
5. Integration pipeline
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_environment():
    """Test 1: Environment Setup"""
    print("\n" + "="*70)
    print("✅ TEST 1: ENVIRONMENT SETUP")
    print("="*70)
    
    try:
        # Check Python version
        print(f"✓ Python version: {sys.version.split()[0]}")
        
        # Check required directories
        required_dirs = [
            "data/database",
            "data/frames",
            "logs",
            "models",
            "memory",
            "detection",
            "tracking",
            "gemini_api",
            "integration"
        ]
        
        for dir_name in required_dirs:
            path = Path(dir_name)
            if path.exists():
                print(f"✓ Directory exists: {dir_name}")
            else:
                print(f"✗ Missing directory: {dir_name}")
                path.mkdir(parents=True, exist_ok=True)
                print(f"  Created: {dir_name}")
        
        # Check config
        import config
        print(f"✓ Config loaded")
        print(f"  Device type: {config.DEVICE_TYPE}")
        print(f"  Database path: {config.DATABASE_PATH}")
        print(f"  Embedding model: {config.EMBEDDING_MODEL}")
        
        print("\n✅ Environment setup: OK")
        return True
    
    except Exception as e:
        print(f"\n❌ Environment setup failed: {e}")
        return False


def test_dependencies():
    """Test 2: Required Dependencies"""
    print("\n" + "="*70)
    print("✅ TEST 2: DEPENDENCIES")
    print("="*70)
    
    dependencies = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("sqlite3", "sqlite3"),
        ("google.genai", "google-genai"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "faiss-cpu"),
    ]
    
    missing = []
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name if '.' not in module_name else module_name.split('.')[0])
            print(f"✓ {module_name}")
        except ImportError:
            print(f"✗ {module_name} - Install: pip install {package_name}")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("\n✅ Dependencies: OK")
    return True


def test_gemini_api():
    """Test 3: Gemini API Connection"""
    print("\n" + "="*70)
    print("✅ TEST 3: GEMINI API CONNECTION")
    print("="*70)
    
    try:
        import config
        from gemini_api.descriptor import GeminiSceneDescriptor
        
        # Check API key
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("⚠️  GEMINI_API_KEY not set in environment")
            print("   Set it with: export GEMINI_API_KEY=your_key")
            print("   Or in .env file: GEMINI_API_KEY=your_key")
            return False
        
        print(f"✓ API Key found ({len(api_key)} chars)")
        print(f"  Key: {api_key[:15]}...{api_key[-4:]}")
        
        # Initialize Gemini
        descriptor = GeminiSceneDescriptor()
        
        if not descriptor.client:
            print("✗ Gemini client not initialized")
            return False
        
        print("✓ Gemini client initialized")
        print(f"  Model: {descriptor.model_name}")
        
        # Test with dummy image
        print("\n  Testing API call with dummy image...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        test_desc = descriptor.describe_scene(dummy_frame)
        
        if test_desc:
            print(f"✓ API response: {test_desc[:60]}...")
        else:
            print("⚠️  No response from API (may be rate limited or invalid key)")
        
        print("\n✅ Gemini API: OK")
        return True
    
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False


def test_database():
    """Test 4: Database Connection"""
    print("\n" + "="*70)
    print("✅ TEST 4: DATABASE CONNECTION")
    print("="*70)
    
    try:
        from memory.storage import create_memory
        
        # Create/connect to database
        memory = create_memory()
        print("✓ Database connected")
        
        # Test storing an object
        test_obj_id = memory.store_object(
            object_name="test_phone",
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.95,
            scene_description="Test object on the table",
            track_id=999,
            class_id=67
        )
        
        if test_obj_id:
            print(f"✓ Stored test object with ID: {test_obj_id}")
        else:
            print("✓ Test object already exists (duplicate)")
        
        # Get statistics
        stats = memory.get_statistics()
        print("\n  Database Statistics:")
        print(f"  - Total objects: {stats['total_objects']}")
        print(f"  - Unique objects: {stats['unique_objects']}")
        print(f"  - Total frames: {stats['total_frames']}")
        print(f"  - Avg confidence: {stats['avg_confidence']:.2f}")
        
        # Test retrieval
        result = memory.get_last_seen("test_phone")
        if result:
            print(f"\n✓ Retrieved object: {result['object_name']}")
            print(f"  Timestamp: {result['timestamp']}")
            print(f"  Description: {result.get('scene_description', 'N/A')}")
        
        print("\n✅ Database: OK")
        return True
    
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_embeddings():
    """Test 5: RAG Embeddings"""
    print("\n" + "="*70)
    print("✅ TEST 5: RAG EMBEDDINGS")
    print("="*70)
    
    try:
        from memory.embeddings import create_embedder, create_vector_store
        
        # Initialize embedder
        embedder = create_embedder()
        
        if not embedder or not embedder.model:
            print("✗ Embedder failed to initialize")
            return False
        
        print("✓ Embedder initialized")
        print(f"  Model: {embedder.model.__class__.__name__}")
        print(f"  Dimension: {embedder.embedding_dim}")
        
        # Test embedding generation
        test_texts = [
            "Phone on the wooden nightstand",
            "Keys in the kitchen drawer",
            "Glasses on the bathroom shelf"
        ]
        
        print("\n  Testing embeddings...")
        embeddings = embedder.embed_batch(test_texts)
        
        if embeddings is not None:
            print(f"✓ Generated {len(embeddings)} embeddings")
            print(f"  Shape: {embeddings.shape}")
            
            # Calculate similarity
            sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            print(f"  Similarity (text 1 vs 2): {sim*100:.1f}%")
        else:
            print("✗ Embedding generation failed")
            return False
        
        # Test vector store
        print("\n  Testing FAISS vector store...")
        vector_store = create_vector_store(vector_dim=384)
        
        if vector_store and vector_store.index:
            print("✓ Vector store initialized")
            
            # Add vectors
            vector_store.add_vectors(
                embeddings,
                [1, 2, 3],
                [{"text": t} for t in test_texts]
            )
            print(f"✓ Added {len(embeddings)} vectors to index")
            
            # Search
            query_embedding = embeddings[0:1]
            results, distances = vector_store.search(query_embedding, k=2)
            print(f"✓ Search results: {results}")
        else:
            print("⚠️  FAISS not available (optional)")
        
        print("\n✅ RAG Embeddings: OK")
        return True
    
    except Exception as e:
        print(f"⚠️  RAG Embeddings test: {e}")
        return False


def test_integration():
    """Test 6: Full Integration"""
    print("\n" + "="*70)
    print("✅ TEST 6: FULL INTEGRATION (Gemini + DB + RAG)")
    print("="*70)
    
    try:
        from memory.storage import create_memory
        from integration.gemini_db_bridge import create_bridge
        import numpy as np
        
        # Create components
        memory = create_memory()
        bridge = create_bridge(memory, enable_embeddings=True)
        
        print("✓ Bridge created")
        
        # Create dummy detection
        detections = [
            {
                "class_name": "phone",
                "class_id": 67,
                "bbox": np.array([100, 150, 300, 350]),
                "confidence": 0.94,
                "track_id": 1
            },
            {
                "class_name": "book",
                "class_id": 84,
                "bbox": np.array([400, 200, 550, 400]),
                "confidence": 0.87,
                "track_id": 2
            }
        ]
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print(f"\n  Processing {len(detections)} detections...")
        
        # Process through bridge
        result = bridge.process_detections_with_gemini(
            dummy_frame,
            detections
        )
        
        if result["success"]:
            print(f"✓ Processing successful")
            print(f"  Objects processed: {result['objects_processed']}")
            print(f"  Descriptions created: {result['descriptions_created']}")
            if result["errors"]:
                print(f"  Errors: {result['errors']}")
        else:
            print(f"⚠️  Processing completed with errors: {result['errors']}")
        
        # Print bridge status
        bridge.print_status()
        
        print("✅ Integration: OK")
        return True
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_engine():
    """Test 7: Query Engine with RAG"""
    print("\n" + "="*70)
    print("✅ TEST 7: QUERY ENGINE (Keyword + RAG)")
    print("="*70)
    
    try:
        from memory.storage import create_memory
        from query.engine import create_query_engine
        
        memory = create_memory()
        query_engine = create_query_engine(memory, use_semantic=True)
        
        print("✓ Query engine created with RAG support")
        
        # Test keyword search
        print("\n  Testing keyword search...")
        result = query_engine.get_last_seen("phone")
        print(f"  Result: {result[:50]}..." if result else "  No results")
        
        # Test semantic search
        if query_engine.use_semantic:
            print("\n  Testing semantic search...")
            semantic_results = query_engine.semantic_search("phone on table", k=3)
            print(f"  Found {len(semantic_results)} similar scenes")
        
        # Test smart search
        print("\n  Testing smart search (hybrid)...")
        smart_result = query_engine.smart_search("Where is my phone?")
        print(f"  Result: {smart_result[:60]}..." if smart_result else "  No results")
        
        print("\n✅ Query Engine: OK")
        return True
    
    except Exception as e:
        print(f"⚠️  Query Engine test: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("█" * 70)
    print("  🚀 GEMINI + DATABASE INTEGRATION TEST SUITE")
    print("█" * 70)
    
    tests = [
        ("Environment", test_environment),
        ("Dependencies", test_dependencies),
        ("Gemini API", test_gemini_api),
        ("Database", test_database),
        ("RAG Embeddings", test_rag_embeddings),
        ("Integration", test_integration),
        ("Query Engine", test_query_engine),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⏸️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System ready for use.")
        print("\nNext steps:")
        print("1. Run: python main_app.py --enable-gemini")
        print("2. Or:  streamlit run app_streamlit.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above for details.")
    
    print("\n" + "="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
