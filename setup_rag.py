#!/usr/bin/env python3
"""
RAG Quick Setup - Install embeddings, initialize vector store, and test
Run: python setup_rag.py
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required packages for RAG"""
    logger.info("📦 Installing RAG dependencies...")
    
    packages = [
        "sentence-transformers",
        "torch",
        "faiss-cpu",
        "numpy",
        "scikit-learn",
    ]
    
    try:
        for package in packages:
            logger.info(f"  Installing {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", package]
            )
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False


def initialize_embedder():
    """Initialize and test embedder"""
    logger.info("\n🧠 Initializing embedder...")
    
    try:
        from memory.embeddings import create_embedder
        
        embedder = create_embedder("all-MiniLM-L6-v2")
        
        if not embedder or not embedder.model:
            logger.error("❌ Embedder failed to initialize")
            return False
        
        # Test embedding
        test_text = "Your phone is on the nightstand"
        embedding = embedder.embed_text(test_text)
        
        if embedding is None:
            logger.error("❌ Embedding generation failed")
            return False
        
        logger.info(f"✅ Embedder initialized")
        logger.info(f"   Model: all-MiniLM-L6-v2")
        logger.info(f"   Embedding dimension: {embedding.shape[0]}")
        logger.info(f"   Test embedding: {embedding[:5]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Embedder initialization failed: {e}")
        return False


def initialize_vector_store():
    """Initialize FAISS vector store"""
    logger.info("\n🗂️  Initializing vector store...")
    
    try:
        from memory.embeddings import create_vector_store
        
        store = create_vector_store(vector_dim=384)
        
        if not store or not store.index:
            logger.error("❌ Vector store initialization failed")
            return False
        
        logger.info(f"✅ Vector store initialized")
        logger.info(f"   Type: FAISS with L2 distance")
        logger.info(f"   Dimension: 384")
        logger.info(f"   Index path: {store.index_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Vector store initialization failed: {e}")
        return False


def test_rag_workflow():
    """Test complete RAG workflow"""
    logger.info("\n🧪 Testing RAG workflow...")
    
    try:
        import numpy as np
        from memory.storage import create_memory
        from memory.embeddings import create_embedder
        from query.engine import create_query_engine
        
        # Initialize components
        memory = create_memory()
        embedder = create_embedder()
        query_engine = create_query_engine(memory, use_semantic=True)
        
        if not embedder or not embedder.model:
            logger.error("❌ Embedder not available")
            return False
        
        # Test 1: Store object with embedding
        logger.info("  Test 1: Storing object with embedding...")
        
        object_id = memory.store_object(
            object_name="test_phone",
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.95,
            scene_description="Phone on the wooden nightstand",
            track_id=999,
            class_id=67
        )
        
        if not object_id:
            logger.warning("  ⚠️ Object storage skipped (may be duplicate)")
        else:
            logger.info(f"    ✓ Stored object ID: {object_id}")
        
        # Test 2: Generate embedding
        logger.info("  Test 2: Generating embedding...")
        
        scene_desc = "Phone on the wooden nightstand"
        embedding = embedder.embed_text(scene_desc)
        
        if embedding is None:
            logger.error("    ❌ Embedding generation failed")
            return False
        
        logger.info(f"    ✓ Generated embedding: {embedding.shape}")
        
        # Test 3: Semantic search
        logger.info("  Test 3: Performing semantic search...")
        
        query = "phone on nightstand"
        results = query_engine.semantic_search(query, k=3)
        
        logger.info(f"    ✓ Found {len(results)} results")
        
        if results:
            for i, result in enumerate(results[:1], 1):
                sim = result.get('similarity', 0)
                desc = result.get('scene_description', 'N/A')
                logger.info(f"      {i}. Similarity: {sim*100:.1f}% | {desc[:50]}...")
        
        logger.info("✅ RAG workflow test completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"❌ RAG workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def update_config():
    """Show what to add to config.py"""
    logger.info("\n📝 Configuration settings to add to config.py:")
    
    config_text = """
# ========================= EMBEDDING SETTINGS =========================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Fast model (22MB)
EMBEDDING_DIM = 384                      # Output dimension

# Semantic search settings
SEMANTIC_SIMILARITY_THRESHOLD = 0.65    # Min similarity (0-1)
SEMANTIC_SEARCH_TOP_K = 5               # Return top 5 results

# FAISS Vector Store
ENABLE_FAISS = True
    """
    
    logger.info(config_text)


def main():
    """Main setup flow"""
    logger.info("=" * 70)
    logger.info("🚀 RAG (Retrieval-Augmented Generation) Setup")
    logger.info("=" * 70)
    
    steps = [
        ("Install Dependencies", install_dependencies),
        ("Initialize Embedder", initialize_embedder),
        ("Initialize Vector Store", initialize_vector_store),
        ("Test RAG Workflow", test_rag_workflow),
    ]
    
    failed_steps = []
    
    for i, (step_name, step_func) in enumerate(steps, 1):
        logger.info(f"\n[{i}/{len(steps)}] {step_name}")
        logger.info("-" * 70)
        
        success = step_func()
        
        if not success:
            failed_steps.append(step_name)
            logger.warning(f"⚠️  {step_name} encountered issues")
    
    # Show summary
    logger.info("\n" + "=" * 70)
    logger.info("📋 SETUP SUMMARY")
    logger.info("=" * 70)
    
    if not failed_steps:
        logger.info("✅ All steps completed successfully!")
        logger.info("\n✨ RAG is now ready to use!")
        logger.info("\nQuick test:")
        logger.info("  python examples/rag_integration_example.py")
        logger.info("\nNext steps:")
        logger.info("  1. Add config settings (see above)")
        logger.info("  2. Update main_app.py to embed scene descriptions")
        logger.info("  3. Use query_engine.semantic_search() in your code")
    else:
        logger.warning(f"\n⚠️  {len(failed_steps)} step(s) need attention:")
        for step in failed_steps:
            logger.warning(f"  • {step}")
        logger.info("\nCheck the logs above for details")
    
    update_config()
    
    logger.info("\n" + "=" * 70)
    logger.info("📚 For more info, read: RAG_IMPLEMENTATION_GUIDE.md")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
