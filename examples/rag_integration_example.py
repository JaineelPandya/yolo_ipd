"""
Example: How to integrate RAG (Retrieval-Augmented Generation) into your YOLO project
This shows how to:
1. Generate embeddings for scene descriptions
2. Store them in the database
3. Perform semantic searches
"""

import numpy as np
from memory.storage import create_memory
from memory.embeddings import create_embedder
from query.engine import create_query_engine

def example_rag_workflow():
    """Complete RAG workflow example"""
    
    print("=" * 70)
    print("RAG INTEGRATION EXAMPLE - Retrieval-Augmented Generation")
    print("=" * 70)
    
    # 1. Initialize components
    print("\n1️⃣ INITIALIZING COMPONENTS...")
    memory = create_memory()
    embedder = create_embedder()
    query_engine = create_query_engine(memory, use_semantic=True)
    
    if not embedder or not embedder.model:
        print("❌ Embedder failed to initialize")
        return
    
    print(f"✓ Embedder dimension: {embedder.embedding_dim}")
    print(f"✓ Memory system ready")
    
    # 2. Example: Store object with embedding
    print("\n2️⃣ STORING OBJECT WITH EMBEDDING...")
    
    # Simulate storing a detection
    object_id = memory.store_object(
        object_name="phone",
        bbox=np.array([100, 200, 300, 400]),
        confidence=0.95,
        scene_description="Your phone is on the wooden nightstand next to the lamp",
        image_path="/path/to/frame.jpg",
        track_id=1,
        class_id=67  # COCO class for cell phone
    )
    
    if object_id:
        print(f"✓ Stored object with ID: {object_id}")
        
        # Get the frame ID and store its embedding
        # In real usage, you'd get frame_id from store_frame()
        frame_id = 1  # Example
        
        # Generate embedding from scene description
        scene_desc = "Your phone is on the wooden nightstand next to the lamp"
        embedding = embedder.embed_text(scene_desc)
        
        if embedding is not None:
            memory.store_embedding(frame_id, embedding)
            print(f"✓ Generated and stored embedding ({embedding.shape})")
    
    # 3. Example: Semantic search
    print("\n3️⃣ SEMANTIC SEARCH (RAG)...")
    
    query = "phone on nightstand"
    print(f"\nQuery: '{query}'")
    
    results = query_engine.semantic_search(query, k=5)
    
    if results:
        print(f"\n✓ Found {len(results)} similar scenes:")
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0)
            description = result.get('scene_description', 'N/A')
            timestamp = result.get('timestamp', 'N/A')
            print(f"\n  {i}. Similarity: {similarity*100:.1f}%")
            print(f"     Description: {description}")
            print(f"     Time: {timestamp}")
    else:
        print("✗ No similar scenes found")
    
    # 4. Example: Smart search (hybrid)
    print("\n4️⃣ SMART SEARCH (Keyword + Semantic)...")
    
    query = "Where is my phone on a shelf?"
    print(f"\nQuery: '{query}'")
    
    response = query_engine.smart_search(query)
    print(f"Response:\n{response}")
    
    # 5. Example: Location-based semantic search
    print("\n5️⃣ LOCATION-BASED SEMANTIC SEARCH...")
    
    location = "on the wooden nightstand"
    print(f"\nLocation: '{location}'")
    
    location_results = query_engine.semantic_location_search(location)
    for result in location_results:
        print(f"  • {result}")
    
    print("\n" + "=" * 70)
    print("RAG EXAMPLE COMPLETE")
    print("=" * 70)


def batch_embedding_example():
    """Example: Batch embed multiple scene descriptions"""
    
    print("\n\nBATCH EMBEDDING EXAMPLE")
    print("=" * 70)
    
    embedder = create_embedder()
    
    if not embedder or not embedder.model:
        print("❌ Embedder not available")
        return
    
    # Multiple scene descriptions
    scenes = [
        "Your phone is on the wooden nightstand next to the lamp",
        "Your keys are in the kitchen drawer",
        "Your glasses are on the bathroom shelf",
        "Your wallet is on the dining table",
    ]
    
    print(f"Embedding {len(scenes)} scenes...")
    embeddings = embedder.embed_batch(scenes)
    
    if embeddings is not None:
        print(f"✓ Generated embeddings: {embeddings.shape}")
        
        # Calculate similarity between first scene and others
        print("\nSimilarity with first scene:")
        for i in range(1, len(embeddings)):
            sim = memory.ObjectMemory._cosine_similarity(
                embeddings[0], 
                embeddings[i]
            )
            print(f"  {scenes[0][:40]}...")
            print(f"  ↔ {scenes[i][:40]}... = {sim*100:.1f}% similar")


def config_required():
    """What configuration is needed?"""
    
    print("\n\nREQUIRED CONFIGURATION")
    print("=" * 70)
    
    print("""
Add these settings to config.py:

    # Embedding settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 22MB
    # EMBEDDING_MODEL = "all-mpnet-base-v2"  # Better quality, 420MB
    
    EMBEDDING_DIM = 384  # Output dimension of embeddings
    
    # Semantic search settings
    SEMANTIC_SIMILARITY_THRESHOLD = 0.6  # Min similarity for results
    SEMANTIC_SEARCH_TOP_K = 5  # Return top 5 similar scenes
    
    # FAISSVector store
    ENABLE_FAISS = True  # Use FAISS for faster vector search
    FAISS_INDEX_PATH = str(DATABASE_DIR / "faiss_index.bin")

Then install:
    pip install sentence-transformers torch faiss-cpu
    
Or for GPU:
    pip install sentence-transformers torch faiss-gpu
    """)


if __name__ == "__main__":
    try:
        example_rag_workflow()
        batch_embedding_example()
        config_required()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
