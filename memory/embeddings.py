"""
Semantic embedding module for RAG-based retrieval memory
Generates embeddings for scene descriptions and performs semantic search
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import config

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticEmbedder:
    """Generate semantic embeddings for scene descriptions and queries"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic embedder
        
        Args:
            model_name: HuggingFace model for embeddings
                - "all-MiniLM-L6-v2" (22MB, fast, CPU-friendly)
                - "all-mpnet-base-v2" (420MB, better quality)
        """
        if not EMBEDDINGS_AVAILABLE:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            self.model = None
            return
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"✗ Failed to load embedding model: {e}")
            self.model = None
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text
        
        Args:
            text: Text to embed
        
        Returns:
            Numpy array of shape (embedding_dim,) or None if failed
        """
        if not self.model:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for multiple texts (more efficient)
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Numpy array of shape (len(texts), embedding_dim) or None
        """
        if not self.model or not texts:
            return None
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return None


class FAISSVectorStore:
    """FAISS-based vector store for fast semantic similarity search"""
    
    def __init__(self, vector_dim: int = 384, index_path: str = None):
        """
        Initialize FAISS vector store
        
        Args:
            vector_dim: Dimension of vectors (match embedding model dim)
            index_path: Path to save/load FAISS index
        """
        if not FAISS_AVAILABLE:
            logger.error("faiss not installed. Run: pip install faiss-cpu")
            self.index = None
            return
        
        self.vector_dim = vector_dim
        self.index_path = index_path or str(Path(config.DATABASE_DIR) / "faiss_index.bin")
        self.metadata = {}  # Store ID → metadata mapping
        self.metadata_path = str(Path(config.DATABASE_DIR) / "faiss_metadata.json")
        
        # Try to load existing index
        self._load_index()
        
        if self.index is None:
            try:
                # Create new index
                import faiss
                self.index = faiss.IndexFlatL2(vector_dim)  # L2 distance
                logger.info(f"✓ Created new FAISS index (dim={vector_dim})")
            except Exception as e:
                logger.error(f"✗ FAISS initialization failed: {e}")
                self.index = None
    
    def add_vectors(self, vectors: np.ndarray, ids: List[int], metadata_list: List[Dict] = None):
        """
        Add vectors to index
        
        Args:
            vectors: Array of shape (n_samples, vector_dim)
            ids: List of IDs corresponding to vectors
            metadata_list: List of metadata dicts for each vector
        """
        if self.index is None or vectors is None:
            return
        
        try:
            # Ensure vectors are float32
            vectors = np.asarray(vectors, dtype=np.float32)
            
            # Add to FAISS index
            self.index.add(vectors)
            
            # Store metadata
            if metadata_list:
                for vid, meta in zip(ids, metadata_list):
                    self.metadata[int(vid)] = meta
            else:
                for vid in ids:
                    self.metadata[int(vid)] = {"id": int(vid)}
            
            logger.debug(f"Added {len(vectors)} vectors to index")
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_vector: Query embedding (1, vector_dim)
            k: Number of results to return
        
        Returns:
            Tuple of (ids, distances)
        """
        if self.index is None or query_vector is None:
            return [], []
        
        try:
            query_vector = np.asarray(query_vector, dtype=np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            distances, indices = self.index.search(query_vector, k)
            
            # Convert to lists and filter out -1 indices
            ids = [int(idx) for idx in indices[0] if idx != -1]
            dists = [float(d) for i, d in enumerate(distances[0]) if indices[0][i] != -1]
            
            return ids, dists
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [], []
    
    def save(self):
        """Save index and metadata to disk"""
        if self.index is None:
            return
        
        try:
            import faiss
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.info(f"✓ Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _load_index(self):
        """Load existing index from disk"""
        if not Path(self.index_path).exists():
            return
        
        try:
            import faiss
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"✓ Loaded FAISS index from {self.index_path}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = None


def create_embedder(model_name: str = "all-MiniLM-L6-v2") -> Optional[SemanticEmbedder]:
    """Factory function to create embedder"""
    return SemanticEmbedder(model_name)


def create_vector_store(vector_dim: int = 384) -> Optional[FAISSVectorStore]:
    """Factory function to create vector store"""
    return FAISSVectorStore(vector_dim)
