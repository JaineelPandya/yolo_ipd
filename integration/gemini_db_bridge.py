"""
Gemini + Database Integration Module
Connects Gemini API with SQLite database for seamless scene description storage and RAG retrieval
"""

import logging
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import config
from memory.storage import ObjectMemory
from memory.embeddings import create_embedder
from gemini_api.descriptor import GeminiSceneDescriptor

logger = logging.getLogger(__name__)


class GeminiDatabaseBridge:
    """
    Bridges Gemini API with Database and Embeddings
    
    Workflow:
    1. Detect objects in frame → Get detections
    2. Generate scene description → Gemini API
    3. Store in database → SQLite
    4. Generate embeddings → Semantic search
    5. Enable RAG retrieval → Query engine
    """
    
    def __init__(self, 
                 memory: ObjectMemory,
                 gemini_api_key: str = None,
                 enable_embeddings: bool = True):
        """
        Initialize bridge
        
        Args:
            memory: ObjectMemory instance (database)
            gemini_api_key: Gemini API key (optional, uses config if not provided)
            enable_embeddings: Enable semantic embeddings for RAG
        """
        self.memory = memory
        self.enable_embeddings = enable_embeddings
        
        # Initialize Gemini descriptor
        self.scene_descriptor = GeminiSceneDescriptor(api_key=gemini_api_key)
        
        # Initialize embedder for RAG
        self.embedder = None
        if enable_embeddings:
            self.embedder = create_embedder(config.EMBEDDING_MODEL)
            if self.embedder and self.embedder.model:
                logger.info(f"✓ Embedder initialized for RAG")
            else:
                logger.warning("✗ Embedder not available - RAG disabled")
                self.enable_embeddings = False
        
        self.stats = {
            "objects_stored": 0,
            "descriptions_generated": 0,
            "embeddings_created": 0,
            "errors": 0
        }
    
    def process_detections_with_gemini(self, 
                                       frame: np.ndarray,
                                       detections: List[Dict],
                                       frame_id: int = None) -> Dict:
        """
        Process detections: Store scene description + embeddings in DB
        
        Args:
            frame: Image frame (numpy array)
            detections: List of detected objects
            frame_id: Optional frame ID (auto-generated if not provided)
        
        Returns:
            Dict with processing results
        """
        result = {
            "success": True,
            "objects_processed": 0,
            "descriptions_created": 0,
            "errors": []
        }
        
        if not detections:
            logger.debug("No detections to process")
            return result
        
        try:
            # Step 1: Generate scene description via Gemini
            scene_description = None
            if self.scene_descriptor and self.scene_descriptor.client:
                logger.info(f"🔄 Generating scene description for {len(detections)} objects...")
                scene_description = self.scene_descriptor.describe_scene(frame, detections)
                
                if scene_description:
                    logger.info(f"✅ Scene description: {scene_description[:80]}...")
                    result["descriptions_created"] += 1
                    self.stats["descriptions_generated"] += 1
            
            # Step 2: Store frame with scene description
            frame_db_id = None
            if scene_description or True:  # Store frame even without description
                try:
                    frame_db_id = self.memory.store_frame(
                        image_path=None,  # Can add path if saved
                        compressed_image_path=None,
                        scene_description=scene_description,
                        num_objects=len(detections),
                        frame_hash=None
                    )
                    
                    if frame_db_id:
                        logger.debug(f"Stored frame with ID: {frame_db_id}")
                        
                        # Generate and store embedding for the frame
                        if self.enable_embeddings and scene_description:
                            self._store_frame_embedding(
                                frame_db_id, 
                                scene_description
                            )
                except Exception as e:
                    logger.error(f"Error storing frame: {e}")
                    result["errors"].append(f"Frame storage: {e}")
            
            # Step 3: Store individual objects with descriptions
            for det in detections:
                try:
                    obj_id = self.memory.store_object(
                        object_name=det['class_name'],
                        bbox=det['bbox'],
                        confidence=det['confidence'],
                        scene_description=scene_description,
                        image_path=None,
                        track_id=det.get('track_id'),
                        class_id=det.get('class_id'),
                        notes=f"Detected with confidence {det['confidence']:.2f}"
                    )
                    
                    if obj_id:
                        result["objects_processed"] += 1
                        self.stats["objects_stored"] += 1
                        
                        logger.info(f"✓ Stored object: {det['class_name']} "
                                  f"(ID: {obj_id}, conf: {det['confidence']:.2f})")
                
                except Exception as e:
                    logger.error(f"Error storing object {det['class_name']}: {e}")
                    result["errors"].append(f"Object storage ({det['class_name']}): {e}")
        
        except Exception as e:
            logger.error(f"Error in process_detections_with_gemini: {e}")
            result["success"] = False
            result["errors"].append(str(e))
        
        return result
    
    def _store_frame_embedding(self, frame_id: int, text: str) -> bool:
        """
        Generate and store embedding for frame description
        
        Args:
            frame_id: Frame ID in database
            text: Text to embed (scene description)
        
        Returns:
            True if successful
        """
        if not self.embedder or not self.embedder.model:
            return False
        
        try:
            embedding = self.embedder.embed_text(text)
            if embedding is not None:
                self.memory.store_embedding(frame_id, embedding)
                self.stats["embeddings_created"] += 1
                logger.debug(f"Stored embedding for frame {frame_id}")
                return True
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
        
        return False
    
    def process_object_with_gemini(self,
                                    frame: np.ndarray,
                                    object_name: str,
                                    bbox: np.ndarray,
                                    class_id: int = None,
                                    track_id: int = None,
                                    confidence: float = None) -> Optional[int]:
        """
        Process single object: Generate description + Store in DB
        
        Args:
            frame: Image frame
            object_name: Name of object
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: COCO class ID (optional)
            track_id: Tracking ID (optional)
            confidence: Detection confidence (optional)
        
        Returns:
            Object ID if stored successfully, None otherwise
        """
        try:
            # Generate per-object scene description
            scene_desc = None
            if self.scene_descriptor and self.scene_descriptor.client:
                scene_desc = self.scene_descriptor.describe_scene_for_object(
                    frame, object_name, bbox
                )
                
                if scene_desc:
                    logger.info(f"✅ Gemini for {object_name}: {scene_desc[:60]}...")
                    self.stats["descriptions_generated"] += 1
            
            # Store object
            obj_id = self.memory.store_object(
                object_name=object_name,
                bbox=bbox,
                confidence=confidence or 0.0,
                scene_description=scene_desc,
                track_id=track_id,
                class_id=class_id
            )
            
            if obj_id:
                self.stats["objects_stored"] += 1
                logger.info(f"✓ Stored {object_name} with ID: {obj_id}")
                
                # Generate embedding for description
                if self.enable_embeddings and scene_desc:
                    self._store_frame_embedding(obj_id, scene_desc)
            
            return obj_id
        
        except Exception as e:
            logger.error(f"Error processing object: {e}")
            self.stats["errors"] += 1
            return None
    
    def get_statistics(self) -> Dict:
        """Get bridge statistics"""
        return {
            "gemini": {
                "descriptions_generated": self.stats["descriptions_generated"],
                "api_status": "✓ Ready" if self.scene_descriptor.client else "✗ Not initialized"
            },
            "database": self.memory.get_statistics(),
            "embeddings": {
                "total_created": self.stats["embeddings_created"],
                "rag_enabled": self.enable_embeddings,
                "embedder_status": "✓ Ready" if self.embedder and self.embedder.model else "✗ Not available"
            },
            "objects": {
                "total_stored": self.stats["objects_stored"],
                "errors": self.stats["errors"]
            }
        }
    
    def print_status(self):
        """Print bridge status"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("🔗 GEMINI + DATABASE BRIDGE STATUS")
        print("="*70)
        
        print("\n📍 Gemini API:")
        print(f"  Status: {stats['gemini']['api_status']}")
        print(f"  Descriptions: {stats['gemini']['descriptions_generated']}")
        
        print("\n🗄️  Database:")
        db_stats = stats['database']
        print(f"  Total Objects: {db_stats['total_objects']}")
        print(f"  Unique Objects: {db_stats['unique_objects']}")
        print(f"  Total Frames: {db_stats['total_frames']}")
        print(f"  Avg Confidence: {db_stats['avg_confidence']:.2f}")
        
        print("\n🧠 Embeddings & RAG:")
        print(f"  Status: {stats['embeddings']['rag_enabled'] and '✓ Enabled' or '✗ Disabled'}")
        print(f"  Embedder: {stats['embeddings']['embedder_status']}")
        print(f"  Total Embeddings: {stats['embeddings']['total_created']}")
        
        print("\n⚠️  Errors: {}".format(stats['objects']['errors']))
        print("="*70 + "\n")


def create_bridge(memory: ObjectMemory, 
                  enable_embeddings: bool = True) -> GeminiDatabaseBridge:
    """Factory function to create bridge"""
    return GeminiDatabaseBridge(
        memory=memory,
        enable_embeddings=enable_embeddings
    )
