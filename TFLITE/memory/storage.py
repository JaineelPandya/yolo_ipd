"""
Memory module - SQLite database for storing object tracking history
"""

import sqlite3
import cv2
import numpy as np
import logging
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import config

logger = logging.getLogger(__name__)

class ObjectMemory:
    """SQLite-based object memory system"""
    
    def __init__(self, db_path: str = None):
        """Initialize database"""
        if db_path is None:
            db_path = config.DATABASE_PATH
        
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._connect()
        self._init_database()
    
    def _connect(self):
        """Connect to database"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"✓ Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"✗ Database connection error: {e}")
            raise
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            cursor = self.conn.cursor()
            
            # Objects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_name TEXT NOT NULL,
                    class_id INTEGER,
                    track_id INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    bbox TEXT,  -- JSON string: [x1, y1, x2, y2]
                    confidence REAL,
                    scene_description TEXT,
                    image_path TEXT,
                    compressed_image_path TEXT,
                    latitude REAL,  -- For future location features
                    longitude REAL,
                    notes TEXT,
                    UNIQUE(track_id, object_name, timestamp)
                )
            ''')
            
            # Frames table (for associating objects with frames)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT UNIQUE,
                    compressed_image_path TEXT,
                    embedding_vector BLOB,  -- For similarity search
                    scene_description TEXT,
                    num_objects INTEGER,
                    frame_hash TEXT UNIQUE  -- To detect duplicates
                )
            ''')
            
            # Object-Frame association
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS object_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id INTEGER,
                    frame_id INTEGER,
                    FOREIGN KEY(object_id) REFERENCES objects(id),
                    FOREIGN KEY(frame_id) REFERENCES frames(id)
                )
            ''')
            
            # Create indices for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_object_name ON objects(object_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON objects(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_id ON objects(track_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frame_timestamp ON frames(timestamp)')
            
            self.conn.commit()
            logger.info("✓ Database tables initialized")
        except Exception as e:
            logger.error(f"✗ Database initialization error: {e}")
            raise
    
    def store_object(self, 
                    object_name: str,
                    bbox: np.ndarray,
                    confidence: float,
                    scene_description: str = None,
                    image_path: str = None,
                    track_id: int = None,
                    class_id: int = None,
                    notes: str = None) -> int:
        """
        Store object detection in database
        
        Args:
            object_name: Name of detected object
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            scene_description: Generated scene description from Gemini
            image_path: Path to full frame
            track_id: ID from tracker
            class_id: COCO class ID
            notes: Additional notes
        
        Returns:
            Object ID in database
        """
        try:
            cursor = self.conn.cursor()
            
            bbox_json = json.dumps(bbox.tolist()) if isinstance(bbox, np.ndarray) else json.dumps(bbox)
            
            cursor.execute('''
                INSERT INTO objects 
                (object_name, class_id, track_id, bbox, confidence, scene_description, 
                 image_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                object_name, class_id, track_id, bbox_json, confidence,
                scene_description, image_path, notes
            ))
            
            self.conn.commit()
            object_id = cursor.lastrowid
            
            logger.debug(f"Stored object: {object_name} (ID: {object_id})")
            return object_id
        
        except sqlite3.IntegrityError as e:
            logger.debug(f"Duplicate entry (already exists): {e}")
            return None
        except Exception as e:
            logger.error(f"Error storing object: {e}")
            return None
    
    def store_frame(self, 
                   image_path: str,
                   compressed_image_path: str = None,
                   scene_description: str = None,
                   num_objects: int = 0,
                   frame_hash: str = None) -> int:
        """
        Store frame information
        
        Args:
            image_path: Path to full resolution frame
            compressed_image_path: Path to compressed frame
            scene_description: Scene description
            num_objects: Number of objects detected
            frame_hash: Hash to detect duplicates
        
        Returns:
            Frame ID in database
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO frames 
                (image_path, compressed_image_path, scene_description, num_objects, frame_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_path, compressed_image_path, scene_description, num_objects, frame_hash))
            
            self.conn.commit()
            frame_id = cursor.lastrowid
            logger.debug(f"Stored frame (ID: {frame_id})")
            return frame_id
        
        except sqlite3.IntegrityError:
            logger.debug(f"Frame already stored")
            return None
        except Exception as e:
            logger.error(f"Error storing frame: {e}")
            return None
    
    def associate_object_frame(self, object_id: int, frame_id: int):
        """Associate object with frame"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO object_frames (object_id, frame_id)
                VALUES (?, ?)
            ''', (object_id, frame_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error associating object-frame: {e}")
    
    def get_last_seen(self, object_name: str, time_range_minutes: int = None) -> Optional[Dict]:
        """
        Get last sighting of an object
        
        Args:
            object_name: Name of object to find
            time_range_minutes: Only search last N minutes (None = unlimited)
        
        Returns:
            Dictionary with object information or None
        """
        try:
            cursor = self.conn.cursor()
            
            query = '''
                SELECT * FROM objects 
                WHERE LOWER(object_name) = LOWER(?)
                ORDER BY timestamp DESC
                LIMIT 1
            '''
            
            if time_range_minutes:
                query = f'''
                    SELECT * FROM objects 
                    WHERE LOWER(object_name) = LOWER(?)
                    AND timestamp > datetime('now', '-{time_range_minutes} minutes')
                    ORDER BY timestamp DESC
                    LIMIT 1
                '''
            
            cursor.execute(query, (object_name,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
        
        except Exception as e:
            logger.error(f"Error querying object: {e}")
            return None
    
    def get_object_history(self, object_name: str, limit: int = 10) -> List[Dict]:
        """
        Get history of object sightings
        
        Args:
            object_name: Name of object
            limit: Maximum number of records to return
        
        Returns:
            List of sightings
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT * FROM objects 
                WHERE LOWER(object_name) = LOWER(?)
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (object_name, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []
    
    def update_object_description(self, object_name: str, timestamp: str, scene_description: str) -> bool:
        """
        Update scene description for an object (on-the-fly generation)
        
        Args:
            object_name: Name of object
            timestamp: Timestamp of record (DATETIME format)
            scene_description: New description from Gemini
        
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                UPDATE objects 
                SET scene_description = ?
                WHERE LOWER(object_name) = LOWER(?) 
                AND timestamp = ?
            ''', (scene_description, object_name, timestamp))
            
            self.conn.commit()
            logger.debug(f"Updated description for {object_name} at {timestamp}")
            return cursor.rowcount > 0
        
        except Exception as e:
            logger.error(f"Error updating description: {e}")
            return False
    
    def search_objects_by_location(self, location_keywords: List[str]) -> List[Dict]:
        """
        Search objects by location mentioned in scene description
        
        Args:
            location_keywords: Keywords like ['cupboard', 'table', 'shelf']
        
        Returns:
            List of objects found at those locations
        """
        try:
            cursor = self.conn.cursor()
            
            where_clauses = ' OR '.join(
                [f"scene_description LIKE ?" for _ in location_keywords]
            )
            params = [f"%{kw}%" for kw in location_keywords]
            
            cursor.execute(f'''
                SELECT * FROM objects 
                WHERE {where_clauses}
                ORDER BY timestamp DESC
            ''', params)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error searching by location: {e}")
            return []
    
    def get_all_objects_today(self) -> List[Dict]:
        """Get all objects detected today"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT DISTINCT object_name, COUNT(*) as count, 
                       MAX(timestamp) as last_seen
                FROM objects 
                WHERE DATE(timestamp) = DATE('now')
                GROUP BY object_name
                ORDER BY last_seen DESC
            ''')
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error getting daily objects: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove data older than specified days"""
        try:
            cursor = self.conn.cursor()
            
            # Delete old frames and associated objects
            cursor.execute(f'''
                DELETE FROM object_frames 
                WHERE frame_id IN (
                    SELECT id FROM frames 
                    WHERE timestamp < datetime('now', '-{days_to_keep} days')
                )
            ''')
            
            cursor.execute(f'''
                DELETE FROM frames 
                WHERE timestamp < datetime('now', '-{days_to_keep} days')
            ''')
            
            cursor.execute(f'''
                DELETE FROM objects 
                WHERE timestamp < datetime('now', '-{days_to_keep} days')
            ''')
            
            self.conn.commit()
            deleted_count = cursor.rowcount
            logger.info(f"✓ Cleaned up data: deleted {deleted_count} old records")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def clear_records_without_descriptions(self) -> int:
        """Delete all records that don't have scene descriptions (old data)"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                DELETE FROM objects 
                WHERE scene_description IS NULL OR scene_description = ''
            ''')
            
            self.conn.commit()
            deleted_count = cursor.rowcount
            logger.info(f"✓ Deleted {deleted_count} records without descriptions")
            return deleted_count
        except Exception as e:
            logger.error(f"Error clearing records: {e}")
            return 0
            logger.info(f"Cleaned up {deleted_count} old records")
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM objects')
            total_objects = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT COUNT(DISTINCT object_name) FROM objects')
            unique_objects = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT COUNT(*) FROM frames')
            total_frames = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT AVG(confidence) FROM objects')
            avg_confidence = cursor.fetchone()[0] or 0
            
            return {
                "total_objects": total_objects,
                "unique_objects": unique_objects,
                "total_frames": total_frames,
                "avg_confidence": float(avg_confidence) if avg_confidence else 0.0
            }
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"total_objects": 0, "unique_objects": 0, "total_frames": 0, "avg_confidence": 0.0}
    
    # ==================== EMBEDDING & SEMANTIC RETRIEVAL ====================
    
    def store_embedding(self, frame_id: int, embedding: np.ndarray) -> bool:
        """
        Store embedding vector for a frame (for semantic search)
        
        Args:
            frame_id: Frame ID from frames table
            embedding: Embedding vector (numpy array)
        
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            # Convert embedding to binary blob
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            cursor.execute('''
                UPDATE frames 
                SET embedding_vector = ?
                WHERE id = ?
            ''', (embedding_bytes, frame_id))
            
            self.conn.commit()
            logger.debug(f"Stored embedding for frame {frame_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False
    
    def get_frame_embedding(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Retrieve embedding vector for a frame
        
        Args:
            frame_id: Frame ID
        
        Returns:
            Numpy array or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT embedding_vector FROM frames WHERE id = ?', (frame_id,))
            row = cursor.fetchone()
            
            if row and row[0]:
                embedding = np.frombuffer(row[0], dtype=np.float32)
                return embedding
            
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving embedding: {e}")
            return None
    
    def semantic_search(self, query_embedding: np.ndarray, k: int = 5, 
                       time_range_minutes: int = None) -> List[Dict]:
        """
        Find similar frames using semantic embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            time_range_minutes: Optional time filter
        
        Returns:
            List of similar frame records
        """
        try:
            cursor = self.conn.cursor()
            
            # Get all embeddings with timestamps
            time_filter = ""
            if time_range_minutes:
                time_filter = f"AND timestamp > datetime('now', '-{time_range_minutes} minutes')"
            
            cursor.execute(f'''
                SELECT id, embedding_vector, scene_description, timestamp 
                FROM frames 
                WHERE embedding_vector IS NOT NULL
                {time_filter}
                ORDER BY timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Calculate similarity for each frame
            results = []
            for row in rows:
                frame_id, emb_bytes, scene_desc, timestamp = row
                
                if emb_bytes:
                    frame_embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                    # Cosine similarity
                    similarity = self._cosine_similarity(query_embedding, frame_embedding)
                    
                    results.append({
                        'frame_id': frame_id,
                        'similarity': similarity,
                        'scene_description': scene_desc,
                        'timestamp': timestamp
                    })
            
            # Sort by similarity (highest first) and return top k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:k]
        
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            a = a.flatten().astype(np.float32)
            b = b.flatten().astype(np.float32)
            
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot_product / (norm_a * norm_b))
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


def create_memory(db_path: str = None) -> ObjectMemory:
    """Factory function to create memory system"""
    return ObjectMemory(db_path)
