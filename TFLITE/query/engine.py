"""
Query module - natural language interface to find objects
Supports both keyword-based retrieval and semantic (RAG) search
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import cv2
import numpy as np
import json
import config
from memory.storage import ObjectMemory
from memory.embeddings import create_embedder

logger = logging.getLogger(__name__)

class ObjectQueryEngine:
    """Query interface for finding objects — with RAG support and on-demand Gemini descriptions"""
    
    def __init__(self, memory: ObjectMemory, scene_descriptor=None, use_semantic: bool = True):
        """
        Initialize query engine
        
        Args:
            memory: ObjectMemory instance
            scene_descriptor: GeminiSceneDescriptor for generating missing descriptions
            use_semantic: Enable semantic search with embeddings
        """
        self.memory = memory
        self.scene_descriptor = scene_descriptor
        self.use_semantic = use_semantic
        self._last_gemini_call = 0.0  # Track last Gemini call timestamp
        
        # Initialize embedder for semantic search
        if self.use_semantic:
            self.embedder = create_embedder()
            if self.embedder and self.embedder.model:
                logger.info("✓ Semantic search (RAG) enabled")
            else:
                logger.warning("✗ Semantic search disabled - embedder unavailable")
                self.use_semantic = False
        else:
            self.embedder = None
    
    def get_last_seen(self, object_name: str, 
                     time_range_minutes: int = None) -> Optional[str]:
        """
        Get natural language description of where object was last seen
        
        Args:
            object_name: Name of object to search
            time_range_minutes: Search within N minutes (default: config setting)
        
        Returns:
            Natural language string describing object location
        """
        if time_range_minutes is None:
            time_range_minutes = config.RECENT_SEARCH_RANGE
        
        try:
            result = self.memory.get_last_seen(object_name, time_range_minutes)
            
            if not result:
                return f"I don't have any recent information about your {object_name}."
            
            # Check if scene description exists
            if not result.get('scene_description'):
                # Generate description on-demand
                result = self._generate_scene_description_on_demand(result)
            
            # Generate natural language response
            return self._format_response(result)
        
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error searching for {object_name}. Please try again."
    
    def _generate_scene_description_on_demand(self, db_record: Dict) -> Dict:
        """
        Generate scene description for a database record if it doesn't exist
        
        Args:
            db_record: Database record from memory
            
        Returns:
            Updated record with scene_description
        """
        if not self.scene_descriptor:
            logger.warning("Scene descriptor not available for on-demand generation")
            return db_record
            
        # Check rate limiting (10-15 seconds between calls)
        import time
        current_time = time.time()
        time_since_last_call = current_time - self._last_gemini_call
        if time_since_last_call < 15.0:  # 15 seconds minimum gap
            logger.info(f"Rate limiting: {15.0 - time_since_last_call:.1f}s until next Gemini call")
            return db_record
            
        # Load the frame image
        image_path = db_record.get('image_path')
        if not image_path:
            logger.warning("No image path in record for scene description generation")
            return db_record
            
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"Could not load frame from {image_path}")
                return db_record
                
            # Extract bbox and object name
            bbox = db_record.get('bbox')
            object_name = db_record.get('object_name')
            
            if bbox and object_name:
                # Convert bbox from JSON string to numpy array if needed
                if isinstance(bbox, str):
                    bbox = np.array(json.loads(bbox))
                    
                # Generate scene description
                logger.info(f"Generating on-demand scene description for {object_name}")
                scene_desc = self.scene_descriptor.describe_scene_for_object(
                    frame, object_name, bbox
                )
                
                if scene_desc:
                    # Update the database record
                    timestamp = db_record.get('timestamp')
                    success = self.memory.update_object_description(
                        object_name, timestamp, scene_desc
                    )
                    
                    if success:
                        # Update the record in memory
                        db_record['scene_description'] = scene_desc
                        self._last_gemini_call = time.time()
                        logger.info(f"Successfully generated and cached scene description for {object_name}")
                    else:
                        logger.error(f"Failed to update database with scene description for {object_name}")
                else:
                    logger.warning(f"Gemini failed to generate scene description for {object_name}")
                    
        except Exception as e:
            logger.error(f"Error generating on-demand scene description: {e}")
            
        return db_record
    
    def find_by_location(self, location_description: str) -> List[str]:
        """
        Find objects by location description
        
        Args:
            location_description: Description like "cupboard", "table", "shelf"
        
        Returns:
            List of objects found at that location
        """
        try:
            # Split location into keywords
            keywords = location_description.lower().split()
            
            results = self.memory.search_objects_by_location(keywords)
            
            if not results:
                return [f"No objects found near {location_description}"]
            
            # Group by object name
            objects_by_name = {}
            for result in results:
                name = result['object_name']
                if name not in objects_by_name:
                    objects_by_name[name] = result
            
            responses = []
            for name, result in objects_by_name.items():
                response = f"Your {name} was last seen at {result['timestamp']} "
                if result['scene_description']:
                    response += f"in the {location_description}. {result['scene_description']}"
                responses.append(response)
            
            return responses
        
        except Exception as e:
            logger.error(f"Location search error: {e}")
            return ["Error searching by location"]
    
    def get_today_summary(self) -> str:
        """Get summary of all objects detected today"""
        try:
            objects = self.memory.get_all_objects_today()
            
            if not objects:
                return "No objects tracked today yet."
            
            summary = "Objects detected today:\n"
            for obj in objects:
                summary += f"- {obj['object_name']}: last seen at {obj['last_seen']}\n"
            
            return summary
        
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return "Error generating summary"
    
    def get_object_history(self, object_name: str, limit: int = 5) -> str:
        """Get history of object sightings"""
        try:
            history = self.memory.get_object_history(object_name, limit)
            
            if not history:
                return f"No history found for {object_name}."
            
            response = f"Location history for {object_name} (last {limit} sightings):\n"
            for i, record in enumerate(history, 1):
                timestamp = record['timestamp']
                if record['scene_description']:
                    response += f"{i}. {timestamp}: {record['scene_description']}\n"
                else:
                    response += f"{i}. {timestamp}: Detected at location\n"
            
            return response
        
        except Exception as e:
            logger.error(f"History error: {e}")
            return "Error retrieving history"
    
    def _format_response(self, db_record: Dict) -> str:
        """Format database record into natural language response"""
        object_name = db_record.get('object_name', 'object')
        timestamp = db_record.get('timestamp', 'unknown time')
        scene_desc = db_record.get('scene_description', '')
        confidence = db_record.get('confidence', 0)
        
        # Parse timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%I:%M %p")  # 10:42 PM format
            date_str = dt.strftime("%A, %B %d")  # Monday, March 14
        except:
            time_str = "an unknown time"
            date_str = ""
        
        # Build response — Line 1: Time
        response = f"🕐 Your **{object_name}** was last seen at **{time_str}**"
        if date_str:
            response += f" on {date_str}"
        response += "."
        
        # Line 2: Scene description from Gemini (only if present)
        if scene_desc and scene_desc.strip():
            response += f"\n\n📍 {scene_desc}"
        else:
            # If no description available (rate limited or error), show status
            response += (
                "\n\n📍 **Location details**: "
                "Scene description will be generated on-demand when you query this object. "
                "Enable Gemini API in settings for detailed descriptions."
            )
        
        # Line 3: Confidence indicator
        if confidence:
            conf_pct = confidence * 100
            if conf_pct >= 80:
                response += f"\n\n✅ High confidence detection ({conf_pct:.0f}%)"
            elif conf_pct >= 50:
                response += f"\n\n⚠️ Moderate confidence ({conf_pct:.0f}%)"
            else:
                response += f"\n\n❓ Low confidence ({conf_pct:.0f}%) — result may be inaccurate"
        
        return response
    
    def process_voice_query(self, query: str) -> str:
        """
        Process natural language voice query
        
        Args:
            query: User query like "Where is my phone?"
        
        Returns:
            Response string
        """
        query = query.lower().strip()
        
        # Common "where is" patterns
        if "where" in query and "my" in query:
            words = query.replace("where is my ", "").replace("where are my ", "").replace("?", "").split()
            object_name = " ".join(words).strip()
            if object_name:
                return self.get_last_seen(object_name)
        
        # "find my X" / "locate my X" patterns
        for prefix in ["find my ", "locate my ", "search for my ", "look for my "]:
            if prefix in query:
                object_name = query.split(prefix, 1)[1].replace("?", "").strip()
                if object_name:
                    return self.get_last_seen(object_name)
        
        # "have you seen my X" pattern
        if "have you seen" in query and "my" in query:
            words = query.split("my ", 1)
            if len(words) > 1:
                object_name = words[1].replace("?", "").strip()
                if object_name:
                    return self.get_last_seen(object_name)
        
        # Summary queries
        if "what did you see" in query or "what did i have" in query or "show me everything" in query:
            return self.get_today_summary()
        
        # History queries
        if "history" in query or "where was my" in query:
            words = query.replace("history", "").replace("where was my", "").replace("?", "").split()
            object_name = " ".join(words).strip()
            if object_name:
                return self.get_object_history(object_name)
        
        # Fallback: try to match any known object name in the query
        for obj_name in config.SMALL_OBJECTS_TO_TRACK:
            if obj_name in query:
                return self.get_last_seen(obj_name)
        
        # Also check all 80 COCO class names
        coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        for obj_name in coco_names:
            if obj_name in query:
                return self.get_last_seen(obj_name)
        
        return "I couldn't understand which object you're looking for. Try asking:\n• 'Where is my phone?'\n• 'Find my teddy bear'\n• 'Have you seen my keys?'"
    
    # ==================== SEMANTIC SEARCH (RAG) ====================
    
    def semantic_search(self, query_text: str, k: int = 5, 
                       time_range_minutes: int = None) -> List[Dict]:
        """
        Perform semantic search using embeddings (Retrieval-Augmented Generation)
        
        Args:
            query_text: Natural language query (e.g., "phone on table")
            k: Number of similar scenes to retrieve
            time_range_minutes: Optional time filter
        
        Returns:
            List of similar frame records with descriptions
        """
        if not self.use_semantic or not self.embedder:
            logger.warning("Semantic search not available")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.embedder.embed_text(query_text)
            if query_embedding is None:
                return []
            
            # Search database for similar scenes
            results = self.memory.semantic_search(
                query_embedding, 
                k=k, 
                time_range_minutes=time_range_minutes
            )
            
            logger.info(f"Semantic search found {len(results)} similar scenes")
            return results
        
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def semantic_location_search(self, location_description: str) -> List[str]:
        """
        Find objects at a location using semantic understanding
        
        Args:
            location_description: Natural language location (e.g., "on the wooden shelf")
        
        Returns:
            List of response strings with objects found
        """
        if not self.use_semantic:
            # Fallback to keyword search
            return self.find_by_location(location_description)
        
        try:
            # Semantic search for similar scenes
            results = self.semantic_search(location_description, k=10)
            
            if not results:
                return [f"No objects found matching '{location_description}'"]
            
            # Group results by object type
            responses = []
            for result in results:
                scene = result.get('scene_description', '')
                similarity = result.get('similarity', 0)
                timestamp = result.get('timestamp', '')
                
                if similarity > 0.6:  # Only include reasonably similar results
                    response = f"📍 Similarity: {similarity*100:.0f}% - {scene}"
                    if timestamp:
                        response += f" (at {timestamp})"
                    responses.append(response)
            
            return responses if responses else [f"No objects found near {location_description}"]
        
        except Exception as e:
            logger.error(f"Semantic location search error: {e}")
            return self.find_by_location(location_description)
    
    def smart_search(self, query: str) -> str:
        """
        Intelligent search combining keyword and semantic approaches
        
        Args:
            query: User query (natural language)
        
        Returns:
            Best response
        """
        # First try exact keyword match (faster)
        if "where" in query and "my" in query:
            words = query.replace("where", "").replace("my", "").replace("?", "").split()
            object_name = " ".join(words).strip()
            if object_name:
                exact_result = self.get_last_seen(object_name)
                if "recent information" not in exact_result:
                    return exact_result
        
        # If exact match failed and semantic is available, try semantic search
        if self.use_semantic:
            logger.info("Keyword search inconclusive, trying semantic search...")
            semantic_results = self.semantic_search(query, k=3)
            
            if semantic_results:
                response = f"🧠 Based on semantic analysis of similar scenes:\n\n"
                for i, result in enumerate(semantic_results, 1):
                    scene = result.get('scene_description', 'scene')
                    similarity = result.get('similarity', 0)
                    response += f"{i}. ({similarity*100:.0f}% match) {scene}\n"
                return response
        
        # Fallback to voice query processing
        return self.process_voice_query(query)


def create_query_engine(memory: ObjectMemory, scene_descriptor=None, use_semantic: bool = True) -> ObjectQueryEngine:
    """Factory function to create query engine with optional Gemini support"""
    return ObjectQueryEngine(memory, scene_descriptor=scene_descriptor, use_semantic=use_semantic)
