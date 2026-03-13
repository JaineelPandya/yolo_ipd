"""
Query module - natural language interface to find objects
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import config
from memory.storage import ObjectMemory

logger = logging.getLogger(__name__)

class ObjectQueryEngine:
    """Query interface for finding objects"""
    
    def __init__(self, memory: ObjectMemory):
        """
        Initialize query engine
        
        Args:
            memory: ObjectMemory instance
        """
        self.memory = memory
    
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
            
            # Generate natural language response
            return self._format_response(result)
        
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error searching for {object_name}. Please try again."
    
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
        
        # Build response
        response = f"Your {object_name} was last seen at {time_str}"
        
        if date_str:
            response += f" on {date_str}"
        
        response += "."
        
        # Add scene description if available
        if scene_desc:
            response += f"\n{scene_desc}"
        
        # Add confidence indicator if it's not very high
        if confidence and confidence < 0.8:
            response += f"\n(Confidence: {confidence*100:.0f}%)"
        
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
        
        # Common patterns
        if "where" in query and "my" in query:
            # Extract object name
            # Remove common words
            words = query.replace("where is my ", "").replace("where are my ", "").replace("?", "").split()
            object_name = " ".join(words)
            
            if object_name:
                return self.get_last_seen(object_name)
        
        if "what did you see" in query or "what did i have" in query:
            return self.get_today_summary()
        
        if "history" in query or "where was my" in query:
            # Extract object name
            words = query.replace("history", "").replace("where was my", "").replace("?", "").split()
            object_name = " ".join(words)
            
            if object_name:
                return self.get_object_history(object_name)
        
        # Default fallback
        # Try to recognize common objects being asked about
        for obj_name in config.SMALL_OBJECTS_TO_TRACK:
            if obj_name in query:
                return self.get_last_seen(obj_name)
        
        return "Please specify which object you're looking for. Say 'where is my [object]?'"


def create_query_engine(memory: ObjectMemory) -> ObjectQueryEngine:
    """Factory function to create query engine"""
    return ObjectQueryEngine(memory)
