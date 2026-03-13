"""
Gemini Vision API integration for scene description
"""

import logging
import base64
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import config

logger = logging.getLogger(__name__)

class GeminiSceneDescriptor:
    """Generate scene descriptions using Gemini Vision API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini API client
        
        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Google API client"""
        try:
            import google.generativeai as genai
            
            if not self.api_key:
                logger.error("Gemini API key not provided")
                return
            
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(config.GEMINI_MODEL)
            
            logger.info("✓ Gemini Vision API initialized")
        except ImportError:
            logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {e}")
            raise
    
    def describe_scene(self, frame: np.ndarray, detections: list = None) -> Optional[str]:
        """
        Generate scene description using Gemini Vision
        
        Args:
            frame: Input frame (BGR)
            detections: List of detected objects
        
        Returns:
            Scene description string or None if error
        """
        if self.client is None:
            logger.warning("Gemini API not initialized")
            return None
        
        try:
            # Convert frame to JPEG bytes
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                logger.error("Failed to encode frame")
                return None
            
            # Encode to base64
            image_data = base64.standard_b64encode(buffer).decode()
            
            # Create custom prompt based on detections
            prompt = self._build_prompt(detections)
            
            # Call Gemini API
            message = self.client.generate_content([
                {
                    "mime_type": "image/jpeg",
                    "data": image_data,
                },
                prompt
            ])
            
            description = message.text
            logger.debug(f"Generated scene description: {description[:100]}...")
            return description
        
        except Exception as e:
            logger.error(f"Error describing scene: {e}")
            return None
    
    def describe_scene_for_object(self, frame: np.ndarray, 
                                  object_name: str, 
                                  bbox: np.ndarray) -> Optional[str]:
        """
        Generate description focusing on a specific object's location
        
        Args:
            frame: Input frame
            object_name: Name of object
            bbox: Bounding box of object
        
        Returns:
            Description of object location in scene
        """
        if self.client is None:
            return None
        
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return None
            
            image_data = base64.standard_b64encode(buffer).decode()
            
            prompt = f"""Look at this image. There is a '{object_name}' in the frame.
            Describe:
            1. What room/environment is this?
            2. What furniture or surfaces are nearby the {object_name}?
            3. Describe the {object_name}'s exact location in simple terms a visually impaired person would understand.
            
            Be specific and concise. Format: "Your {object_name} is [description of location in the scene]."
            """
            
            message = self.client.generate_content([
                {
                    "mime_type": "image/jpeg",
                    "data": image_data,
                },
                prompt
            ])
            
            return message.text
        
        except Exception as e:
            logger.error(f"Error describing object: {e}")
            return None
    
    def extract_location_info(self, frame: np.ndarray) -> Optional[dict]:
        """
        Extract location/environment information from frame
        
        Args:
            frame: Input frame
        
        Returns:
            Dictionary with location details
        """
        if self.client is None:
            return None
        
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return None
            
            image_data = base64.standard_b64encode(buffer).decode()
            
            prompt = """Analyze this image and provide location information in JSON format:
            {
                "room_type": "bedroom/kitchen/living_room/etc",
                "main_furniture": ["list of furniture items visible"],
                "lighting": "bright/dim/natural/artificial",
                "environment_type": "indoor/outdoor",
                "notable_features": ["special features like windows, doors, etc"],
                "surface_types": ["types of surfaces: wooden, carpeted, tiled, etc"]
            }
            
            Respond with ONLY the JSON, no other text.
            """
            
            message = self.client.generate_content([
                {
                    "mime_type": "image/jpeg",
                    "data": image_data,
                },
                prompt
            ])
            
            import json
            try:
                location_info = json.loads(message.text)
                return location_info
            except json.JSONDecodeError:
                logger.warning("Failed to parse location info JSON")
                return None
        
        except Exception as e:
            logger.error(f"Error extracting location info: {e}")
            return None
    
    def _build_prompt(self, detections: list = None) -> str:
        """Build dynamic prompt based on detections"""
        base_prompt = config.GEMINI_SCENE_PROMPT
        
        if detections:
            detected_items = ", ".join([d.get('class_name', 'unknown') for d in detections])
            base_prompt += f"\n\nDetected objects in the scene: {detected_items}"
        
        return base_prompt


def create_scene_descriptor(api_key: str = None) -> GeminiSceneDescriptor:
    """Factory function to create scene descriptor"""
    try:
        return GeminiSceneDescriptor(api_key)
    except Exception as e:
        logger.error(f"Failed to create scene descriptor: {e}")
        return None
