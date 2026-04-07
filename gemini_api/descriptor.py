"""
Gemini Vision API — scene description for Object Memory Assistant.

Uses google-genai SDK (google.genai.Client) with inline JPEG bytes.
Includes built-in rate limiting to avoid quota exhaustion.
"""

import logging
import time
import cv2
import numpy as np
from typing import Optional
import config

logger = logging.getLogger(__name__)


class GeminiSceneDescriptor:
    """Generate per-object scene descriptions using Gemini Vision."""

    def __init__(self, api_key: str = None):
        self.api_key = (api_key or config.GEMINI_API_KEY or "").strip()
        self.model_name = config.GEMINI_MODEL
        self.client = None
        self._last_call_ts: float = 0.0   # epoch seconds of last Gemini call
        self.error_count = 0
        
        # Debug logging
        logger.info(f"🔍 GeminiSceneDescriptor init:")
        logger.info(f"   API Key provided: {bool(api_key)}")
        logger.info(f"   API Key from config: {bool(config.GEMINI_API_KEY)}")
        logger.info(f"   Final API Key: {len(self.api_key)} chars")
        
        self._init_client()

    # ------------------------------------------------------------------
    def _init_client(self):
        if not self.api_key:
            logger.error("❌ GEMINI_API_KEY is EMPTY or NOT SET")
            logger.error("   Set GEMINI_API_KEY in .env file or environment variable")
            return
        
        logger.info(f"🔑 Initializing Gemini client...")
        logger.info(f"   API Key: {self.api_key[:10]}...{self.api_key[-4:]}")
        logger.info(f"   Model: {self.model_name}")
        
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"✅ Gemini client initialized successfully!")
        except ImportError:
            logger.error("❌ google-genai not installed. Run: pip install google-genai")
        except Exception as e:
            logger.error(f"❌ Gemini init error: {e}")
            logger.error(f"   Type: {type(e).__name__}")
            if "API_KEY" in str(e) or "key" in str(e).lower():
                logger.error("   → This looks like an API key issue")
                logger.error(f"   → API Key length: {len(self.api_key)} chars")
                if len(self.api_key) < 20:
                    logger.error("   → API Key is too short! Check .env file")

    # ------------------------------------------------------------------
    def _can_call(self) -> bool:
        """Enforce minimum interval between API calls."""
        return time.time() - self._last_call_ts >= config.GEMINI_MIN_INTERVAL_SECONDS

    def _mark_call(self):
        self._last_call_ts = time.time()

    # ------------------------------------------------------------------
    def _frame_to_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode BGR frame → JPEG bytes (compressed for API efficiency)."""
        try:
            # Resize to max 640px wide to reduce token cost
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes() if ok else None
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return None

    # ------------------------------------------------------------------
    def describe_scene_for_object(
        self,
        frame: np.ndarray,
        object_name: str,
        bbox: np.ndarray,
    ) -> Optional[str]:
        """
        Generate a natural-language location description for `object_name`
        visible in `frame`.

        Returns None if rate-limited, client not ready, or API error.
        """
        if self.client is None:
            if self.error_count == 0:
                logger.debug(f"⚠️ Gemini client not initialized for '{object_name}'")
                self.error_count += 1
            return None
        if not self._can_call():
            # Don't spam logs, just silently skip
            return None

        image_bytes = self._frame_to_jpeg(frame)
        if image_bytes is None:
            logger.debug(f"Could not encode frame for '{object_name}'")
            return None

        prompt = config.GEMINI_OBJECT_PROMPT_TEMPLATE.format(object_name=object_name)

        try:
            from google.genai import types
            self._mark_call()
            logger.debug(f"🔄 Calling Gemini for '{object_name}'...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            text = (response.text or "").strip()
            if not text:
                logger.warning(f"❌ Gemini returned empty response for '{object_name}'")
                return None
            logger.info(f"✅ Gemini Scene [{object_name}]: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f"❌ Gemini API error for '{object_name}': {e}")
            self.error_count += 1
            if self.error_count >= 3:
                logger.error(f"   Gemini API has failed {self.error_count} times. Check API key?")
            return None

    # ------------------------------------------------------------------
    def describe_scene(
        self,
        frame: np.ndarray,
        detections: list = None,
    ) -> Optional[str]:
        """
        General scene description (used to annotate stored frames).
        Returns None if rate-limited or error.
        """
        if self.client is None:
            return None
        if not self._can_call():
            return None

        image_bytes = self._frame_to_jpeg(frame)
        if image_bytes is None:
            return None

        detected_items = ""
        if detections:
            names = ", ".join(d.get("class_name", "?") for d in detections[:6])
            detected_items = f"\n\nDetected objects: {names}"

        prompt = (
            "Briefly describe this scene in 1–2 sentences for a visually "
            "impaired person. Focus on the room type and where objects are placed."
            + detected_items
        )

        try:
            from google.genai import types
            self._mark_call()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            return (response.text or "").strip() or None
        except Exception as e:
            logger.error(f"Gemini scene describe error: {e}")
            return None


def create_scene_descriptor(api_key: str = None) -> Optional[GeminiSceneDescriptor]:
    """Factory — returns None if Gemini cannot be initialised."""
    try:
        return GeminiSceneDescriptor(api_key)
    except Exception as e:
        logger.error(f"Failed to create GeminiSceneDescriptor: {e}")
        return None
