"""Integration module - Gemini + Database connections"""

from .gemini_db_bridge import GeminiDatabaseBridge, create_bridge

__all__ = ['GeminiDatabaseBridge', 'create_bridge']
