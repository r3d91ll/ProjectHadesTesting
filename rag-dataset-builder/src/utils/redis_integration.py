"""
Redis Integration for PathRAG Dataset Builder

This module provides integration between the PathRAG dataset builder and Redis,
allowing for high-performance vector storage and retrieval using Redis as a cache.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Add the main project root to the Python path
main_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if main_project_root not in sys.path:
    sys.path.insert(0, main_project_root)

# Import the PathRAG Redis adapter
try:
    from src.pathrag_redis_integration import PathRAGRedisAdapter
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class RedisIntegration:
    """Redis integration for PathRAG dataset builder"""
    
    def __init__(self):
        """Initialize the Redis integration"""
        self.redis_enabled = os.environ.get("PATHRAG_REDIS_ENABLED", "false").lower() == "true"
        
        if not self.redis_enabled:
            logger.info("Redis integration is disabled")
            return
        
        if not REDIS_AVAILABLE:
            logger.error("Redis integration is enabled but PathRAGRedisAdapter is not available")
            self.redis_enabled = False
            return
        
        # Get Redis configuration from environment variables
        self.redis_host = os.environ.get("PATHRAG_REDIS_HOST", "localhost")
        self.redis_port = int(os.environ.get("PATHRAG_REDIS_PORT", "6379"))
        self.redis_db = int(os.environ.get("PATHRAG_REDIS_DB", "0"))
        self.redis_password = os.environ.get("PATHRAG_REDIS_PASSWORD", "")
        self.redis_prefix = os.environ.get("PATHRAG_REDIS_PREFIX", "pathrag")
        self.redis_ttl = int(os.environ.get("PATHRAG_REDIS_TTL", "604800"))  # 7 days
        self.vector_dim = int(os.environ.get("PATHRAG_VECTOR_DIM", "384"))
        
        # Initialize the Redis adapter
        try:
            self.adapter = PathRAGRedisAdapter(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                vector_dim=self.vector_dim,
                prefix=self.redis_prefix,
                ttl=self.redis_ttl
            )
            logger.info(f"Redis integration initialized with prefix '{self.redis_prefix}'")
            
            # Log Redis stats
            stats = self.adapter.get_stats()
            logger.info(f"Redis stats: {stats}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis adapter: {e}")
            self.redis_enabled = False
    
    def store_vector(self, path: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a vector in Redis"""
        if not self.redis_enabled:
            return False
        
        try:
            return self.adapter.store_vector(path, vector, metadata)
        except Exception as e:
            logger.error(f"Error storing vector in Redis: {e}")
            return False
    
    def get_vector(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve a vector from Redis"""
        if not self.redis_enabled:
            return None
        
        try:
            return self.adapter.get_vector(path)
        except Exception as e:
            logger.error(f"Error retrieving vector from Redis: {e}")
            return None
    
    def search_similar_vectors(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in Redis"""
        if not self.redis_enabled:
            return []
        
        try:
            return self.adapter.search_similar_vectors(query_vector, top_k)
        except Exception as e:
            logger.error(f"Error searching vectors in Redis: {e}")
            return []
    
    def delete_vector(self, path: str) -> bool:
        """Delete a vector from Redis"""
        if not self.redis_enabled:
            return False
        
        try:
            return self.adapter.delete_vector(path)
        except Exception as e:
            logger.error(f"Error deleting vector from Redis: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """Clear the Redis cache"""
        if not self.redis_enabled:
            return False
        
        try:
            return self.adapter.clear_cache()
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.redis_enabled:
            return {"redis_enabled": False}
        
        try:
            return self.adapter.get_stats()
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {"error": str(e)}


# Singleton instance
redis_integration = RedisIntegration()
