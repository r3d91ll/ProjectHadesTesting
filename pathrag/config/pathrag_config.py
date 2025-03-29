"""
PathRAG Configuration

This module provides configuration settings for PathRAG with Arize Phoenix integration.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# LLM API settings
# Model provider can be 'openai' or 'ollama'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')

# OpenAI API settings (used when LLM_PROVIDER is 'openai')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_API_MODEL = os.getenv('OPENAI_API_MODEL', 'gpt-4o')
OPENAI_API_MAX_TOKENS = int(os.getenv('OPENAI_API_MAX_TOKENS', '8192'))

# Ollama API settings (used when LLM_PROVIDER is 'ollama')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')

# Arize Phoenix settings
PHOENIX_HOST = os.getenv('PHOENIX_HOST', 'localhost')
PHOENIX_PORT = int(os.getenv('PHOENIX_PORT', '8084'))
PHOENIX_PROJECT_NAME = os.getenv('PHOENIX_PROJECT_NAME', 'pathrag-inference')
PHOENIX_INFERENCE_PROJECT_NAME = os.getenv('PHOENIX_INFERENCE_PROJECT_NAME', 'pathrag-inference')

# Document storage settings
DOCUMENT_STORE_TYPE = os.getenv('DOCUMENT_STORE_TYPE', 'chroma')
DOCUMENT_STORE_PATH = os.getenv('DOCUMENT_STORE_PATH', 
                               os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                           'data/chroma_db'))

# Embedding model settings
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'openai')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1536'))

# Performance tracking settings
TRACK_PERFORMANCE = os.getenv('TRACK_PERFORMANCE', 'true').lower() == 'true'
PERFORMANCE_METRICS_INTERVAL = int(os.getenv('PERFORMANCE_METRICS_INTERVAL', '60'))

def get_config() -> Dict[str, Any]:
    """
    Get the configuration settings for PathRAG.
    
    Returns:
        Dict containing configuration settings
    """
    return {
        # LLM provider
        "llm_provider": LLM_PROVIDER,
        
        # OpenAI API settings
        "openai_api_key": OPENAI_API_KEY,
        "model_name": OPENAI_API_MODEL,
        "max_tokens": OPENAI_API_MAX_TOKENS,
        
        # Ollama API settings
        "ollama_host": OLLAMA_HOST,
        "ollama_port": OLLAMA_PORT,
        "ollama_model": OLLAMA_MODEL,
        
        # PathRAG settings
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "similarity_top_k": 5,
        "graph_construction": "similarity",
        "path_search_method": "bfs",
        "path_time_limit": 10,  # seconds
        
        # Document storage settings
        "document_store_type": DOCUMENT_STORE_TYPE,
        "document_store_path": DOCUMENT_STORE_PATH,
        
        # Embedding settings
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        
        # Arize Phoenix settings for telemetry
        "phoenix_host": PHOENIX_HOST,
        "phoenix_port": PHOENIX_PORT,
        "project_name": PHOENIX_INFERENCE_PROJECT_NAME,
        "track_performance": TRACK_PERFORMANCE,
        "performance_metrics_interval": PERFORMANCE_METRICS_INTERVAL
    }

def validate_config() -> bool:
    """
    Validate the configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    # Check if OpenAI API key is set
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY is not set. Set it in the .env file.")
        return False
    
    # Check if document store path exists or create it
    os.makedirs(DOCUMENT_STORE_PATH, exist_ok=True)
    
    # Check if Arize Phoenix is reachable if tracking is enabled
    if TRACK_PERFORMANCE:
        import requests
        try:
            response = requests.get(f"http://{PHOENIX_HOST}:{PHOENIX_PORT}/health", timeout=2)
            if response.status_code != 200:
                print(f"Warning: Arize Phoenix health check failed with status code {response.status_code}")
                print(f"Make sure Arize Phoenix is running at {PHOENIX_HOST}:{PHOENIX_PORT}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to connect to Arize Phoenix: {e}")
            print(f"Make sure Arize Phoenix is running at {PHOENIX_HOST}:{PHOENIX_PORT}")
            return False
    
    return True
