#!/usr/bin/env python3
"""
Test script to verify the PathRAGArizeAdapter can connect to Phoenix
"""

import os
import sys
import logging
import uuid
from typing import Dict, Any
from datetime import datetime

# Add the project to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("adapter_phoenix_test")

def test_adapter_phoenix_connection():
    """Test the PathRAGArizeAdapter with the Phoenix Docker container."""
    try:
        # Import the adapter
        from implementations.pathrag.arize_integration.adapter import PathRAGArizeAdapter
        logger.info("✅ Successfully imported PathRAGArizeAdapter")
    except ImportError as e:
        logger.error(f"❌ Failed to import PathRAGArizeAdapter: {e}")
        return False
    
    # Create a test configuration
    config = {
        "phoenix_host": "localhost",
        "phoenix_port": 8084,  # Updated port to match Docker container
        "track_performance": True,
        "model_name": "test-model",
    }
    
    # Initialize the adapter
    try:
        adapter = PathRAGArizeAdapter(config)
        logger.info("✅ Successfully created PathRAGArizeAdapter instance")
    except Exception as e:
        logger.error(f"❌ Failed to create PathRAGArizeAdapter: {e}")
        return False
    
    # Initialize the adapter
    try:
        adapter.initialize()
        logger.info("✅ Successfully initialized PathRAGArizeAdapter")
    except Exception as e:
        logger.error(f"❌ Failed to initialize PathRAGArizeAdapter: {e}")
        return False
    
    # Create a test trace
    try:
        trace_id = str(uuid.uuid4())
        trace_data = {
            "id": trace_id,
            "name": "Test Query from test_adapter_phoenix.py",
            "model": "test-model",
            "inputs": {
                "query": "This is a test query"
            },
            "outputs": {
                "response": "This is a test response"
            },
            "metadata": {
                "test_timestamp": str(datetime.now()),
                "test_source": "test_adapter_phoenix.py"
            },
            "latency_ms": 100,
            "token_usage": {
                "prompt": 10,
                "completion": 5,
                "total": 15
            }
        }
        
        # Call the log method
        adapter._log_to_phoenix(trace_data)
        logger.info(f"✅ Successfully logged trace {trace_id} to Phoenix")
        logger.info(f"📊 View trace at http://localhost:8084/span/{trace_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to log trace to Phoenix: {e}")
        logger.error(f"Error details: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Testing PathRAGArizeAdapter with Phoenix Docker container")
    result = test_adapter_phoenix_connection()
    
    if result:
        logger.info("✅ All adapter Phoenix tests passed!")
        logger.info("📊 Visit http://localhost:8084 to view your traces in Phoenix")
        sys.exit(0)
    else:
        logger.error("❌ Adapter Phoenix tests failed")
        sys.exit(1)
