#!/usr/bin/env python3
"""
Test script to verify Arize Phoenix connection
"""

import requests
import logging
import time
import sys
import uuid

# Try the same import pattern as in the adapter
try:
    from arize.phoenix.session import Session
    from arize.phoenix.trace import LLMTrace
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    logging.warning("Arize Phoenix not available. Checking if it's installed with pip...")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phoenix_test")

def test_phoenix_connection(host="localhost", port=8084):
    """Test connection to Arize Phoenix server."""
    phoenix_url = f"http://{host}:{port}"
    
    # First check health endpoint
    logger.info(f"Testing Phoenix health at {phoenix_url}/health")
    try:
        response = requests.get(f"{phoenix_url}/health", timeout=2)
        if response.status_code == 200:
            logger.info("✅ Phoenix health check successful")
        else:
            logger.error(f"❌ Phoenix health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Phoenix health check failed: {e}")
        return False
    
    # Try to initialize Phoenix session
    logger.info(f"Testing Phoenix session at {phoenix_url}")
    try:
        session = Session(url=phoenix_url)
        logger.info("✅ Phoenix session created successfully")
    except Exception as e:
        logger.error(f"❌ Phoenix session creation failed: {e}")
        return False
    
    # Try to send a simple trace
    logger.info("Testing Phoenix trace creation")
    try:
        # Create and save trace to Phoenix
        trace_id = str(uuid.uuid4())
        trace = LLMTrace(
            id=trace_id,
            name="Phoenix Test Trace",
            model="test-model",
            events=[],
            inputs={
                "query": "This is a test query"
            },
            outputs={
                "response": "This is a test response"
            }
        )
        session.log_trace(trace)
        logger.info(f"✅ Phoenix trace {trace_id} created successfully")
        logger.info(f"📊 View traces at {phoenix_url}")
        return True
    except Exception as e:
        logger.error(f"❌ Phoenix trace creation failed: {e}")
        return False

if __name__ == "__main__":
    # Check if Phoenix is available
    if not PHOENIX_AVAILABLE:
        logger.error("❌ Arize Phoenix is not properly installed in this environment")
        logger.error("Run: pip install arize-phoenix")
        sys.exit(1)
    
    # Allow custom host/port from command line
    host = "localhost"
    port = 8084
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    logger.info(f"Testing Phoenix connection at {host}:{port}")
    result = test_phoenix_connection(host, port)
    
    if result:
        logger.info("✅ All Phoenix tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Phoenix tests failed")
        sys.exit(1)
