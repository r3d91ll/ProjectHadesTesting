#!/usr/bin/env python3
"""
Test script to verify Arize Phoenix integration with your PathRAG adapter
"""

import os
import sys
import time
import uuid
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phoenix_adapter_test")

# Arize Phoenix imports for telemetry
PHOENIX_AVAILABLE = False
try:
    from arize.phoenix.session import Session
    from arize.phoenix.trace import LLMTrace
    PHOENIX_AVAILABLE = True
    logger.info("✅ Successfully imported Arize Phoenix")
except ImportError as e:
    logger.error(f"❌ Failed to import Arize Phoenix: {e}")
    logger.error("Run: pip install arize-phoenix")

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
    
    if not PHOENIX_AVAILABLE:
        logger.error("❌ Arize Phoenix libraries not available")
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
            name="PathRAG Test Trace",
            model="test-model",
            inputs={
                "query": "This is a test query from dev-utilities/phoenix_adapter_test.py"
            },
            outputs={
                "response": "This is a test response"
            },
            metadata={
                "test_timestamp": str(datetime.now()),
                "test_source": "phoenix_adapter_test.py"
            }
        )
        session.log_trace(trace)
        logger.info(f"✅ Phoenix trace {trace_id} created successfully")
        logger.info(f"📊 View trace at {phoenix_url}/span/{trace_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Phoenix trace creation failed: {e}")
        logger.error(f"Error details: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if Phoenix is available
    if not PHOENIX_AVAILABLE:
        logger.error("❌ Could not import Arize Phoenix libraries")
        logger.error("Make sure you've run: pip install arize-phoenix")
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
        logger.info(f"Visit http://{host}:{port} to view your traces in Phoenix")
        sys.exit(0)
    else:
        logger.error("❌ Phoenix tests failed")
        sys.exit(1)
