#!/usr/bin/env python3
"""
Test script to verify Arize Phoenix connection using direct HTTP
"""

import requests
import logging
import time
import sys
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phoenix_http_test")

def test_phoenix_connection(host="localhost", port=8084):
    """Test connection to Arize Phoenix server using direct HTTP."""
    phoenix_url = f"http://{host}:{port}"
    
    # First check health endpoint
    logger.info(f"Testing Phoenix health at {phoenix_url}/health")
    try:
        response = requests.get(f"{phoenix_url}/health", timeout=2)
        if response.status_code == 200:
            logger.info(f"✅ Phoenix health check successful: {response.text}")
            logger.info(f"📊 Access Phoenix UI at {phoenix_url}")
            return True
        else:
            logger.error(f"❌ Phoenix health check failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Phoenix health check failed: {e}")
        return False

if __name__ == "__main__":
    # Allow custom host/port from command line
    host = "localhost"
    port = 8084
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    logger.info(f"Testing Phoenix HTTP connection at {host}:{port}")
    result = test_phoenix_connection(host, port)
    
    if result:
        logger.info("✅ Phoenix is running and accessible!")
        sys.exit(0)
    else:
        logger.error("❌ Phoenix connection failed")
        sys.exit(1)
