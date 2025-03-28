#!/usr/bin/env python3
"""
Simple HTTP client for the Arize Phoenix Docker container
Bypasses complex import issues by using direct HTTP calls
"""

import os
import sys
import time
import json
import uuid
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phoenix_http_client")

class PhoenixHTTPClient:
    """Simple HTTP client for Arize Phoenix."""
    
    def __init__(self, host="localhost", port=8084):
        """Initialize the Phoenix HTTP client."""
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.verify_connection()
    
    def verify_connection(self) -> bool:
        """Verify connection to Phoenix server."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ Successfully connected to Phoenix at {self.base_url}")
                return True
            else:
                logger.error(f"❌ Failed to connect to Phoenix: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Phoenix: {e}")
            return False
    
    def log_trace(self, trace_data: Dict[str, Any]) -> Optional[str]:
        """
        Log a trace to Phoenix using the HTTP API.
        
        Args:
            trace_data: Dictionary with trace data
            
        Returns:
            Trace ID if successful, None otherwise
        """
        # Ensure we have a trace ID
        trace_id = trace_data.get("id", str(uuid.uuid4()))
        trace_data["id"] = trace_id
        
        # Ensure we have required fields
        if "name" not in trace_data:
            trace_data["name"] = "HTTP Client Trace"
        
        if "model" not in trace_data:
            trace_data["model"] = "unknown-model"
        
        # Add timestamp if not present
        if "timestamp" not in trace_data:
            trace_data["timestamp"] = datetime.now().isoformat()
        
        try:
            # Send trace to Phoenix via HTTP
            response = requests.post(
                f"{self.base_url}/api/traces",
                json=trace_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200 or response.status_code == 201:
                logger.info(f"✅ Successfully logged trace {trace_id} to Phoenix")
                logger.info(f"📊 View trace at {self.base_url}/span/{trace_id}")
                return trace_id
            else:
                logger.error(f"❌ Failed to log trace: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to log trace: {e}")
            return None

def test_phoenix_client():
    """Test the Phoenix HTTP client."""
    client = PhoenixHTTPClient()
    
    # Create a test trace
    trace_id = str(uuid.uuid4())
    trace_data = {
        "id": trace_id,
        "name": "Test Query from phoenix_http_client.py",
        "model": "test-model",
        "inputs": {
            "query": "This is a test query from the HTTP client"
        },
        "outputs": {
            "response": "This is a test response"
        },
        "metadata": {
            "test_timestamp": str(datetime.now()),
            "test_source": "phoenix_http_client.py"
        }
    }
    
    # Log the trace
    result = client.log_trace(trace_data)
    return result is not None

if __name__ == "__main__":
    # Allow custom host/port from command line
    host = "localhost"
    port = 8084
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    logger.info(f"Testing Phoenix HTTP client at {host}:{port}")
    client = PhoenixHTTPClient(host, port)
    
    # Run the test
    if test_phoenix_client():
        logger.info("✅ Phoenix HTTP client test passed!")
        logger.info(f"📊 Visit http://{host}:{port} to view your traces in Phoenix")
        sys.exit(0)
    else:
        logger.error("❌ Phoenix HTTP client test failed")
        sys.exit(1)
