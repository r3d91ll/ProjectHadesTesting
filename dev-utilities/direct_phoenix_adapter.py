#!/usr/bin/env python3
"""
Direct Phoenix Adapter

This module provides a simplified adapter for connecting to the Phoenix Docker container
running on port 8084. It bypasses the complex import structure and uses direct HTTP calls
for sending telemetry data.
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
logger = logging.getLogger("direct_phoenix")

class DirectPhoenixAdapter:
    """
    Direct adapter for Phoenix Docker container.
    Bypasses the complex Python SDK and uses HTTP calls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Phoenix adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get("model_name", "unknown-model")
        self.track_performance = config.get("track_performance", True)
        
        # Phoenix configuration
        self.phoenix_host = config.get("phoenix_host", "localhost")
        self.phoenix_port = config.get("phoenix_port", 8084)  # Docker container port
        self.phoenix_url = f"http://{self.phoenix_host}:{self.phoenix_port}"
        
        # Check if Phoenix is available
        if self.track_performance:
            try:
                response = requests.get(f"{self.phoenix_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(f"Phoenix health check successful at {self.phoenix_url}")
                    self.phoenix_available = True
                else:
                    logger.warning(f"Phoenix health check failed: {response.status_code}")
                    logger.warning("Telemetry will not be recorded for this session")
                    self.phoenix_available = False
            except Exception as e:
                logger.warning(f"Phoenix not available at {self.phoenix_url}: {e}")
                logger.warning("Telemetry will not be recorded for this session")
                self.phoenix_available = False
        else:
            self.phoenix_available = False
    
    def log_telemetry(self, 
                    trace_id: str,
                    query: str, 
                    response: str, 
                    path: Optional[List[Dict[str, Any]]] = None,
                    latency_ms: Optional[float] = None,
                    token_usage: Optional[Dict[str, int]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log telemetry data to Phoenix.
        
        Args:
            trace_id: Unique identifier for this trace
            query: User query
            response: System response
            path: Path information (for PathRAG)
            latency_ms: Latency in milliseconds
            token_usage: Token usage information
            metadata: Additional metadata
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not self.track_performance or not self.phoenix_available:
            logger.debug("Skipping telemetry logging - Phoenix not available or tracking disabled")
            return False
        
        # Create trace data
        trace_data = {
            "id": trace_id,
            "name": "PathRAG Query",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "query": query
            },
            "outputs": {
                "response": response
            }
        }
        
        # Add path information if available
        if path:
            trace_data["inputs"]["path"] = path
        
        # Add latency if available
        if latency_ms is not None:
            trace_data["latency_ms"] = latency_ms
        
        # Add token usage if available
        if token_usage:
            trace_data["token_usage"] = token_usage
        
        # Add metadata if available
        if metadata:
            trace_data["metadata"] = metadata
        
        # Log the trace
        try:
            response = requests.post(
                f"{self.phoenix_url}/api/v1/traces",
                json=trace_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200 or response.status_code == 201:
                logger.info(f"Logged trace {trace_id} to Phoenix")
                return True
            else:
                logger.warning(f"Failed to log trace to Phoenix: {response.status_code}")
                logger.debug(f"Response: {response.text}")
                return False
        except Exception as e:
            logger.warning(f"Failed to log trace to Phoenix: {e}")
            return False

def test_phoenix_connection():
    """
    Test the Phoenix connection by sending a test trace.
    """
    config = {
        "phoenix_host": "localhost",
        "phoenix_port": 8084,
        "track_performance": True,
        "model_name": "test-model"
    }
    
    adapter = DirectPhoenixAdapter(config)
    
    # Generate a unique trace ID
    trace_id = str(uuid.uuid4())
    
    # Log a test trace
    success = adapter.log_telemetry(
        trace_id=trace_id,
        query="This is a test query from direct_phoenix_adapter.py",
        response="This is a test response",
        path=[
            {"node": "test", "score": 0.95}
        ],
        latency_ms=123.45,
        token_usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        },
        metadata={
            "test": True,
            "timestamp": str(datetime.now())
        }
    )
    
    if success:
        logger.info(f"✅ Successfully sent test trace to Phoenix")
        logger.info(f"📊 View trace at {adapter.phoenix_url}/span/{trace_id}")
        return True
    else:
        logger.error(f"❌ Failed to send test trace to Phoenix")
        return False

if __name__ == "__main__":
    logger.info("Testing direct Phoenix adapter...")
    result = test_phoenix_connection()
    
    if result:
        logger.info("✅ Phoenix connection test passed!")
        sys.exit(0)
    else:
        logger.error("❌ Phoenix connection test failed")
        sys.exit(1)
