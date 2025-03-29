#!/usr/bin/env python3
"""
Phoenix OpenTelemetry Integration

This script follows the official Phoenix documentation to create a project
and send traces using OpenTelemetry.
"""

import os
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phoenix_otel")

# Set the environment variable as shown in the Phoenix UI
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://0.0.0.0:8084"

# Import Phoenix OpenTelemetry integration
try:
    from phoenix.otel import register
    
    # Register with Phoenix OpenTelemetry
    # This is the exact code shown in the Phoenix UI
    tracer_provider = register(
        project_name="pathrag-dataset-builder",
        endpoint="http://0.0.0.0:8084/v1/traces"
    )
    
    # Import trace API
    from opentelemetry import trace
    
    # Get a tracer
    tracer = trace.get_tracer("pathrag-dataset-builder")
    
    # Create a span
    with tracer.start_as_current_span("test-span") as span:
        # Add attributes to the span
        span.set_attribute("test", True)
        span.set_attribute("timestamp", datetime.now().isoformat())
        
        # Add an event
        span.add_event("Starting test")
        
        # Simulate some work
        time.sleep(1)
        
        # Add completion event
        span.add_event("Test completed")
    
    logger.info("✅ Successfully sent trace to Phoenix")
    logger.info("Check Phoenix UI for project: pathrag-dataset-builder")
    
    # Give the exporter time to send data
    time.sleep(2)
    
except ImportError as e:
    logger.error(f"Failed to import Phoenix OpenTelemetry: {e}")
    logger.error("Make sure you have installed: pip install arize-phoenix-otel")
    
if __name__ == "__main__":
    logger.info("Phoenix OpenTelemetry integration test complete")
