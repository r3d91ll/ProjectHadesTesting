#!/usr/bin/env python3
"""
Phoenix UI Method

This script follows the exact pattern shown in the Phoenix UI to create a project
and send traces using OpenTelemetry.
"""

import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phoenix_ui_method")

# Set the environment variable as shown in the UI
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://0.0.0.0:8084"

# Using the exact code pattern from the UI
try:
    # Import Phoenix OpenTelemetry integration
    logger.info("Importing arize.phoenix.otel...")
    from arize.phoenix.otel import register
    
    # Register with Phoenix OpenTelemetry
    logger.info("Registering with Phoenix...")
    tracer_provider = register(
        project_name="pathrag-dataset-builder",
        endpoint="http://0.0.0.0:8084/v1/traces"
    )
    
    # Import trace API
    from opentelemetry import trace
    
    # Get a tracer
    tracer = trace.get_tracer("pathrag-dataset-builder")
    
    # Create a span
    logger.info("Creating and sending a test span...")
    with tracer.start_as_current_span("test-span") as span:
        # Add attributes to the span
        span.set_attribute("test", True)
        
        # Add an event
        span.add_event("Starting test")
        
        # Simulate some work
        logger.info("Simulating work...")
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
    logger.info("Starting Phoenix UI method test...")
