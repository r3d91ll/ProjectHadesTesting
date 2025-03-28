#!/usr/bin/env python3
"""
Phoenix Direct Integration

This script uses the exact code pattern shown in the Phoenix UI to create a project
and send traces using the arize-phoenix-otel package.
"""

import os
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phoenix_direct")

# First, let's verify we have the right package
try:
    import pip
    logger.info("Installing arize-phoenix-otel...")
    pip.main(['install', 'arize-phoenix-otel'])
    logger.info("✅ Installation complete")
except Exception as e:
    logger.error(f"Error during package installation: {e}")

# Set the environment variable as shown in the Phoenix UI
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://0.0.0.0:8084"

# The exact code from the Phoenix UI
try:
    from phoenix.otel import register
    
    logger.info("Registering with Phoenix using OpenTelemetry...")
    tracer_provider = register(
        project_name="pathrag-dataset-builder",
        endpoint="http://0.0.0.0:8084/v1/traces"
    )
    
    # Import trace API
    from opentelemetry import trace
    
    # Get a tracer
    tracer = trace.get_tracer("pathrag-dataset-builder-tracer")
    
    # Create a span
    logger.info("Creating and sending a test span...")
    with tracer.start_as_current_span("test-span") as span:
        # Add attributes to the span
        span.set_attribute("test", True)
        span.set_attribute("timestamp", datetime.now().isoformat())
        span.set_attribute("environment", "development")
        
        # Add an event
        span.add_event("Starting test operation")
        
        # Simulate some work
        time.sleep(1)
        
        # Add completion event
        span.add_event("Test operation completed")
    
    logger.info("✅ Successfully sent trace to Phoenix")
    logger.info("Check Phoenix UI for the 'pathrag-dataset-builder' project")
    
    # Give the exporter time to send data
    time.sleep(2)
    
except ImportError as e:
    logger.error(f"Failed to import Phoenix OpenTelemetry: {e}")
    logger.error("Make sure you have the correct package installed")
    
if __name__ == "__main__":
    logger.info("Phoenix direct integration test complete")
