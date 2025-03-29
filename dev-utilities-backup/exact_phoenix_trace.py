#!/usr/bin/env python3
"""
Exact Phoenix Trace Sender

This script uses the exact pattern from the Phoenix UI documentation 
with no additional code or complexity.
"""

import os
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("exact_phoenix")

# Set environment variable exactly as shown in the UI
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:8084"

# Import from phoenix module directly (based on installed package structure)
logger.info("Importing phoenix.otel...")
try:
    # First try the documented import
    from arize.phoenix.otel import register
    logger.info("Using arize.phoenix.otel import path")
except ImportError:
    try:
        # Then try the direct phoenix module import
        from phoenix.otel import register
        logger.info("Using phoenix.otel import path")
    except ImportError:
        logger.error("Could not import from phoenix.otel or arize.phoenix.otel")
        # Check if the module exists at all
        logger.info("Checking available phoenix modules...")
        import pkgutil
        import importlib
        
        # Try to find any module containing 'phoenix'
        phoenix_modules = [m for m in pkgutil.iter_modules() if 'phoenix' in m.name]
        logger.info(f"Found phoenix-related modules: {phoenix_modules}")
        
        # If no modules found, exit
        if not phoenix_modules:
            logger.error("No phoenix modules found in the environment!")
            import sys
            sys.exit(1)

# Register exactly as shown in the UI
logger.info("Registering with Phoenix...")
endpoint = "http://localhost:8084/v1/traces"
logger.info(f"Using endpoint: {endpoint}")
tracer_provider = register(
    project_name="pathrag-dataset-builder",
    endpoint=endpoint
)

# Get tracer
from opentelemetry import trace
tracer = trace.get_tracer("pathrag-dataset-builder")

# Create a simple span
logger.info("Creating test span...")
with tracer.start_as_current_span("test-span") as span:
    # Add an attribute
    span.set_attribute("test", True)
    span.set_attribute("project", "pathrag-dataset-builder")
    
    # Add additional attributes that might help with debugging
    import socket
    span.set_attribute("hostname", socket.gethostname())
    span.set_attribute("timestamp", time.time())
    
    # Add an event
    span.add_event("Starting test")
    
    # Simulate work
    logger.info("Working...")
    time.sleep(1)
    
    # Add completion event
    span.add_event("Test completed")
    
    # Log span information
    import inspect
    logger.info(f"Span context: {span.get_span_context()}") 
    logger.info(f"Span methods: {[m for m in dir(span) if not m.startswith('_')]}")

logger.info("✅ Test span sent to Phoenix")
logger.info("Check Phoenix UI for project: pathrag-dataset-builder")

# Give the exporter time to send data
logger.info("Waiting for exporter to send data...")
time.sleep(5)

# Try to verify that the trace was sent
logger.info("Checking Phoenix accessibility...")
import requests
try:
    response = requests.get("http://localhost:8084/health")
    logger.info(f"Phoenix health check response: {response.status_code} - {response.text}")
except Exception as e:
    logger.error(f"Could not connect to Phoenix: {e}")
