#!/usr/bin/env python3
"""
Direct Phoenix Test

This is a minimal script that focuses solely on creating a new project in Phoenix
using the exact packages and setup available in the environment.
"""

import os
import sys
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phoenix_direct_test")

# Print Python path for debugging
logger.debug(f"Python path: {sys.path}")

# List all installed phoenix packages
try:
    import pkg_resources
    phoenix_packages = [p.project_name for p in pkg_resources.working_set if "phoenix" in p.project_name.lower()]
    logger.info(f"Installed Phoenix packages: {phoenix_packages}")
except:
    logger.warning("Could not check installed packages")

# Set required environment variables
os.environ["PHOENIX_PROJECT_NAME"] = "pathrag-dataset-builder"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:8084"

# Try all possible import paths
try:
    logger.info("Trying import from phoenix.otel...")
    from phoenix.otel import register as phoenix_register
    IMPORT_SOURCE = "phoenix.otel"
except ImportError as e:
    logger.info(f"Import from phoenix.otel failed: {e}")
    try:
        logger.info("Trying import from arize.phoenix.otel...")
        from arize.phoenix.otel import register as phoenix_register
        IMPORT_SOURCE = "arize.phoenix.otel"
    except ImportError as e:
        logger.error(f"Import from arize.phoenix.otel failed: {e}")
        try:
            logger.info("Trying from opentelemetry direct integration...")
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.semconv.resource import ResourceAttributes
            
            # Create exporter
            otlp_endpoint = "http://localhost:8084/v1/traces"
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            
            # Create resource with project name
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: "pathrag-dataset-builder",
                "project.name": "pathrag-dataset-builder"
            })
            
            # Set up tracer provider
            tracer_provider = TracerProvider(resource=resource)
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(tracer_provider)
            
            logger.info("Using direct OpenTelemetry setup")
            IMPORT_SOURCE = "opentelemetry.direct"
        except ImportError as e:
            logger.error(f"Direct OpenTelemetry setup failed: {e}")
            logger.error("Could not find any working Phoenix integration")
            sys.exit(1)

# Create a tracer and send a test span
try:
    logger.info(f"Creating a test span with {IMPORT_SOURCE} integration")
    
    if IMPORT_SOURCE == "opentelemetry.direct":
        # We've already set up the tracer provider
        tracer = trace.get_tracer("pathrag-dataset-builder")
    else:
        # Use the register function from whichever module succeeded
        logger.info("Registering with Phoenix...")
        tracer_provider = phoenix_register(
            project_name="pathrag-dataset-builder",
            endpoint="http://localhost:8084/v1/traces"
        )
        tracer = trace.get_tracer("pathrag-dataset-builder")
    
    # Create and record a span
    with tracer.start_as_current_span("direct-phoenix-test") as span:
        span.set_attribute("test_type", "direct")
        span.set_attribute("timestamp", time.time())
        span.set_attribute("project", "pathrag-dataset-builder")
        
        # Add an event
        span.add_event("Starting test")
        
        # Simulate work
        logger.info("Working in test span...")
        time.sleep(1)
        
        # Add completion event
        span.add_event("Test completed")
    
    logger.info("✅ Test span sent to Phoenix")
    logger.info("Check Phoenix UI for project: pathrag-dataset-builder")
    
    # Give the exporter time to send data
    time.sleep(2)
    
except Exception as e:
    logger.error(f"Error creating test span: {e}")
    import traceback
    logger.error(traceback.format_exc())

logger.info("Phoenix test complete")
