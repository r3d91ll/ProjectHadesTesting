#!/usr/bin/env python3
"""
Simple Phoenix Trace Sender

This utility script sends a simple trace to Phoenix using the OpenTelemetry API,
which is the recommended way to integrate with Phoenix.
"""

import os
import sys
import time
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_phoenix")

PROJECT_NAME = "pathrag-dataset-builder"

try:
    # Import OpenTelemetry
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.semconv.resource import ResourceAttributes
    
    # Import Phoenix-specific utilities if available
    try:
        from arize.phoenix.otel import register as phoenix_register
        PHOENIX_OTEL_AVAILABLE = True
    except ImportError:
        PHOENIX_OTEL_AVAILABLE = False
        logger.warning("Phoenix OTEL integration not available - will use generic OpenTelemetry")
    
    OTEL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import OpenTelemetry: {e}")
    logger.error("Make sure opentelemetry packages are installed")
    OTEL_AVAILABLE = False
    sys.exit(1)

def set_up_opentelemetry():
    """Set up OpenTelemetry with Phoenix exporter"""
    
    # Phoenix endpoint
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:8084")
    if not endpoint.endswith("/v1/traces"):
        # Make sure we're pointing to the OTLP endpoint
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        endpoint = f"{endpoint}/v1/traces"
    
    logger.info(f"Using Phoenix endpoint: {endpoint}")
    
    # Create resource attributes 
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: PROJECT_NAME,
        "project.name": PROJECT_NAME,
        "phoenix.project": PROJECT_NAME
    })
    
    # Set up tracer provider with resource
    provider = TracerProvider(resource=resource)
    
    # Create OTLP exporter and processor
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    return trace.get_tracer("simple_phoenix_tracer")

def send_simple_trace():
    """Send a simple trace to Phoenix"""
    
    # Set up OpenTelemetry if Phoenix-specific registration isn't available
    if PHOENIX_OTEL_AVAILABLE:
        logger.info("Using Phoenix OTEL integration")
        phoenix_register(service_name=PROJECT_NAME)
        tracer = trace.get_tracer("simple_phoenix_tracer")
    else:
        logger.info("Using generic OpenTelemetry setup")
        tracer = set_up_opentelemetry()
    
    # Generate unique IDs for this test
    test_id = str(uuid.uuid4())
    
    # Create and record a trace
    with tracer.start_as_current_span(
        name="test_span",
        attributes={
            "test.id": test_id,
            "test.timestamp": datetime.now().isoformat(),
            "test.type": "simple_test",
            "project.name": PROJECT_NAME
        }
    ) as span:
        # Add some events and attributes to the span
        span.add_event("Starting test operation")
        
        # Simulate some work
        time.sleep(0.5)
        
        # Add more data to the span
        span.set_attribute("test.status", "success")
        span.set_attribute("test.result", "Trace successfully created")
        
        span.add_event("Completing test operation")
    
    logger.info(f"✅ Trace sent with ID: {test_id}")
    logger.info(f"Check Phoenix UI for project: {PROJECT_NAME}")
    
    # Give the exporter a moment to send the data
    time.sleep(2)
    
    return True

if __name__ == "__main__":
    logger.info("Starting simple Phoenix trace sender...")
    send_simple_trace()
