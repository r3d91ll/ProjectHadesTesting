#!/usr/bin/env python3
"""
Test script to connect to Phoenix using the OpenTelemetry OTLP protocol
Based on: https://docs.arize.com/phoenix/tracing/integrations-tracing/langchain
"""

import os
import sys
import time
import uuid
import logging
import datetime
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phoenix_otlp_test")

try:
    # Import OpenTelemetry components
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    import requests

    OTLP_AVAILABLE = True
    logger.info("✅ Successfully imported OpenTelemetry OTLP")
except ImportError as e:
    OTLP_AVAILABLE = False
    logger.error(f"❌ Failed to import OpenTelemetry: {e}")
    logger.error("Run: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")

def setup_phoenix_tracing(phoenix_host="localhost", phoenix_port=8084):
    """
    Set up Phoenix tracing using OpenTelemetry.
    
    Args:
        phoenix_host: Phoenix host
        phoenix_port: Phoenix port
    """
    # Construct the OTLP endpoint URL
    otlp_endpoint = f"http://{phoenix_host}:{phoenix_port}/v1/traces"
    
    # First check if Phoenix is available
    try:
        response = requests.get(f"http://{phoenix_host}:{phoenix_port}/health", timeout=2)
        if response.status_code == 200:
            logger.info(f"✅ Phoenix health check successful at http://{phoenix_host}:{phoenix_port}")
        else:
            logger.error(f"❌ Phoenix health check failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Phoenix health check failed: {e}")
        return None
    
    # Set up TracerProvider with resource attributes
    resource = Resource.create({"service.name": "pathrag-test"})
    tracer_provider = TracerProvider(resource=resource)
    
    # Set up OTLP exporter for Phoenix
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Set the global TracerProvider
    trace.set_tracer_provider(tracer_provider)
    
    # Create a tracer for our service
    tracer = trace.get_tracer("pathrag-phoenix-test")
    
    logger.info(f"✅ Phoenix tracing set up at {otlp_endpoint}")
    return tracer

def test_phoenix_connection():
    """
    Test Phoenix connection by creating a trace.
    """
    if not OTLP_AVAILABLE:
        logger.error("❌ OpenTelemetry not available - can't test Phoenix connection")
        return False
    
    # Set up Phoenix tracing
    tracer = setup_phoenix_tracing()
    if tracer is None:
        logger.error("❌ Failed to set up Phoenix tracing")
        return False
    
    # Create a trace
    trace_id = str(uuid.uuid4())
    
    try:
        with tracer.start_as_current_span(f"pathrag-query-{trace_id}", kind=trace.SpanKind.SERVER) as span:
            # Add attributes to the span
            span.set_attribute("app.trace_id", trace_id)
            span.set_attribute("app.model", "test-model")
            span.set_attribute("app.query", "This is a test query from phoenix_otlp_test.py")
            span.set_attribute("app.response", "This is a test response")
            span.set_attribute("app.latency_ms", 123.45)
            span.set_attribute("app.token_usage.prompt", 10)
            span.set_attribute("app.token_usage.completion", 5)
            span.set_attribute("app.token_usage.total", 15)
            
            # Add path information as attributes
            path_info = [{"node": "test", "score": 0.95}]
            span.set_attribute("app.path", str(path_info))
            
            # Simulate some processing time
            time.sleep(0.2)
            
            # Log an event
            span.add_event(
                name="query_processed",
                attributes={"timestamp": datetime.datetime.now().isoformat()}
            )
            
            logger.info(f"✅ Created trace {trace_id}")
            
        logger.info(f"✅ Trace {trace_id} completed")
        logger.info(f"📊 View trace at http://localhost:8084")
        
        # Sleep to ensure the batch processor sends the spans
        time.sleep(2)
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to create trace: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing Phoenix connection using OpenTelemetry OTLP...")
    
    if not OTLP_AVAILABLE:
        logger.error("❌ OpenTelemetry not available - can't test Phoenix connection")
        sys.exit(1)
    
    result = test_phoenix_connection()
    
    if result:
        logger.info("✅ Phoenix connection test passed!")
        logger.info("📊 Visit http://localhost:8084 to view your traces")
        sys.exit(0)
    else:
        logger.error("❌ Phoenix connection test failed")
        sys.exit(1)
