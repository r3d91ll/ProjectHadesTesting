#!/usr/bin/env python3
"""
Simple Phoenix LangChain Test
A minimal test script for connecting PathRAG to Phoenix using OpenTelemetry
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phoenix_test")

def test_phoenix_connection():
    """
    Test Phoenix connection using OpenTelemetry.
    """
    try:
        # Import OpenTelemetry components
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        import requests
        
        # First check if Phoenix is actually running
        try:
            response = requests.get("http://localhost:8084/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Phoenix health check successful")
            else:
                logger.error(f"❌ Phoenix health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Phoenix health check failed: {e}")
            return False
            
        # Set up a resource (this identifies your service)
        resource = Resource.create({"service.name": "pathrag-test"})
        
        # Create a tracer provider with the resource
        tracer_provider = TracerProvider(resource=resource)
        
        # Create an OTLP exporter to send traces to Phoenix
        otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:8084/v1/traces")
        
        # Add the exporter to the tracer provider
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set the tracer provider as the global default
        trace.set_tracer_provider(tracer_provider)
        
        # Create a tracer
        tracer = trace.get_tracer("pathrag.test")
        
        # Create a span
        with tracer.start_as_current_span("test-query") as span:
            # Set some attributes on the span
            span.set_attribute("model", "test-model")
            span.set_attribute("query", "This is a test query from PathRAG")
            span.set_attribute("response", "This is a test response from the model")
            span.set_attribute("latency_ms", 123.45)
            
            # Add path information as attributes
            span.set_attribute("path.node_count", 2)
            span.set_attribute("path.node.0.text", "First document chunk")
            span.set_attribute("path.node.0.score", 0.95)
            span.set_attribute("path.node.1.text", "Second document chunk")
            span.set_attribute("path.node.1.score", 0.85)
            
            # Simulate some work
            time.sleep(0.5)
            
            # Add an event to the span
            span.add_event("processed_query")
            
        # Wait for batched spans to be exported
        time.sleep(2)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error during Phoenix test: {e}")
        return False

def try_langchain_instrumentation():
    """
    Try to instrument LangChain for Phoenix telemetry.
    This is a separate function to isolate any import errors.
    """
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        
        # Initialize and instrument LangChain
        LangChainInstrumentor().instrument()
        logger.info("✅ Successfully instrumented LangChain")
        
        # Try to import some LangChain components
        from langchain.schema.document import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Create a simple document
        docs = [Document(page_content="This is a test document for PathRAG")]
        
        # Split the document
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        split_docs = splitter.split_documents(docs)
        
        logger.info(f"✅ Created {len(split_docs)} document chunks")
        return True
    
    except ImportError as e:
        logger.warning(f"⚠️ Could not import LangChain instrumentation: {e}")
        logger.warning("This is expected if you haven't installed all required packages")
        return False
    except Exception as e:
        logger.error(f"❌ Error in LangChain instrumentation: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing Phoenix connection...")
    
    # Try to instrument LangChain (optional)
    try_langchain_instrumentation()
    
    # Test basic OpenTelemetry connection to Phoenix
    success = test_phoenix_connection()
    
    if success:
        logger.info("✅ Phoenix connection test passed!")
        logger.info("📊 Visit http://localhost:8084 to view your traces")
    else:
        logger.error("❌ Phoenix connection test failed")
        sys.exit(1)
