#!/usr/bin/env python3
"""
Phoenix PathRAG Test Script
This script demonstrates how to properly integrate PathRAG with Arize Phoenix
using the OpenTelemetry approach and LangChain instrumentation.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phoenix_pathrag_test")

# Set the Phoenix port
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://0.0.0.0:8084"

# Import Phoenix packages
try:
    # Use the official Phoenix OpenTelemetry integration
    from phoenix.otel import register
    from opentelemetry import trace
    import uuid
    import datetime
    import json
    PHOENIX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Arize Phoenix not available: {str(e)}. Install with: pip install arize-phoenix")
    PHOENIX_AVAILABLE = False

# Import RAG Dataset Builder modules
from src.core.config_loader import get_configuration
from src.implementations.pathrag import PathRAG
from src.core.base import NetworkXStorageBackend

class OTelSpanWrapper:
    """A wrapper class to provide a simpler interface for OpenTelemetry spans"""
    def __init__(self, span):
        self.span = span
        
    def add_event(self, name, attributes=None):
        """Add an event to the span"""
        self.span.add_event(name, attributes)
        
    def set_attribute(self, key, value):
        """Set an attribute on the span"""
        self.span.set_attribute(key, value)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.span.end()

def setup_phoenix():
    """Set up Phoenix tracing for the PathRAG project using OpenTelemetry."""
    if not PHOENIX_AVAILABLE:
        logger.warning("Phoenix not available, skipping setup.")
        return None
        
    try:
        # Register with Phoenix OpenTelemetry
        logger.info("Setting up Phoenix OpenTelemetry tracer...")
        tracer_provider = register(
            project_name="pathrag-test",
            endpoint="http://0.0.0.0:8084/v1/traces"
        )
        
        # Get a tracer for our application
        tracer = trace.get_tracer("pathrag.example")
        logger.info("Phoenix OpenTelemetry tracer initialized successfully")
        
        return tracer
    except Exception as e:
        logger.error(f"Error setting up Phoenix OpenTelemetry: {str(e)}")
        return None

def main():
    """Main test function."""
    logger.info("Starting Phoenix PathRAG test...")
    
    # Set up Phoenix
    session = setup_phoenix()
    if not session and PHOENIX_AVAILABLE:
        logger.error("Failed to initialize Phoenix session.")
        return
    
    # Load configuration
    logger.info("Loading configuration...")
    config = get_configuration()
    
    # Update configuration for testing
    logger.info("Updating configuration for monitoring...")
    if isinstance(config, dict):
        # Update the dictionary directly
        if "monitoring" not in config:
            config["monitoring"] = {}
        
        config["monitoring"]["enabled"] = True
        
        if "arize_phoenix" not in config["monitoring"]:
            config["monitoring"]["arize_phoenix"] = {}
            
        config["monitoring"]["arize_phoenix"]["enabled"] = True
        config["monitoring"]["arize_phoenix"]["project_name"] = "pathrag-test"
        config["monitoring"]["arize_phoenix"]["server_port"] = 8084
    
    # Create a configuration directory for testing
    custom_config_dir = Path(__file__).parent / "config.d"
    custom_config_dir.mkdir(exist_ok=True)
    
    # Set up simple storage backend
    logger.info("Setting up storage backend...")
    storage_backend = NetworkXStorageBackend()
    
    # Initialize PathRAG
    logger.info("Initializing PathRAG...")
    # Set up PathRAG attributes directly as in the direct_pathrag_test.py
    pathrag_config = config.get("retrieval_systems", {}).get("pathrag", {})
    PathRAG.config = pathrag_config
    PathRAG.storage_backend = storage_backend
    
    # Add some test data
    logger.info("Adding test data...")
    document_content = "This is a test document about artificial intelligence and machine learning."
    
    # In a real application, this would be a document ingestion
    doc_id = "test-doc-1"
    storage_backend.add_item(
        doc_id, 
        "document", 
        title="Test Document", 
        content=document_content
    )
    
    # Add some chunks manually (in a real app, the chunker would do this)
    chunk1_id = "chunk-1"
    chunk2_id = "chunk-2"
    
    storage_backend.add_item(
        chunk1_id, 
        "chunk", 
        content="This is a test chunk about artificial intelligence."
    )
    
    storage_backend.add_item(
        chunk2_id, 
        "chunk", 
        content="This is a test chunk about machine learning."
    )
    
    # Add relationships
    storage_backend.add_relationship(doc_id, chunk1_id, "contains")
    storage_backend.add_relationship(doc_id, chunk2_id, "contains")
    
    # Track the RAG process with OpenTelemetry spans
    query = "What is artificial intelligence?"
    logger.info(f"Simulating query: {query}")
    
    # Use OpenTelemetry tracing if available
    if session:
        # Create a parent span for the entire RAG flow
        with session.start_as_current_span("rag_query") as rag_span:
            # Wrap the span to simplify usage
            span_wrapper = OTelSpanWrapper(rag_span)
            
            # Add query information to the span
            span_wrapper.set_attribute("query", query)
            span_wrapper.set_attribute("timestamp_start", datetime.datetime.now().isoformat())
            
            # Simulate retrieval phase
            logger.info("Simulating retrieval phase...")
            span_wrapper.add_event("retrieval_start", {
                "query": query,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Get chunks from the storage backend
            retrieved_chunks = storage_backend.query({"item_type": "chunk"}, limit=2)
            retrieved_content = [
                chunk.get('content', '') if isinstance(chunk, dict) else str(chunk)
                for chunk in retrieved_chunks
            ]
            
            # Log retrieved chunks
            span_wrapper.add_event("retrieval_complete", {
                "num_chunks": len(retrieved_chunks),
                "chunks": json.dumps(retrieved_content),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Simulate response generation
            logger.info("Simulating response generation...")
            span_wrapper.add_event("generation_start", {
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Wait a bit to simulate generation time
            time.sleep(0.5)
            
            # Generate response
            response = "Artificial Intelligence (AI) is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."
            
            # Log generation completion
            span_wrapper.add_event("generation_complete", {
                "response": response,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Set overall results
            span_wrapper.set_attribute("response", response)
            span_wrapper.set_attribute("num_chunks_retrieved", len(retrieved_chunks))
            span_wrapper.set_attribute("timestamp_end", datetime.datetime.now().isoformat())
            
            # Output the results
            logger.info(f"Query: {query}")
            logger.info(f"Response: {response}")
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
    else:
        logger.info("Phoenix tracer not available, running without tracing...")
        # Run the same code without tracing
        query = "What is artificial intelligence?"
        retrieved_chunks = storage_backend.query({"item_type": "chunk"}, limit=2)
        logger.info(f"Query: {query}")
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        for chunk in retrieved_chunks:
            content = chunk.get('content', '') if isinstance(chunk, dict) else str(chunk)
            logger.info(f"  - Chunk: {content}")
    
    logger.info("Phoenix PathRAG test completed successfully!")

if __name__ == "__main__":
    main()
