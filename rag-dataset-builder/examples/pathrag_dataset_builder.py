#!/usr/bin/env python3
"""
PathRAG Dataset Builder with Arize Phoenix Integration

This script builds a dataset for PathRAG while monitoring performance with Arize Phoenix.
"""

import os
import sys
import time
import logging
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import OpenTelemetry and Phoenix
try:
    from phoenix.otel import register
    from opentelemetry import trace
    import torch  # For GPU monitoring if available
    PHOENIX_AVAILABLE = True
except ImportError as e:
    PHOENIX_AVAILABLE = False
    print(f"Phoenix not available: {e}. Install with: pip install arize-phoenix")

from src.core.config_loader import get_configuration
from src.core.plugin import discover_plugins, create_retrieval_system
from src.core.base import NetworkXStorageBackend
from src.implementations.pathrag import PathRAG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pathrag_dataset_builder")

class OTelSpanWrapper:
    """A wrapper class for OpenTelemetry spans to simplify usage"""
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
    """Set up Phoenix tracing for PathRAG with OpenTelemetry."""
    if not PHOENIX_AVAILABLE:
        logger.warning("Phoenix not available, skipping setup.")
        return None
        
    try:
        # Register with Phoenix OpenTelemetry
        logger.info("Setting up Phoenix OpenTelemetry tracer...")
        tracer_provider = register(
            project_name="pathrag-dataset-builder",
            endpoint="http://0.0.0.0:8084/v1/traces"
        )
        # Set environment variable for Phoenix collector endpoint
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://0.0.0.0:8084"
        
        # Get a tracer for our application
        tracer = trace.get_tracer("pathrag.dataset_builder")
        logger.info("Phoenix OpenTelemetry tracer initialized successfully")
        
        return tracer
    except Exception as e:
        logger.error(f"Error setting up Phoenix OpenTelemetry: {str(e)}")
        return None

def setup_environment():
    """Set up necessary environment for running PathRAG."""
    # Create data directories if they don't exist
    os.makedirs("./data/input", exist_ok=True)
    os.makedirs("./data/output", exist_ok=True)
    os.makedirs("./data/chunks", exist_ok=True)
    os.makedirs("./data/embeddings", exist_ok=True)
    
    # Set environment variables if needed from .env file
    env_file = Path.home() / "ML-Lab" / "New-HADES" / ".env"
    if env_file.exists():
        logger.info(f"Loading environment variables from {env_file}")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
                    logger.debug(f"Set {key}={value}")

def monitor_gpu_metrics(span_wrapper):
    """Add GPU monitoring metrics to the span if available."""
    try:
        if torch.cuda.is_available():
            # Get number of GPUs
            gpu_count = torch.cuda.device_count()
            span_wrapper.set_attribute("gpu_count", gpu_count)
            
            for i in range(gpu_count):
                # Get device properties
                props = torch.cuda.get_device_properties(i)
                span_wrapper.set_attribute(f"gpu_{i}_name", props.name)
                span_wrapper.set_attribute(f"gpu_{i}_total_memory_gb", props.total_memory / 1e9)
                
                # Get current memory usage
                memory_allocated = torch.cuda.memory_allocated(i) / 1e9  # Convert to GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1e9  # Convert to GB
                span_wrapper.set_attribute(f"gpu_{i}_memory_allocated_gb", memory_allocated)
                span_wrapper.set_attribute(f"gpu_{i}_memory_reserved_gb", memory_reserved)
                
                # Get current device utilization 
                # Note: torch doesn't provide utilization directly, would need nvidia-smi for that
                
            logger.info(f"Added metrics for {gpu_count} GPUs")
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.warning(f"Could not monitor GPU metrics: {e}")
        span_wrapper.set_attribute("gpu_monitoring_error", str(e))

def build_pathrag_dataset(config: Dict[str, Any], tracer: Optional[Any] = None):
    """Build a dataset for PathRAG with performance tracking."""
    start_time = time.time()
    
    # If we have a tracer, create a parent span for the entire process
    dataset_span = None
    span_wrapper = None
    
    if tracer:
        dataset_span = tracer.start_span("build_pathrag_dataset")
        span_wrapper = OTelSpanWrapper(dataset_span)
        span_wrapper.set_attribute("start_time", datetime.datetime.now().isoformat())
    
    try:
        # Set up the storage backend
        logger.info("Setting up storage backend...")
        if span_wrapper:
            span_wrapper.add_event("setup_storage_backend_start")
            
        storage_backend = NetworkXStorageBackend()
        
        if span_wrapper:
            span_wrapper.add_event("setup_storage_backend_complete")
        
        # Initialize PathRAG
        logger.info("Initializing PathRAG...")
        if span_wrapper:
            span_wrapper.add_event("init_pathrag_start")
            
        # Get all configuration settings needed for PathRAG
        pathrag_config = config.get("retrieval_systems", {}).get("pathrag", {})
        
        # Set up PathRAG attributes directly as class attributes
        PathRAG.config = pathrag_config
        PathRAG.storage_backend = storage_backend
        
        # Initialize a PathRAG instance
        pathrag = PathRAG()
        
        if span_wrapper:
            span_wrapper.add_event("init_pathrag_complete")
            span_wrapper.set_attribute("pathrag_config", json.dumps(pathrag_config))
        
        # Process documents in the input directory
        input_dir = config.get("directories", {}).get("input", "./data/input")
        logger.info(f"Processing documents from {input_dir}...")
        
        if span_wrapper:
            span_wrapper.add_event("document_processing_start")
        
        # Get all text files in the input directory
        doc_files = list(Path(input_dir).glob("*.txt"))
        processed_docs = 0
        
        # Process each document
        for doc_path in doc_files:
            try:
                # Start document processing span
                doc_span = None
                doc_wrapper = None
                if tracer:
                    doc_span = tracer.start_span(f"process_document_{doc_path.name}")
                    doc_wrapper = OTelSpanWrapper(doc_span)
                    doc_wrapper.set_attribute("document_path", str(doc_path))
                
                logger.info(f"Processing document: {doc_path}")
                
                # Read the document
                with open(doc_path, 'r') as f:
                    content = f.read()
                
                # Add document to the storage backend directly
                doc_id = f"doc-{processed_docs}"
                storage_backend.add_item(doc_id, "document", title=doc_path.name, content=content)
                
                # Create chunks from the document (simple splitting for demo)
                chunks = content.split('\n\n')  # Split by paragraph
                chunk_ids = []
                
                # Add each chunk to storage backend
                for i, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():
                        continue
                    chunk_id = f"{doc_id}-chunk-{i}"
                    storage_backend.add_item(chunk_id, "chunk", content=chunk_text.strip())
                    storage_backend.add_relationship(doc_id, chunk_id, "contains")
                    chunk_ids.append(chunk_id)
                    
                processed_docs += 1
                
                if doc_wrapper:
                    doc_wrapper.set_attribute("document_id", doc_id)
                    doc_wrapper.set_attribute("document_size_bytes", len(content))
                    # Add GPU metrics
                    monitor_gpu_metrics(doc_wrapper)
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
                if doc_wrapper:
                    doc_wrapper.set_attribute("error", str(e))
            finally:
                # Close the document span
                if doc_span:
                    doc_span.end()
        
        if span_wrapper:
            span_wrapper.add_event("document_processing_complete")
            span_wrapper.set_attribute("processed_documents_count", processed_docs)
        
        # Run a test query to verify the dataset
        logger.info("Running test queries...")
        if span_wrapper:
            span_wrapper.add_event("test_queries_start")
        
        test_queries = [
            "What is artificial intelligence?",
            "Explain natural language processing.",
            "How does PathRAG improve upon standard RAG approaches?"
        ]
        
        for query in test_queries:
            try:
                # Start query span
                query_span = None
                query_wrapper = None
                if tracer:
                    query_span = tracer.start_span(f"test_query")
                    query_wrapper = OTelSpanWrapper(query_span)
                    query_wrapper.set_attribute("query", query)
                
                logger.info(f"Running test query: {query}")
                
                # Retrieve relevant documents/chunks
                if query_wrapper:
                    query_wrapper.add_event("retrieval_start")
                
                retrieval_start = time.time()
                # Query the storage backend directly instead of using pathrag.retrieve()
                results = storage_backend.query({"item_type": "chunk"}, limit=5)
                retrieval_time = time.time() - retrieval_start
                
                if query_wrapper:
                    query_wrapper.add_event("retrieval_complete")
                    query_wrapper.set_attribute("retrieval_time_seconds", retrieval_time)
                    query_wrapper.set_attribute("retrieved_items_count", len(results))
                    
                    # Log top retrieved items
                    for i, item in enumerate(results[:3]):
                        if hasattr(item, 'metadata') and hasattr(item, 'content'):
                            query_wrapper.set_attribute(f"top_result_{i}_id", item.id if hasattr(item, 'id') else "unknown")
                            query_wrapper.set_attribute(f"top_result_{i}_content_preview", item.content[:100] if item.content else "")
                    
                    # Add GPU metrics
                    monitor_gpu_metrics(query_wrapper)
                
                logger.info(f"Query '{query}' retrieved {len(results)} results in {retrieval_time:.3f} seconds")
            except Exception as e:
                logger.error(f"Error running query '{query}': {e}")
                if query_wrapper:
                    query_wrapper.set_attribute("error", str(e))
            finally:
                # Close the query span
                if query_span:
                    query_span.end()
        
        if span_wrapper:
            span_wrapper.add_event("test_queries_complete")
            span_wrapper.set_attribute("test_queries_count", len(test_queries))
        
        # Calculate and log final stats
        total_time = time.time() - start_time
        logger.info(f"Dataset built successfully in {total_time:.3f} seconds")
        logger.info(f"Processed {processed_docs} documents")
        
        if span_wrapper:
            span_wrapper.set_attribute("total_build_time_seconds", total_time)
            span_wrapper.set_attribute("end_time", datetime.datetime.now().isoformat())
            # Add final GPU metrics
            monitor_gpu_metrics(span_wrapper)
        
        return True
    except Exception as e:
        logger.error(f"Error building PathRAG dataset: {e}")
        if span_wrapper:
            span_wrapper.set_attribute("error", str(e))
        return False
    finally:
        # Close the dataset span
        if dataset_span:
            dataset_span.end()

def main():
    """Run the PathRAG dataset builder with Phoenix integration."""
    # Set up environment
    setup_environment()
    
    # Discover plugins
    discover_plugins()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = get_configuration()
    
    # Set up Phoenix OpenTelemetry integration
    tracer = setup_phoenix()
    
    # Build the dataset with performance tracking
    success = build_pathrag_dataset(config, tracer)
    
    if success:
        logger.info("PathRAG dataset built successfully!")
        logger.info("To view the Phoenix dashboard: http://localhost:8084")
    else:
        logger.error("Failed to build PathRAG dataset. Check logs for details.")

if __name__ == "__main__":
    main()
