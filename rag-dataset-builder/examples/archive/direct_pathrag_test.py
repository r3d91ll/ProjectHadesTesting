#!/usr/bin/env python3
"""
Direct PathRAG Test

This script demonstrates how to use the PathRAG implementation directly,
without relying on the plugin system. This is useful for testing the
implementation while the plugin system is still being configured.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import required modules
import time
from src.core.config_loader import get_configuration
from src.implementations.pathrag import PathRAG
from src.core.base import NetworkXStorageBackend

# Explicitly check for phoenix and force availability flag
try:
    import phoenix as px
    import uuid
    import datetime
    import json
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# Now import the tracker
from src.trackers.arize_phoenix_tracker import ArizePhoenixTracker
# Override the flag in the module to ensure it recognizes phoenix as available
import src.trackers.arize_phoenix_tracker
src.trackers.arize_phoenix_tracker.PHOENIX_AVAILABLE = PHOENIX_AVAILABLE

# Create a custom TrackerAdapter to handle API changes in newer Phoenix versions
class PhoenixTrackerAdapter:
    def __init__(self, project_name):
        self.project_name = project_name
        self.span_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now().isoformat()
        # Store events to be flushed later
        self.events = []
        
    def log_event(self, event_type, data):
        # Use the OpenInference format that Phoenix 8.x supports
        metadata = {
            "span_id": self.span_id,
            "event_type": event_type,
            **data
        }
        # Store the event for later logging
        self.events.append((event_type, metadata))
        
    def flush_events(self):
        # Log all stored events to Phoenix
        for event_type, metadata in self.events:
            self._log_to_phoenix(event_type, metadata)
        # Clear the events after flushing
        self.events = []
        
    def _log_to_phoenix(self, event_type, metadata):
        # Log directly to Phoenix using the current API for version 8.x
        try:
            # Convert metadata to string for logging
            metadata_str = json.dumps(metadata)
            # Use px.log which is available in Phoenix 8.x
            px.log(
                messages=[{"role": "system", "content": f"Event: {event_type}"}, 
                          {"role": "assistant", "content": f"Metadata: {metadata_str}"}],
                name=self.project_name,
                span_id=self.span_id,
                span_kind="RAG",
                span_attributes=metadata
            )
        except Exception as e:
            print(f"Error logging to Phoenix: {str(e)}")
        
    # Implement methods that are called from the ArizePhoenixTracker
    def start_span(self, **kwargs):
        # Just create a new event with the span data
        span_data = {"operation": "span_start", **kwargs}
        self.log_event("span_start", span_data)
        return self  # Return self to act as the span
    
    def end_span(self, **kwargs):
        # Log the end of a span
        self.log_event("span_end", kwargs)
    
    def add_event(self, name, attributes=None):
        # Add an event to the current span
        event_data = {"name": name}
        if attributes:
            event_data.update(attributes)
        self.log_event(name, event_data)
    
    # Make this object act as a span for the tracker
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Patch the tracker initialization
original_init = src.trackers.arize_phoenix_tracker.ArizePhoenixTracker.__init__

def patched_init(self, enabled=True, project_name="rag-dataset-builder", auto_init=True, server_port=8084, log_file=None):
    super(src.trackers.arize_phoenix_tracker.ArizePhoenixTracker, self).__init__(enabled=enabled)
    
    if not src.trackers.arize_phoenix_tracker.PHOENIX_AVAILABLE:
        src.trackers.arize_phoenix_tracker.logger.warning("Arize Phoenix is not available. Install with: pip install arize-phoenix")
        self.enabled = False
        return
    
    self.project_name = project_name
    
    if auto_init and self.enabled:
        try:
            # Initialize Phoenix with the current API
            px.launch_app(port=server_port)
            
            # Create trace session using our adapter
            self.session_trace = PhoenixTrackerAdapter(project_name)
            src.trackers.arize_phoenix_tracker.logger.info(f"Initialized Arize Phoenix trace session for project: {project_name}")
            
        except Exception as e:
            src.trackers.arize_phoenix_tracker.logger.error(f"Error initializing Arize Phoenix: {str(e)}")
            self.enabled = False

# Patch the tracker methods
src.trackers.arize_phoenix_tracker.ArizePhoenixTracker.__init__ = patched_init

# Also patch the flush method to use our adapter
original_flush = src.trackers.arize_phoenix_tracker.ArizePhoenixTracker.flush

def patched_flush(self):
    if self.enabled and hasattr(self, 'session_trace') and self.session_trace:
        if hasattr(self.session_trace, 'flush_events'):
            self.session_trace.flush_events()
        # Call the original flush for any remaining functionality
        original_flush(self)
            
src.trackers.arize_phoenix_tracker.ArizePhoenixTracker.flush = patched_flush

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct_pathrag_test")

def main():
    """Run a direct test of the PathRAG implementation."""
    logger.info("Testing direct PathRAG implementation...")
    
    # Load configuration
    custom_config_path = Path(__file__).parent / "pathrag_custom.yaml"
    if not custom_config_path.exists():
        logger.info("Creating custom configuration...")
        with open(custom_config_path, "w") as f:
            f.write("""
# Custom configuration for PathRAG example
input_dir: "./data/test_input"
output_dir: "./data/test_output"

monitoring:
  arize_phoenix:
    enabled: true
    project_name: "pathrag-direct-test"
    server_port: 8084

retrieval_systems:
  pathrag:
    max_paths: 3
    similarity_threshold: 0.75
    storage_backend: "networkx"
    add_similarity_edges: true
            """)
    
    # Get configuration
    config = get_configuration(user_config=str(custom_config_path))
    
    # Initialize Arize Phoenix tracker (optional)
    logger.info("Initializing tracker...")
    tracker = None
    if config.get("monitoring", {}).get("arize_phoenix", {}).get("enabled", False):
        tracker = ArizePhoenixTracker(
            project_name=config["monitoring"]["arize_phoenix"].get("project_name", "pathrag-direct-test"),
            server_port=config["monitoring"]["arize_phoenix"].get("server_port", 8084)
        )
    
    # Initialize PathRAG directly
    logger.info("Initializing PathRAG directly...")
    pathrag = PathRAG(tracker=tracker)
    
    # Create a storage backend
    storage_backend = NetworkXStorageBackend()
    
    # Configure PathRAG
    pathrag_config = config["retrieval_systems"]["pathrag"]
    pathrag.config = pathrag_config
    pathrag.storage_backend = storage_backend
    
    # Test basic functionality
    logger.info("Testing basic functionality...")
    
    # Add some test nodes to the storage backend
    storage_backend.add_item("doc1", "document", title="Test Document 1", content="This is a test document about AI.")
    storage_backend.add_item("chunk1", "chunk", content="This is a test chunk about AI.")
    storage_backend.add_item("chunk2", "chunk", content="This is another test chunk about machine learning.")
    
    # Add relationships
    storage_backend.add_relationship("doc1", "chunk1", "contains")
    storage_backend.add_relationship("doc1", "chunk2", "contains")
    
    # Query the storage backend
    logger.info("Querying storage backend...")
    items = storage_backend.query({"item_type": "chunk"}, limit=10)
    logger.info(f"Found {len(items)} chunks:")
    for item in items:
        # Print the structure of the item to debug
        logger.info(f"  - Item structure: {item}")
        # Handle the item data based on its actual structure
        item_id = item.get('id', str(item))
        if isinstance(item, dict):
            content = item.get('content', '')
            if 'properties' in item:
                content = item['properties'].get('content', '')
            logger.info(f"  - {item_id}: {content}")
        else:
            logger.info(f"  - {item_id}: (No content available)")
            
    # Test the Arize Phoenix integration by sending telemetry data
    if tracker and tracker.enabled:
        logger.info("Sending telemetry data to Arize Phoenix...")
        
        # Track document processing
        tracker.track_document_processing(
            document_id="doc1",
            document_path="/path/to/test/document.txt",
            processing_time=0.5,
            metadata={"format": "text", "size": 1024},
            success=True
        )
        
        # Track chunk embedding generation
        tracker.track_embedding_generation(
            chunk_id="chunk1",
            embedding_time=0.2,
            embedding_model="test-embeddings",
            success=True
        )
        
        # Track a simulated query
        logger.info("Simulating a query to test RAG flow tracking...")
        query = "What is artificial intelligence?"
        
        # Simulate a RAG flow
        start_time = time.time()
        time.sleep(0.2)  # Simulate retrieval time
        retrieved_chunks = ["This is a test chunk about AI.", "This is another test chunk about machine learning."]
        retrieval_time = time.time() - start_time
        
        # Generate a simulated response
        start_time = time.time()
        time.sleep(0.3)  # Simulate generation time
        response = "Artificial Intelligence (AI) is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."
        generation_time = time.time() - start_time
        
        # Log to Phoenix
        if hasattr(tracker.session_trace, 'log_event'):
            tracker.session_trace.log_event("rag_query", {
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "response": response,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + generation_time
            })
            logger.info("Telemetry data sent to Arize Phoenix")
            
        # Flush data to ensure it's saved
        tracker.flush()
        logger.info("Data flushed to Arize Phoenix")
    
    logger.info("Direct PathRAG test completed successfully!")

if __name__ == "__main__":
    main()
