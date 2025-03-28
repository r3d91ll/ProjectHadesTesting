"""
PathRAG Database Builder Arize Phoenix Integration

This module integrates the PathRAG database builder with Arize Phoenix
for telemetry and performance tracking during the dataset building process.
"""

import os
import time
import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
import gc

# Arize Phoenix imports for telemetry
try:
    import pandas as pd
    from arize.phoenix.session import Session
    from arize.phoenix.trace import trace, response_converter
    from arize.phoenix.trace.trace import LLMTrace
    from arize.phoenix.trace import model_call_converter
except ImportError:
    raise ImportError(
        "Failed to import Arize Phoenix. Install it with: pip install arize-phoenix"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PathRAGDBArizeAdapter:
    """Adapter for PathRAG Database Builder with Arize Phoenix integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PathRAG Database Builder adapter with Arize Phoenix integration.
        
        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config
        self.phoenix_host = config.get("phoenix_host", "localhost")
        self.phoenix_port = config.get("phoenix_port", 8080)
        self.phoenix_url = f"http://{self.phoenix_host}:{self.phoenix_port}"
        self.track_performance = config.get("track_performance", True)
        
        # Initialize Phoenix session if tracking is enabled
        if self.track_performance:
            try:
                self.phoenix_session = Session(url=self.phoenix_url)
                logger.info(f"Connected to Arize Phoenix at {self.phoenix_url}")
                
                # Check Phoenix connection
                response = requests.get(f"{self.phoenix_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("Arize Phoenix health check successful")
                else:
                    logger.warning(f"Arize Phoenix health check failed: {response.status_code}")
                    self.track_performance = False
            except Exception as e:
                logger.error(f"Failed to connect to Arize Phoenix: {e}")
                self.track_performance = False
    
    def _log_to_phoenix(self, trace_data: Dict[str, Any]) -> None:
        """
        Log trace data to Arize Phoenix.
        
        Args:
            trace_data: Dictionary containing trace data
        """
        if not self.track_performance:
            return
        
        try:
            # Create and save trace to Phoenix
            trace_obj = LLMTrace(
                id=trace_data.get("id", str(uuid.uuid4())),
                name=trace_data.get("name", "PathRAG DB Operation"),
                model=trace_data.get("model", "sentence-transformers"),
                input=trace_data.get("input", ""),
                output=trace_data.get("output", ""),
                prompt_tokens=trace_data.get("input_size", 0),
                completion_tokens=trace_data.get("output_size", 0),
                latency_ms=trace_data.get("latency_ms", 0),
                metadata=trace_data.get("metadata", {}),
                spans=trace_data.get("spans", [])
            )
            
            self.phoenix_session.log_trace(trace_obj)
            logger.info(f"Logged trace {trace_obj.id} to Arize Phoenix")
        except Exception as e:
            logger.error(f"Failed to log to Arize Phoenix: {e}")
    
    def track_embedding_generation(self, texts: List[str], model_name: str, 
                                embedding_dim: int, batch_size: int) -> Dict[str, Any]:
        """
        Track the embedding generation process.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the embedding model
            embedding_dim: Dimensionality of the embeddings
            batch_size: Batch size for embedding generation
            
        Returns:
            Dictionary with the trace ID
        """
        if not self.track_performance:
            return {"trace_id": None}
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Record memory usage before operation
        memory_before = self._get_memory_usage()
        
        # Prepare trace data
        trace_data = {
            "id": trace_id,
            "name": "Embedding Generation",
            "model": model_name,
            "input": f"Batch of {len(texts)} texts",
            "output": f"Embeddings of dim {embedding_dim}",
            "input_size": sum(len(text.split()) for text in texts),
            "output_size": len(texts) * embedding_dim,
            "latency_ms": 0,  # Will be updated later
            "metadata": {
                "operation_type": "embedding_generation",
                "num_texts": len(texts),
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "mean_text_length": sum(len(text.split()) for text in texts) / len(texts),
                "memory_before_mb": memory_before,
                "memory_after_mb": 0,  # Will be updated later
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Return the trace data and ID for later completion
        return {
            "trace_id": trace_id,
            "trace_data": trace_data,
            "start_time": start_time
        }
    
    def complete_embedding_tracking(self, tracking_info: Dict[str, Any]) -> None:
        """
        Complete the embedding generation tracking.
        
        Args:
            tracking_info: Information from track_embedding_generation
        """
        if not self.track_performance or tracking_info.get("trace_id") is None:
            return
        
        # Calculate latency
        end_time = time.time()
        latency_ms = int((end_time - tracking_info["start_time"]) * 1000)
        
        # Record memory usage after operation
        memory_after = self._get_memory_usage()
        
        # Update trace data
        trace_data = tracking_info["trace_data"]
        trace_data["latency_ms"] = latency_ms
        trace_data["metadata"]["memory_after_mb"] = memory_after
        trace_data["metadata"]["memory_delta_mb"] = memory_after - trace_data["metadata"]["memory_before_mb"]
        
        # Log to Phoenix
        self._log_to_phoenix(trace_data)
    
    def track_document_processing(self, doc_path: str, operation: str = "processing") -> Dict[str, Any]:
        """
        Track document processing operations.
        
        Args:
            doc_path: Path to the document being processed
            operation: Type of operation (processing, chunking, etc.)
            
        Returns:
            Dictionary with the trace ID
        """
        if not self.track_performance:
            return {"trace_id": None}
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Get document metadata
        doc_size = os.path.getsize(doc_path) if os.path.exists(doc_path) else 0
        doc_type = os.path.splitext(doc_path)[1]
        
        # Record memory usage before operation
        memory_before = self._get_memory_usage()
        
        # Prepare trace data
        trace_data = {
            "id": trace_id,
            "name": f"Document {operation.capitalize()}",
            "model": "pathrag-processor",
            "input": doc_path,
            "output": f"Processed {doc_path}",
            "input_size": doc_size,
            "output_size": 0,  # Will be updated later
            "latency_ms": 0,  # Will be updated later
            "metadata": {
                "operation_type": f"document_{operation}",
                "doc_path": doc_path,
                "doc_size_bytes": doc_size,
                "doc_type": doc_type,
                "memory_before_mb": memory_before,
                "memory_after_mb": 0,  # Will be updated later
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Return the trace data and ID for later completion
        return {
            "trace_id": trace_id,
            "trace_data": trace_data,
            "start_time": start_time
        }
    
    def complete_document_tracking(self, tracking_info: Dict[str, Any], 
                                 num_chunks: int, output_size: int) -> None:
        """
        Complete the document processing tracking.
        
        Args:
            tracking_info: Information from track_document_processing
            num_chunks: Number of chunks generated
            output_size: Size of the output in bytes
        """
        if not self.track_performance or tracking_info.get("trace_id") is None:
            return
        
        # Calculate latency
        end_time = time.time()
        latency_ms = int((end_time - tracking_info["start_time"]) * 1000)
        
        # Record memory usage after operation
        memory_after = self._get_memory_usage()
        
        # Update trace data
        trace_data = tracking_info["trace_data"]
        trace_data["latency_ms"] = latency_ms
        trace_data["output_size"] = output_size
        trace_data["metadata"]["num_chunks"] = num_chunks
        trace_data["metadata"]["memory_after_mb"] = memory_after
        trace_data["metadata"]["memory_delta_mb"] = memory_after - trace_data["metadata"]["memory_before_mb"]
        
        # Log to Phoenix
        self._log_to_phoenix(trace_data)
    
    def track_graph_update(self, num_nodes: int, num_edges: int, 
                         operation: str = "update") -> Dict[str, Any]:
        """
        Track knowledge graph update operations.
        
        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges in the graph
            operation: Type of operation (update, save, etc.)
            
        Returns:
            Dictionary with the trace ID
        """
        if not self.track_performance:
            return {"trace_id": None}
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Record memory usage before operation
        memory_before = self._get_memory_usage()
        
        # Prepare trace data
        trace_data = {
            "id": trace_id,
            "name": f"Graph {operation.capitalize()}",
            "model": "pathrag-graph",
            "input": f"Graph with {num_nodes} nodes and {num_edges} edges",
            "output": f"Updated graph",
            "input_size": num_nodes + num_edges,
            "output_size": 0,  # Will be updated later
            "latency_ms": 0,  # Will be updated later
            "metadata": {
                "operation_type": f"graph_{operation}",
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "memory_before_mb": memory_before,
                "memory_after_mb": 0,  # Will be updated later
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Return the trace data and ID for later completion
        return {
            "trace_id": trace_id,
            "trace_data": trace_data,
            "start_time": start_time
        }
    
    def complete_graph_tracking(self, tracking_info: Dict[str, Any], 
                              new_nodes: int, new_edges: int) -> None:
        """
        Complete the graph update tracking.
        
        Args:
            tracking_info: Information from track_graph_update
            new_nodes: Number of new nodes added
            new_edges: Number of new edges added
        """
        if not self.track_performance or tracking_info.get("trace_id") is None:
            return
        
        # Calculate latency
        end_time = time.time()
        latency_ms = int((end_time - tracking_info["start_time"]) * 1000)
        
        # Record memory usage after operation
        memory_after = self._get_memory_usage()
        
        # Update trace data
        trace_data = tracking_info["trace_data"]
        trace_data["latency_ms"] = latency_ms
        trace_data["output_size"] = new_nodes + new_edges
        trace_data["metadata"]["new_nodes"] = new_nodes
        trace_data["metadata"]["new_edges"] = new_edges
        trace_data["metadata"]["memory_after_mb"] = memory_after
        trace_data["metadata"]["memory_delta_mb"] = memory_after - trace_data["metadata"]["memory_before_mb"]
        
        # Log to Phoenix
        self._log_to_phoenix(trace_data)
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        # Force garbage collection
        gc.collect()
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb
        except ImportError:
            # Fallback method if psutil is not available
            if hasattr(os, 'getrusage'):
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                return usage.ru_maxrss / 1024  # Convert kilobytes to megabytes
            return 0  # Cannot determine memory usage
