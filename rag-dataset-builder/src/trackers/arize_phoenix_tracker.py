#!/usr/bin/env python3
"""
Arize Phoenix Performance Tracker

This module provides an implementation of the PerformanceTracker interface
that sends telemetry data to Arize Phoenix.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import optional dependencies
try:
    import phoenix as px
    from phoenix.trace.datatypes import LLMSpan
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

from src.core.interfaces import PerformanceTracker
from src.core.base import BasePerformanceTracker
from src.core.plugin import register

logger = logging.getLogger("rag_dataset_builder.trackers.arize_phoenix")

@register('trackers', 'arize_phoenix')
class ArizePhoenixTracker(BasePerformanceTracker):
    """
    Performance tracker that sends telemetry data to Arize Phoenix.
    Requires the phoenix package to be installed.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        project_name: str = "rag-dataset-builder",
        auto_init: bool = True,
        server_port: int = 8084,
        log_file: Optional[str] = None
    ):
        """
        Initialize the Arize Phoenix tracker.
        
        Args:
            enabled: Whether tracking is enabled
            project_name: Name of the project in Phoenix
            auto_init: Whether to automatically initialize Phoenix
            server_port: Port for the Phoenix server
            log_file: Path to log file for Phoenix events
        """
        super().__init__(enabled=enabled)
        
        if not PHOENIX_AVAILABLE:
            logger.warning("Arize Phoenix is not available. Install with: pip install arize-phoenix")
            self.enabled = False
            return
        
        self.project_name = project_name
        self.session_trace = None
        
        if auto_init and self.enabled:
            try:
                # Initialize Phoenix
                px.launch_app(port=server_port)
                
                # Create trace session
                self.session_trace = px.Trace(name=project_name)
                logger.info(f"Initialized Arize Phoenix trace session for project: {project_name}")
                
                # Set up log file if specified
                if log_file:
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    px.Client.configure_persistence(log_file)
                    logger.info(f"Phoenix events will be logged to: {log_file}")
                    
            except Exception as e:
                logger.error(f"Error initializing Arize Phoenix: {e}")
                self.enabled = False
    
    def track_document_processing(
        self,
        document_id: str,
        document_path: str,
        processing_time: float,
        metadata: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Track document processing performance in Arize Phoenix.
        
        Args:
            document_id: Unique identifier for the document
            document_path: Path to the document
            processing_time: Time taken to process the document
            metadata: Document metadata
            success: Whether processing succeeded
            error: Error message if processing failed
        """
        if not self.enabled or not PHOENIX_AVAILABLE:
            return
        
        try:
            with self.session_trace.span(
                name="document_processing",
                span_type="processor",
                span_attributes={
                    "document_id": document_id,
                    "document_path": document_path,
                    "processing_time_ms": processing_time * 1000,
                    "success": success,
                    "file_type": metadata.get("file_type", "unknown"),
                    "file_size": metadata.get("file_size", 0),
                }
            ):
                if not success and error:
                    self.session_trace.add_event(
                        name="processing_error",
                        attributes={"error": error, "document_id": document_id}
                    )
                    
            # Add basic metrics
            super().track_document_processing(
                document_id, document_path, processing_time, metadata, success, error
            )
            
        except Exception as e:
            logger.error(f"Error tracking document processing in Phoenix: {e}")
    
    def track_embedding_generation(
        self,
        chunk_id: str,
        embedding_time: float,
        embedding_model: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Track embedding generation performance in Arize Phoenix.
        
        Args:
            chunk_id: Unique identifier for the chunk
            embedding_time: Time taken to generate embeddings
            embedding_model: Name of the embedding model
            success: Whether embedding generation succeeded
            error: Error message if embedding generation failed
        """
        if not self.enabled or not PHOENIX_AVAILABLE:
            return
        
        try:
            with self.session_trace.span(
                name="embedding_generation",
                span_type="embedder",
                span_attributes={
                    "chunk_id": chunk_id,
                    "embedding_time_ms": embedding_time * 1000,
                    "embedding_model": embedding_model,
                    "success": success
                }
            ):
                if not success and error:
                    self.session_trace.add_event(
                        name="embedding_error",
                        attributes={"error": error, "chunk_id": chunk_id}
                    )
                    
            # Add basic metrics
            super().track_embedding_generation(
                chunk_id, embedding_time, embedding_model, success, error
            )
            
        except Exception as e:
            logger.error(f"Error tracking embedding generation in Phoenix: {e}")
    
    def track_chunking(
        self,
        document_id: str,
        chunker_type: str,
        num_chunks: int,
        chunking_time: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Track chunking performance in Arize Phoenix.
        
        Args:
            document_id: Unique identifier for the document
            chunker_type: Type of chunker used
            num_chunks: Number of chunks created
            chunking_time: Time taken for chunking
            success: Whether chunking succeeded
            error: Error message if chunking failed
        """
        if not self.enabled or not PHOENIX_AVAILABLE:
            return
        
        try:
            with self.session_trace.span(
                name="chunking",
                span_type="chunker",
                span_attributes={
                    "document_id": document_id,
                    "chunker_type": chunker_type,
                    "num_chunks": num_chunks,
                    "chunking_time_ms": chunking_time * 1000,
                    "success": success
                }
            ):
                if not success and error:
                    self.session_trace.add_event(
                        name="chunking_error",
                        attributes={"error": error, "document_id": document_id}
                    )
                    
            # Add basic metrics
            super().track_chunking(
                document_id, chunker_type, num_chunks, chunking_time, success, error
            )
            
        except Exception as e:
            logger.error(f"Error tracking chunking in Phoenix: {e}")
    
    def track_output_generation(
        self,
        output_type: str,
        num_documents: int,
        num_chunks: int,
        output_time: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Track output generation performance in Arize Phoenix.
        
        Args:
            output_type: Type of output generated
            num_documents: Number of documents processed
            num_chunks: Number of chunks processed
            output_time: Time taken for output generation
            success: Whether output generation succeeded
            error: Error message if output generation failed
        """
        if not self.enabled or not PHOENIX_AVAILABLE:
            return
        
        try:
            with self.session_trace.span(
                name="output_generation",
                span_type="formatter",
                span_attributes={
                    "output_type": output_type,
                    "num_documents": num_documents,
                    "num_chunks": num_chunks,
                    "output_time_ms": output_time * 1000,
                    "success": success
                }
            ):
                if not success and error:
                    self.session_trace.add_event(
                        name="output_error",
                        attributes={"error": error, "output_type": output_type}
                    )
                    
            # Add basic metrics
            super().track_output_generation(
                output_type, num_documents, num_chunks, output_time, success, error
            )
            
        except Exception as e:
            logger.error(f"Error tracking output generation in Phoenix: {e}")
    
    def track_system_resources(self, resource_data: Dict[str, Any]) -> None:
        """
        Track system resource usage in Arize Phoenix.
        
        Args:
            resource_data: Dictionary of resource usage metrics
        """
        if not self.enabled or not PHOENIX_AVAILABLE:
            return
        
        try:
            with self.session_trace.span(
                name="system_resources",
                span_type="metrics",
                span_attributes=resource_data
            ):
                pass
                
        except Exception as e:
            logger.error(f"Error tracking system resources in Phoenix: {e}")
    
    def track_gpu_metrics(self, gpu_metrics: Dict[str, Any]) -> None:
        """
        Track GPU metrics in Arize Phoenix.
        
        Args:
            gpu_metrics: Dictionary of GPU metrics
        """
        if not self.enabled or not PHOENIX_AVAILABLE:
            return
        
        try:
            with self.session_trace.span(
                name="gpu_metrics",
                span_type="metrics",
                span_attributes=gpu_metrics
            ):
                pass
                
        except Exception as e:
            logger.error(f"Error tracking GPU metrics in Phoenix: {e}")
    
    def flush(self) -> None:
        """Flush performance tracking data to storage."""
        if not self.enabled or not PHOENIX_AVAILABLE:
            return
        
        try:
            # No explicit flush needed for Phoenix, but we can add session end event
            self.session_trace.add_event(
                name="session_completed",
                attributes={"timestamp": time.time()}
            )
            
            # Flush parent class data
            super().flush()
            
        except Exception as e:
            logger.error(f"Error flushing Phoenix tracking data: {e}")
