"""
Arize Phoenix Integration for RAG Dataset Builder

This module provides utilities to track performance metrics of the RAG Dataset Builder
using Arize Phoenix for visualization and analysis.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Define fallback classes for when Phoenix client is not available
class RecordType:
    DOCUMENT = "document"
    DOCUMENT_PROCESSING = "document_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    CHUNKING = "chunking"
    QUERY = "query"
    RESPONSE = "response"
    EVALUATION = "evaluation"

class Metric:
    """Placeholder for Arize Phoenix Metric class when client is not available"""
    def __init__(self, value):
        self.value = value

class Document:
    """Placeholder for Arize Phoenix Document class when client is not available"""
    def __init__(self, text=None):
        self.text = text
        
class Record:
    """Placeholder for Arize Phoenix Record class when client is not available"""
    def __init__(self, id=None, record_type=None, timestamp=None, features=None, document=None, metadata=None, metrics=None):
        self.id = id
        self.record_type = record_type
        self.timestamp = timestamp
        self.features = features or {}
        self.document = document
        self.metadata = metadata or {}
        self.metrics = metrics or {}

# Try to import the Phoenix OpenTelemetry integration first (recommended approach)
try:
    from phoenix.otel import register  # Import from phoenix.otel instead of arize.phoenix.otel
    from opentelemetry import trace
    PHOENIX_OTEL_AVAILABLE = True
    logging.info("✅ Phoenix OpenTelemetry integration successfully imported")
except ImportError:
    PHOENIX_OTEL_AVAILABLE = False
    logging.warning("⚠️ Phoenix OpenTelemetry integration not available. Run 'pip install arize-phoenix-otel' to enable.")

# Fall back to direct client if OpenTelemetry is not available
try:
    from arize.phoenix.client import Client
    from arize.phoenix.types import (
        Embedding,
        Metric,
        Record as PhoenixRecord,
        RecordType as PhoenixRecordType,
        Document,
        Query,
        Response,
        RetrievalResult,
        DistanceType
    )
    # Use the Phoenix types instead of our placeholder
    Record = PhoenixRecord
    RecordType = PhoenixRecordType
    # No need to reassign Metric and Document as they're already imported
    PHOENIX_CLIENT_AVAILABLE = True
except ImportError:
    PHOENIX_CLIENT_AVAILABLE = False
    # Keep using our placeholder classes
    if not PHOENIX_OTEL_AVAILABLE:
        logging.warning("Arize Phoenix not installed. Run 'pip install arize-phoenix' to enable performance tracking.")

PHOENIX_AVAILABLE = PHOENIX_OTEL_AVAILABLE or PHOENIX_CLIENT_AVAILABLE

class RAGDatasetBuilderArizeAdapter:
    """
    Adapter for tracking RAG Dataset Builder performance metrics with Arize Phoenix.
    """
    
    def __init__(
        self, 
        project_name: str = "pathrag-dataset-builder",
        enabled: bool = True,
        phoenix_url: str = "http://localhost:8084",
        batch_size: int = 100
    ):
        """
        Initialize the Arize Phoenix adapter.
        
        Args:
            project_name: The name of the project in Arize Phoenix
            enabled: Whether tracking is enabled
            phoenix_url: URL of the Arize Phoenix instance
            batch_size: Number of records to batch before sending to Phoenix
        """
        self.enabled = enabled and PHOENIX_AVAILABLE
        self.project_name = project_name
        self.batch_size = batch_size
        self.pending_records = []
        self.traced_operations = {}
        
        # Initialize OpenTelemetry tracer if available (recommended approach)
        if self.enabled and PHOENIX_OTEL_AVAILABLE:
            # Extract endpoint path for OpenTelemetry
            if not phoenix_url.endswith("/v1/traces"):
                endpoint = f"{phoenix_url}/v1/traces"
            else:
                endpoint = phoenix_url
            
            logging.info(f"🔍 Setting up Phoenix OpenTelemetry with project name: {self.project_name}")
            logging.info(f"🔍 Using Phoenix endpoint: {endpoint}")
                
            # Register the Phoenix tracer provider
            self.tracer_provider = register(
                project_name=self.project_name,
                endpoint=endpoint
            )
            self.tracer = trace.get_tracer("rag-dataset-builder-tracer")
            logging.info(f"✅ Arize Phoenix OpenTelemetry integration enabled. Project: {self.project_name}")
            logging.info(f"Dashboard available at {phoenix_url}")
            
            # Send a test trace to verify integration
            with self.tracer.start_as_current_span("phoenix-integration-test") as span:
                span.set_attribute("test", True)
                span.set_attribute("project", self.project_name)
                span.set_attribute("component", "rag-dataset-builder")
                span.set_attribute("timestamp", datetime.now().isoformat())
                span.add_event("Integration test")
                
                # Add a child span to ensure we have a proper trace hierarchy
                with self.tracer.start_as_current_span("verify-phoenix-connection") as child_span:
                    child_span.set_attribute("test", True)
                    child_span.set_attribute("component", "connection-test")
                    child_span.add_event("Connection verified")
                    
                logging.info(f"🔍 Sent test trace to Phoenix project '{self.project_name}'")
        
        # Fall back to direct client if OpenTelemetry is not available
        elif self.enabled and PHOENIX_CLIENT_AVAILABLE:
            self.client = Client(url=phoenix_url)
            logging.warning("⚠️ Using Phoenix Client API instead of recommended OpenTelemetry integration.")
            logging.info(f"Arize Phoenix Client integration enabled. Dashboard available at {phoenix_url}")
        else:
            self.client = None
            if enabled and not PHOENIX_AVAILABLE:
                logging.warning(
                    "Performance tracking is enabled but Arize Phoenix is not installed. "
                    "Run 'pip install arize-phoenix' to enable tracking."
                )
    
    def _create_trace_span(self, operation_name, attributes=None):
        """Create and return a new trace span for the given operation"""
        if not self.enabled or not PHOENIX_OTEL_AVAILABLE:
            return None
            
        try:
            # Create a span for this operation - use start_as_current_span instead of start_span
            # to properly establish the trace context
            span = self.tracer.start_as_current_span(operation_name)
            
            # Add common attributes
            span.set_attribute("project", self.project_name)
            span.set_attribute("timestamp", datetime.now().isoformat())
            
            # Add operation-specific attributes
            if attributes:
                for key, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(key, value)
                    elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                        # Only add numeric lists (like embeddings)
                        continue  # Skip large vectors
                    elif value is not None:
                        span.set_attribute(key, str(value))
            
            # Log the trace creation
            logging.info(f"📊 Created Phoenix trace: {operation_name}")
            return span
        except Exception as e:
            logging.error(f"❌ Failed to create trace span: {e}")
            return None
            
    def _end_trace_span(self, span, success=True, error=None):
        """End a trace span with success/error information"""
        if span is None:
            return
            
        try:
            # Add completion status
            span.set_attribute("success", success)
            span.set_attribute("end_timestamp", datetime.now().isoformat())
            
            if error:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(error))
                span.record_exception(Exception(error))
                span.add_event("Error", {"message": str(error)})
            else:
                span.add_event("Completed", {"success": success})
                
            # With context manager, the span is automatically ended
            # If not using context manager, we need to end it manually
            if not span.__class__.__name__ == "_Span":
                span.end()
                
            logging.info(f"📊 Completed Phoenix trace{' with error' if error else ''}")
        except Exception as e:
            logging.error(f"❌ Failed to end trace span: {e}")
    
    def track_document_processing(
        self,
        document_id: str,
        document_path: str,
        document_type: str,
        processing_time: float,
        document_size: int,
        metadata: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ):
        """
        Track metrics for document processing.
        
        Args:
            document_id: Unique identifier for the document
            document_path: Path to the document
            document_type: Type of document (pdf, text, code)
            processing_time: Time taken to process the document in seconds
            document_size: Size of the document in bytes
            metadata: Additional document metadata
            success: Whether processing succeeded
            error: Error message if processing failed
        """
        if not self.enabled:
            return

        # For OpenTelemetry approach, create and record spans
        if PHOENIX_OTEL_AVAILABLE and hasattr(self, 'tracer'):
            # Create a span for document processing
            doc_name = os.path.basename(document_path)
            with self.tracer.start_as_current_span(f"process-document-{doc_name}") as span:
                # Add document attributes
                span.set_attribute("document.id", document_id)
                span.set_attribute("document.path", document_path)
                span.set_attribute("document.type", document_type)
                span.set_attribute("document.size_bytes", document_size)
                span.set_attribute("processing.time_seconds", processing_time)
                span.set_attribute("success", success)
                
                # Add all metadata as attributes
                for key, value in metadata.items():
                    # Convert to string to ensure compatibility
                    span.set_attribute(f"metadata.{key}", str(value))
                
                if error:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error)
                    span.record_exception(Exception(error))
                
                span.add_event("Document processing complete", {
                    "success": success,
                    "processing_time": processing_time
                })
                
                logging.info(f"📊 Recorded document processing trace for {doc_name} in project '{self.project_name}'")
            return
            
        # For Client API approach, use records
        metrics = {
            "processing_time_seconds": Metric(processing_time),
            "document_size_bytes": Metric(document_size),
            "success": Metric(1.0 if success else 0.0),
        }
        
        if error:
            metrics["error"] = Metric(1.0)
            
        # Create document record
        record = Record(
            id=document_id,
            record_type=RecordType.DOCUMENT,
            document=Document(text=os.path.basename(document_path)),
            metadata={
                "document_path": document_path,
                "document_type": document_type,
                "timestamp": datetime.now().isoformat(),
                "error_message": error or "",
                **metadata
            },
            metrics=metrics
        )
        
        self._add_record(record)
    
    def track_chunking(
        self,
        document_id: str,
        chunker_type: str,
        num_chunks: int,
        chunk_sizes: List[int],
        chunking_time: float,
        avg_chunk_size: float,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Track metrics for text chunking.
        
        Args:
            document_id: ID of the document being chunked
            chunker_type: Type of chunker used
            num_chunks: Number of chunks created
            chunk_sizes: List of chunk sizes
            chunking_time: Time taken for chunking in seconds
            avg_chunk_size: Average chunk size
            success: Whether chunking succeeded
            error: Error message if chunking failed
        """
        if not self.enabled:
            return
            
        # For OpenTelemetry approach, create and record spans
        if PHOENIX_OTEL_AVAILABLE and hasattr(self, 'tracer'):
            with self.tracer.start_as_current_span(f"chunk-document-{document_id}") as span:
                # Add chunking attributes
                span.set_attribute("document.id", document_id)
                span.set_attribute("chunker.type", chunker_type)
                span.set_attribute("chunks.count", num_chunks)
                span.set_attribute("chunks.avg_size", avg_chunk_size)
                span.set_attribute("chunking.time_seconds", chunking_time)
                span.set_attribute("success", success)
                
                # Add chunk size histogram (up to 10 samples to avoid too many attributes)
                for i, size in enumerate(chunk_sizes[:10]):
                    span.set_attribute(f"chunks.size_{i}", size)
                
                if error:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error)
                    span.record_exception(Exception(error))
                
                span.add_event("Document chunking complete", {
                    "success": success,
                    "chunking_time": chunking_time,
                    "num_chunks": num_chunks
                })
                
                logging.info(f"📊 Recorded chunking trace for document {document_id} in project '{self.project_name}'")
            return
            
        # For Client API approach, use records
        metrics = {
            "num_chunks": Metric(num_chunks),
            "avg_chunk_size": Metric(avg_chunk_size),
            "chunking_time_seconds": Metric(chunking_time),
            "success": Metric(1.0 if success else 0.0),
        }
        
        if error:
            metrics["error"] = Metric(1.0)
            
        # Create record
        record = Record(
            id=f"{document_id}_chunking",
            record_type=RecordType.FEATURE_STORE,
            metadata={
                "document_id": document_id,
                "chunker_type": chunker_type,
                "chunk_sizes": json.dumps(chunk_sizes),
                "timestamp": datetime.now().isoformat(),
                "error_message": error or "",
            },
            metrics=metrics
        )
        
        self._add_record(record)
    
    def track_embedding_generation(
        self,
        chunk_id: str,
        document_id: str,
        embedder_type: str,
        embedding_model: str,
        embedding_dimensions: int,
        embedding_time: float,
        embedding: List[float],
        success: bool,
        error: Optional[str] = None
    ):
        """
        Track metrics for embedding generation.
        
        Args:
            chunk_id: ID of the chunk being embedded
            document_id: ID of the parent document
            embedder_type: Type of embedder used
            embedding_model: Name of the embedding model
            embedding_dimensions: Number of dimensions in the embedding
            embedding_time: Time taken for embedding generation in seconds
            embedding: The actual embedding vector
            success: Whether embedding generation succeeded
            error: Error message if embedding generation failed
        """
        if not self.enabled:
            return
            
        # For OpenTelemetry approach, create and record spans
        if PHOENIX_OTEL_AVAILABLE and hasattr(self, 'tracer'):
            with self.tracer.start_as_current_span(f"generate-embedding-{chunk_id}") as span:
                # Add embedding attributes
                span.set_attribute("chunk.id", chunk_id)
                span.set_attribute("document.id", document_id)
                span.set_attribute("embedder.type", embedder_type)
                span.set_attribute("embedding.model", embedding_model)
                span.set_attribute("embedding.dimensions", embedding_dimensions)
                span.set_attribute("embedding.time_seconds", embedding_time)
                span.set_attribute("success", success)
                
                # Add a few embedding values as a sample (first 5)
                for i, val in enumerate(embedding[:5]):
                    span.set_attribute(f"embedding.sample_{i}", val)
                
                if error:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error)
                    span.record_exception(Exception(error))
                
                span.add_event("Embedding generation complete", {
                    "success": success,
                    "embedding_time": embedding_time,
                    "embedding_dimensions": embedding_dimensions
                })
                
                logging.info(f"📊 Recorded embedding trace for chunk {chunk_id} in project '{self.project_name}'")
            return
            
        # For Client API approach, use records
        metrics = {
            "embedding_dimensions": Metric(embedding_dimensions),
            "embedding_time_seconds": Metric(embedding_time),
            "success": Metric(1.0 if success else 0.0),
        }
        
        if error:
            metrics["error"] = Metric(1.0)
            
        # Create record
        record = Record(
            id=chunk_id,
            record_type=RecordType.FEATURE_STORE,
            embeddings={
                "text_embedding": Embedding(
                    value=embedding,
                    distance_type=DistanceType.COSINE
                )
            },
            metadata={
                "document_id": document_id,
                "embedder_type": embedder_type,
                "embedding_model": embedding_model,
                "timestamp": datetime.now().isoformat(),
                "error_message": error or "",
            },
            metrics=metrics
        )
        
        self._add_record(record)
    
    def track_output_generation(
        self,
        output_id: str,
        formatter_type: str,
        num_documents: int,
        num_chunks: int,
        total_chunks_size: int,
        output_size: int,
        processing_time: float,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Track metrics for output dataset generation.
        
        Args:
            output_id: Unique identifier for the output
            formatter_type: Type of formatter used
            num_documents: Number of documents processed
            num_chunks: Number of chunks processed
            total_chunks_size: Total size of all chunks in bytes
            output_size: Size of the output dataset in bytes
            processing_time: Time taken for output generation in seconds
            success: Whether output generation succeeded
            error: Error message if output generation failed
        """
        if not self.enabled:
            return
            
        metrics = {
            "num_documents": Metric(num_documents),
            "num_chunks": Metric(num_chunks),
            "total_chunks_size_bytes": Metric(total_chunks_size),
            "output_size_bytes": Metric(output_size),
            "processing_time_seconds": Metric(processing_time),
            "success": Metric(1.0 if success else 0.0),
        }
        
        if error:
            metrics["error"] = Metric(1.0)
            
        # Create record
        record = Record(
            id=output_id,
            record_type=RecordType.FEATURE_STORE,
            metadata={
                "formatter_type": formatter_type,
                "timestamp": datetime.now().isoformat(),
                "error_message": error or "",
            },
            metrics=metrics
        )
        
        self._add_record(record)
    
    def track_collection_performance(
        self,
        collector_id: str,
        collector_type: str,
        source: str,
        num_queries: int,
        num_documents_found: int,
        num_documents_processed: int,
        collection_time: float,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Track metrics for data collection.
        
        Args:
            collector_id: Unique identifier for the collection run
            collector_type: Type of collector used
            source: Source of the collected data
            num_queries: Number of queries performed
            num_documents_found: Number of documents found
            num_documents_processed: Number of documents successfully processed
            collection_time: Time taken for collection in seconds
            success: Whether collection succeeded
            error: Error message if collection failed
        """
        if not self.enabled:
            return
            
        metrics = {
            "num_queries": Metric(num_queries),
            "num_documents_found": Metric(num_documents_found),
            "num_documents_processed": Metric(num_documents_processed),
            "collection_time_seconds": Metric(collection_time),
            "success_rate": Metric(num_documents_processed / max(1, num_documents_found)),
            "success": Metric(1.0 if success else 0.0),
        }
        
        if error:
            metrics["error"] = Metric(1.0)
            
        # Create record
        record = Record(
            id=collector_id,
            record_type=RecordType.FEATURE_STORE,
            metadata={
                "collector_type": collector_type,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "error_message": error or "",
            },
            metrics=metrics
        )
        
        self._add_record(record)
    
    def _add_record(self, record: Record):
        """Add a record to the pending records and flush if batch size is reached."""
        self.pending_records.append(record)
        
        if len(self.pending_records) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Flush all pending records to Arize Phoenix."""
        if not self.enabled or not self.pending_records:
            return
        
        # If we're using the OpenTelemetry integration, we don't need to flush
        # as spans are automatically exported
        if PHOENIX_OTEL_AVAILABLE:
            # Pending records aren't needed with OpenTelemetry
            self.pending_records = []
            return
            
        try:
            # Log before sending for debugging
            logging.info(f"Sending {len(self.pending_records)} records to Arize Phoenix project '{self.project_name}'")
            
            # Always use the consistent project name
            self.client.log_records(
                project_name=self.project_name,
                records=self.pending_records
            )
            logging.info(f"✅ Successfully sent {len(self.pending_records)} records to Arize Phoenix project '{self.project_name}'")
            self.pending_records = []
        except Exception as e:
            logging.error(f"Failed to send records to Arize Phoenix: {str(e)}")
    
    def __del__(self):
        """Ensure all records are flushed when the adapter is destroyed."""
        self.flush()


def get_arize_adapter(config: Dict[str, Any]) -> Union[RAGDatasetBuilderArizeAdapter, None]:
    """
    Factory function to create an Arize Phoenix adapter based on configuration.
    
    Args:
        config: Configuration dictionary with tracking settings
    
    Returns:
        An initialized Arize Phoenix adapter or None if tracking is disabled
    """
    tracking_config = config.get("performance_tracking", {})
    enabled = tracking_config.get("enabled", False)
    
    if not enabled:
        return None
    
    # Get project name from config first, then environment variable, then default
    project_name = tracking_config.get(
        "project_name",
        os.environ.get("PHOENIX_PROJECT_NAME", "pathrag-dataset-builder")
    )
    
    # Get Phoenix URL from config first, then environment variable, then default
    phoenix_url = tracking_config.get(
        "phoenix_url",
        os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:8084")
    )
    
    logging.info(f"Initializing Arize Phoenix adapter with project: {project_name}")
    logging.info(f"Phoenix endpoint: {phoenix_url}")
    
    return RAGDatasetBuilderArizeAdapter(
        project_name=project_name,
        enabled=True,
        phoenix_url=phoenix_url,
        batch_size=tracking_config.get("batch_size", 100)
    )


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "performance_tracking": {
            "enabled": True,
            "project_name": "rag_dataset_builder",
            "phoenix_url": "http://localhost:8084",
            "batch_size": 50
        }
    }
    
    # Create adapter
    adapter = get_arize_adapter(config)
    
    if adapter:
        # Track document processing
        start_time = time.time()
        time.sleep(0.1)  # Simulate processing
        adapter.track_document_processing(
            document_id="doc-123",
            document_path="/path/to/document.pdf",
            document_type="pdf",
            processing_time=time.time() - start_time,
            document_size=1024 * 1024,  # 1MB
            metadata={"author": "John Doe", "date": "2025-01-01"},
            success=True
        )
        
        # Flush records
        adapter.flush()
        print("Performance tracking example completed.")
