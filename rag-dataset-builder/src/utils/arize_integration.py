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

try:
    from arize.phoenix.client import Client
    from arize.phoenix.types import (
        Embedding,
        Metric,
        Record,
        RecordType,
        Document,
        Query,
        Response,
        RetrievalResult,
        DistanceType
    )
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    logging.warning("Arize Phoenix not installed. Run 'pip install arize-phoenix' to enable performance tracking.")

class RAGDatasetBuilderArizeAdapter:
    """
    Adapter for tracking RAG Dataset Builder performance metrics with Arize Phoenix.
    """
    
    def __init__(
        self, 
        project_name: str = "rag_dataset_builder",
        enabled: bool = True,
        phoenix_url: str = "http://localhost:8080",
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
        
        if self.enabled:
            self.client = Client(url=phoenix_url)
            logging.info(f"Arize Phoenix integration enabled. Dashboard available at {phoenix_url}")
        else:
            self.client = None
            if enabled and not PHOENIX_AVAILABLE:
                logging.warning(
                    "Performance tracking is enabled but Arize Phoenix is not installed. "
                    "Run 'pip install arize-phoenix' to enable tracking."
                )
    
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
            
        try:
            self.client.log_records(
                project_name=self.project_name,
                records=self.pending_records
            )
            logging.debug(f"Sent {len(self.pending_records)} records to Arize Phoenix")
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
        
    return RAGDatasetBuilderArizeAdapter(
        project_name=tracking_config.get("project_name", "rag_dataset_builder"),
        enabled=True,
        phoenix_url=tracking_config.get("phoenix_url", "http://localhost:8080"),
        batch_size=tracking_config.get("batch_size", 100)
    )


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "performance_tracking": {
            "enabled": True,
            "project_name": "rag_dataset_builder",
            "phoenix_url": "http://localhost:8080",
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
