#!/usr/bin/env python3
"""
Core Interfaces for RAG Dataset Builder

This module defines the interfaces and abstract base classes for all components
of the RAG Dataset Builder framework. These interfaces provide the extension points
for adding new processors, chunkers, embedders, and RAG implementations.
"""

import abc
from typing import Dict, List, Any, Optional, Tuple, Protocol, Union, Callable
from pathlib import Path


class DocumentProcessor(abc.ABC):
    """Interface for document processors that extract text and metadata from files."""
    
    @abc.abstractmethod
    def process(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a document and extract text and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        pass
    
    @abc.abstractmethod
    def supports_file_type(self, file_path: str) -> bool:
        """
        Check if this processor supports the given file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if this processor can handle the file, False otherwise
        """
        pass


class TextChunker(abc.ABC):
    """Interface for text chunkers that split text into smaller chunks."""
    
    @abc.abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks, each with text content and metadata
        """
        pass


class Embedder(abc.ABC):
    """Interface for embedders that generate vector representations of text."""
    
    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per text)
        """
        pass
    
    @property
    @abc.abstractmethod
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimensionality of the embedding vectors
        """
        pass


class StorageBackend(abc.ABC):
    """Interface for data storage backends (vector databases, graph databases, etc.)."""
    
    @abc.abstractmethod
    def add_item(self, item_id: str, item_type: str, **properties) -> None:
        """
        Add an item to the storage backend.
        
        Args:
            item_id: Unique identifier for the item
            item_type: Type of item (document, chunk, entity, etc.)
            **properties: Additional item properties
        """
        pass
    
    @abc.abstractmethod
    def add_relationship(self, from_item: str, to_item: str, relationship_type: str, **properties) -> None:
        """
        Add a relationship between two items.
        
        Args:
            from_item: Source item ID
            to_item: Target item ID
            relationship_type: Type of relationship
            **properties: Additional relationship properties
        """
        pass
    
    @abc.abstractmethod
    def save(self, path: str) -> None:
        """
        Save the data to disk.
        
        Args:
            path: Path to save the data
        """
        pass
    
    @abc.abstractmethod
    def load(self, path: str) -> None:
        """
        Load data from disk.
        
        Args:
            path: Path to load the data from
        """
        pass
        
    @abc.abstractmethod
    def query(self, query_params: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the storage backend.
        
        Args:
            query_params: Parameters for the query
            limit: Maximum number of results to return
            
        Returns:
            List of matching items
        """
        pass


class OutputFormatter(abc.ABC):
    """Interface for output formatters that prepare data for retrieval systems."""
    
    @abc.abstractmethod
    def format_output(self, 
                      chunks: List[Dict[str, Any]], 
                      embeddings: List[List[float]],
                      metadata: Dict[str, Any]) -> None:
        """
        Format and save chunks, embeddings and metadata for retrieval systems.
        
        Args:
            chunks: List of text chunks with metadata
            embeddings: List of embedding vectors
            metadata: Document metadata
        """
        pass


class DatasetCollector(abc.ABC):
    """Interface for dataset collectors that gather documents from various sources."""
    
    @abc.abstractmethod
    def collect_documents(self, output_dir: str, **kwargs) -> List[str]:
        """
        Collect documents and save them to the output directory.
        
        Args:
            output_dir: Directory to save collected documents
            **kwargs: Additional collection parameters
            
        Returns:
            List of paths to collected documents
        """
        pass


class PerformanceTracker(abc.ABC):
    """Interface for tracking system performance."""
    
    @abc.abstractmethod
    def track_document_processing(self, 
                                 document_id: str,
                                 document_path: str,
                                 processing_time: float,
                                 metadata: Dict[str, Any],
                                 success: bool,
                                 error: Optional[str] = None) -> None:
        """
        Track document processing performance.
        
        Args:
            document_id: Unique identifier for the document
            document_path: Path to the document
            processing_time: Time taken to process the document
            metadata: Document metadata
            success: Whether processing succeeded
            error: Error message if processing failed
        """
        pass
    
    @abc.abstractmethod
    def track_embedding_generation(self,
                                  chunk_id: str,
                                  embedding_time: float,
                                  embedding_model: str,
                                  success: bool,
                                  error: Optional[str] = None) -> None:
        """
        Track embedding generation performance.
        
        Args:
            chunk_id: Unique identifier for the chunk
            embedding_time: Time taken to generate embeddings
            embedding_model: Name of the embedding model
            success: Whether embedding generation succeeded
            error: Error message if embedding generation failed
        """
        pass
    
    @abc.abstractmethod
    def flush(self) -> None:
        """Flush performance tracking data to storage."""
        pass


class RetrievalSystem(abc.ABC):
    """Interface for retrieval system implementations (PathRAG, VectorRAG, etc.)."""
    
    @abc.abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the retrieval system with configuration.
        
        Args:
            config: Configuration dictionary
        """
        pass
    
    @abc.abstractmethod
    def process_document(self, 
                        file_path: str, 
                        processor: DocumentProcessor,
                        chunker: TextChunker,
                        embedder: Embedder) -> None:
        """
        Process a document and add it to the retrieval system's dataset.
        
        Args:
            file_path: Path to the document
            processor: Document processor to use
            chunker: Text chunker to use
            embedder: Embedder to use
        """
        pass
    
    @abc.abstractmethod
    def save(self, output_dir: str) -> None:
        """
        Save the retrieval system's dataset.
        
        Args:
            output_dir: Directory to save the dataset
        """
        pass
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get the name of this retrieval system implementation.
        
        Returns:
            Name of the retrieval system implementation
        """
        pass
