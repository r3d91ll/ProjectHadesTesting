#!/usr/bin/env python3
"""
Base Classes for RAG Dataset Builder

This module provides base implementations of the interfaces defined in interfaces.py.
These classes provide common functionality while allowing for extension by specific
implementations.
"""

import os
import json
import time
import logging
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import networkx as nx

from .interfaces import (
    DocumentProcessor, 
    TextChunker, 
    Embedder,
    StorageBackend,
    OutputFormatter,
    DatasetCollector,
    PerformanceTracker,
    RetrievalSystem
)

# Configure logging
logger = logging.getLogger("rag_dataset_builder")


class BaseDocumentProcessor(DocumentProcessor):
    """Base class for document processors with common functionality."""
    
    def __init__(self, extract_metadata: bool = True, track_license: bool = True):
        """
        Initialize the document processor.
        
        Args:
            extract_metadata: Whether to extract metadata from documents
            track_license: Whether to track license information
        """
        self.extract_metadata = extract_metadata
        self.track_license = track_license
    
    def process(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a document and extract text and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        if not self.supports_file_type(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Track processing time
        start_time = time.time()
        
        try:
            # Extract text and basic metadata
            text = self._extract_text(file_path)
            metadata = self._extract_basic_metadata(file_path)
            
            # Extract additional metadata if requested
            if self.extract_metadata:
                additional_metadata = self._extract_additional_metadata(file_path, text)
                metadata.update(additional_metadata)
            
            # Track license information if requested
            if self.track_license:
                license_info = self._extract_license_info(file_path)
                metadata["license"] = license_info
            
            # Track processing time
            metadata["processing"] = {
                "processor": self.__class__.__name__,
                "time": time.time() - start_time,
                "timestamp": time.time()
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from a document. Must be implemented by subclasses.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text
        """
        raise NotImplementedError("Subclasses must implement _extract_text")
    
    def _extract_basic_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract basic metadata common to all document types.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Basic metadata dictionary
        """
        file_path = Path(file_path)
        file_stats = file_path.stat()
        
        return {
            "filename": file_path.name,
            "path": str(file_path.absolute()),
            "file_type": file_path.suffix.lstrip('.').lower(),
            "file_size": file_stats.st_size,
            "created": file_stats.st_ctime,
            "modified": file_stats.st_mtime,
            "id": self._generate_document_id(file_path)
        }
    
    def _extract_additional_metadata(self, file_path: str, text: str) -> Dict[str, Any]:
        """
        Extract additional metadata specific to document type.
        Subclasses should override this to provide type-specific metadata.
        
        Args:
            file_path: Path to the document file
            text: Extracted document text
            
        Returns:
            Additional metadata dictionary
        """
        return {}
    
    def _extract_license_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extract license information from document.
        Subclasses should override this to provide type-specific license extraction.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            License information dictionary
        """
        return {
            "type": "unknown",
            "details": "No license information available",
            "attribution_required": True  # Default to requiring attribution
        }
    
    def _generate_document_id(self, file_path: Path) -> str:
        """
        Generate a unique document ID based on file path and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Unique document ID
        """
        # Use file path and modification time to create a unique hash
        unique_string = f"{file_path.absolute()}_{file_path.stat().st_mtime}"
        return f"doc-{hashlib.md5(unique_string.encode()).hexdigest()[:12]}"


class BaseTextChunker(TextChunker):
    """Base class for text chunkers with common functionality."""
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks, each with text content and metadata
        """
        # Track chunking time
        start_time = time.time()
        
        # Perform chunking (to be implemented by subclasses)
        text_chunks = self._split_text(text)
        
        # Create chunk objects with metadata
        document_id = metadata.get("id", "unknown")
        chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            
            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "position": i,
                    "chunk_size": len(chunk_text),
                    "chunker": self.__class__.__name__,
                    "chunking_time": time.time() - start_time
                }
            })
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks. Must be implemented by subclasses.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        raise NotImplementedError("Subclasses must implement _split_text")


class BaseEmbedder(Embedder):
    """Base class for embedders with common functionality."""
    
    def __init__(self, cache_embeddings: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize the embedder.
        
        Args:
            cache_embeddings: Whether to cache embeddings
            cache_dir: Directory to store cached embeddings
        """
        self.cache_embeddings = cache_embeddings
        self.cache_dir = cache_dir or "./.cache/embeddings"
        
        if self.cache_embeddings:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per text)
        """
        # Check cache if enabled
        if self.cache_embeddings:
            cached_embeddings, texts_to_embed, indices = self._get_cached_embeddings(texts)
            
            if not texts_to_embed:
                return cached_embeddings
        else:
            texts_to_embed = texts
            indices = list(range(len(texts)))
            cached_embeddings = [None] * len(texts)
        
        # Generate embeddings for texts not in cache
        generated_embeddings = self._generate_embeddings(texts_to_embed)
        
        # Update cache if enabled
        if self.cache_embeddings:
            self._update_cache(texts_to_embed, generated_embeddings)
        
        # Combine cached and generated embeddings
        result = cached_embeddings.copy()
        for i, embedding in zip(indices, generated_embeddings):
            result[i] = embedding
        
        return result
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts. Must be implemented by subclasses.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError("Subclasses must implement _generate_embeddings")
    
    def _get_cached_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], List[str], List[int]]:
        """
        Get cached embeddings for texts.
        
        Args:
            texts: List of text strings to check in cache
            
        Returns:
            Tuple of (cached_embeddings, texts_to_embed, indices)
            where cached_embeddings is a list of cached embeddings (or None if not cached),
            texts_to_embed is a list of texts that need to be embedded,
            and indices is a list of indices in the original list that need embedding
        """
        result = [None] * len(texts)
        texts_to_embed = []
        indices = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{text_hash}.pkl")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        result[i] = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Error loading cached embedding: {e}")
                    texts_to_embed.append(text)
                    indices.append(i)
            else:
                texts_to_embed.append(text)
                indices.append(i)
        
        return result, texts_to_embed, indices
    
    def _update_cache(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Update cache with new embeddings.
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{text_hash}.pkl")
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Error caching embedding: {e}")


class NetworkXStorageBackend(StorageBackend):
    """NetworkX implementation of the storage backend."""
    
    def __init__(self):
        """Initialize the NetworkX graph backend."""
        self.graph = nx.DiGraph()
    
    def add_item(self, item_id: str, item_type: str, **properties) -> None:
        """
        Add an item to the storage backend.
        
        Args:
            item_id: Unique identifier for the item
            item_type: Type of item (document, chunk, entity, etc.)
            **properties: Additional item properties
        """
        self.graph.add_node(item_id, item_type=item_type, **properties)
    
    def add_relationship(self, from_item: str, to_item: str, relationship_type: str, **properties) -> None:
        """
        Add a relationship between two items.
        
        Args:
            from_item: Source item ID
            to_item: Target item ID
            relationship_type: Type of relationship
            **properties: Additional relationship properties
        """
        self.graph.add_edge(from_item, to_item, relationship_type=relationship_type, **properties)
    
    def save(self, path: str) -> None:
        """
        Save the data to disk.
        
        Args:
            path: Path to save the data
        """
        # Save as pickle for efficient loading
        with open(f"{path}.pickle", 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save as JSON for interoperability
        graph_data = {
            "nodes": [{"id": node, **data} for node, data in self.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)]
        }
        
        with open(f"{path}.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load(self, path: str) -> None:
        """
        Load data from disk.
        
        Args:
            path: Path to load the data from
        """
        # Try loading from pickle first (more efficient)
        pickle_path = f"{path}.pickle"
        json_path = f"{path}.json"
        
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    self.graph = pickle.load(f)
                return
            except Exception as e:
                logger.warning(f"Error loading graph from pickle: {e}")
        
        # Fall back to JSON if pickle fails or doesn't exist
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    graph_data = json.load(f)
                
                self.graph = nx.DiGraph()
                
                # Add nodes
                for node in graph_data.get("nodes", []):
                    node_id = node.pop("id")
                    self.graph.add_node(node_id, **node)
                
                # Add edges
                for edge in graph_data.get("edges", []):
                    source = edge.pop("source")
                    target = edge.pop("target")
                    self.graph.add_edge(source, target, **edge)
                    
            except Exception as e:
                logger.error(f"Error loading graph from JSON: {e}")
                raise
        else:
            logger.warning(f"No graph file found at {path}")
            # Initialize an empty graph
            self.graph = nx.DiGraph()
    
    def query(self, query_params: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the storage backend.
        
        Args:
            query_params: Parameters for the query
            limit: Maximum number of results to return
            
        Returns:
            List of matching items
        """
        results = []
        
        # Extract query parameters
        item_type = query_params.get('item_type')
        properties = query_params.get('properties', {})
        relationships = query_params.get('relationships', {})
        embedding = query_params.get('embedding')
        similarity_threshold = query_params.get('similarity_threshold', 0.7)
        
        # Find matching nodes
        for node_id, node_data in self.graph.nodes(data=True):
            # Check if node type matches
            if item_type and node_data.get('item_type') != item_type:
                continue
                
            # Check if properties match
            match = True
            for prop_key, prop_value in properties.items():
                if prop_key not in node_data or node_data[prop_key] != prop_value:
                    match = False
                    break
            
            # If embedding is provided, calculate similarity
            if match and embedding is not None and 'embedding' in node_data:
                import numpy as np
                node_embedding = node_data['embedding']
                similarity = np.dot(embedding, node_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(node_embedding)
                )
                if similarity < similarity_threshold:
                    match = False
                    
            if match:
                # Add node to results
                result = {'id': node_id}
                result.update(node_data)
                results.append(result)
        
        # If relationships are specified, filter by relationships
        if relationships:
            filtered_results = []
            for relation_type, related_ids in relationships.items():
                for result in results:
                    node_id = result['id']
                    has_relation = False
                    
                    # Check outgoing edges
                    for _, target, edge_data in self.graph.edges(node_id, data=True):
                        if edge_data.get('relationship_type') == relation_type and target in related_ids:
                            has_relation = True
                            break
                    
                    # Check incoming edges
                    if not has_relation:
                        for source, _, edge_data in self.graph.in_edges(node_id, data=True):
                            if edge_data.get('relationship_type') == relation_type and source in related_ids:
                                has_relation = True
                                break
                    
                    if has_relation:
                        filtered_results.append(result)
            
            results = filtered_results
                
        # Sort by similarity if embedding was provided
        if embedding is not None:
            results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Limit results
        return results[:limit]


class BaseOutputFormatter(OutputFormatter):
    """Base class for output formatters with common functionality."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the formatter.
        
        Args:
            output_dir: Directory to save output
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def format_output(self, 
                     chunks: List[Dict[str, Any]], 
                     embeddings: List[List[float]], 
                     metadata: Dict[str, Any]) -> None:
        """
        Format and save chunks, embeddings and metadata for RAG systems.
        
        Args:
            chunks: List of text chunks with metadata
            embeddings: List of embedding vectors
            metadata: Document metadata
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})")
        
        # To be implemented by subclasses
        self._format_and_save(chunks, embeddings, metadata)
    
    def _format_and_save(self, 
                        chunks: List[Dict[str, Any]], 
                        embeddings: List[List[float]], 
                        metadata: Dict[str, Any]) -> None:
        """
        Format and save data. Must be implemented by subclasses.
        
        Args:
            chunks: List of chunks
            embeddings: List of embeddings
            metadata: Document metadata
        """
        raise NotImplementedError("Subclasses must implement _format_and_save")
    
    def save_json(self, data: Dict[str, Any], path: str) -> None:
        """
        Save data as JSON.
        
        Args:
            data: Data to save
            path: Path to save to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving JSON to {path}: {e}")
            raise


class BasePerformanceTracker(PerformanceTracker):
    """Base class for performance trackers with common functionality."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the performance tracker.
        
        Args:
            enabled: Whether tracking is enabled
        """
        self.enabled = enabled
        self.tracking_data = {
            "document_processing": [],
            "embedding_generation": [],
            "chunking": [],
            "output_generation": []
        }
    
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
        if not self.enabled:
            return
            
        self.tracking_data["document_processing"].append({
            "document_id": document_id,
            "document_path": document_path,
            "processing_time": processing_time,
            "metadata": metadata,
            "success": success,
            "error": error,
            "timestamp": time.time()
        })
    
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
        if not self.enabled:
            return
            
        self.tracking_data["embedding_generation"].append({
            "chunk_id": chunk_id,
            "embedding_time": embedding_time,
            "embedding_model": embedding_model,
            "success": success,
            "error": error,
            "timestamp": time.time()
        })
    
    def track_chunking(self,
                      document_id: str,
                      chunker_type: str,
                      num_chunks: int,
                      chunking_time: float,
                      success: bool,
                      error: Optional[str] = None) -> None:
        """
        Track chunking performance.
        
        Args:
            document_id: Unique identifier for the document
            chunker_type: Type of chunker used
            num_chunks: Number of chunks created
            chunking_time: Time taken for chunking
            success: Whether chunking succeeded
            error: Error message if chunking failed
        """
        if not self.enabled:
            return
            
        self.tracking_data["chunking"].append({
            "document_id": document_id,
            "chunker_type": chunker_type,
            "num_chunks": num_chunks,
            "chunking_time": chunking_time,
            "success": success,
            "error": error,
            "timestamp": time.time()
        })
    
    def track_output_generation(self,
                               output_type: str,
                               num_documents: int,
                               num_chunks: int,
                               output_time: float,
                               success: bool,
                               error: Optional[str] = None) -> None:
        """
        Track output generation performance.
        
        Args:
            output_type: Type of output generated
            num_documents: Number of documents processed
            num_chunks: Number of chunks processed
            output_time: Time taken for output generation
            success: Whether output generation succeeded
            error: Error message if output generation failed
        """
        if not self.enabled:
            return
            
        self.tracking_data["output_generation"].append({
            "output_type": output_type,
            "num_documents": num_documents,
            "num_chunks": num_chunks,
            "output_time": output_time,
            "success": success,
            "error": error,
            "timestamp": time.time()
        })
    
    def flush(self) -> None:
        """Flush performance tracking data to storage."""
        # Base implementation does nothing
        pass


class BaseRetrievalSystem(RetrievalSystem):
    """Base class for RAG implementations with common functionality."""
    
    def __init__(self, name: str, tracker: Optional[PerformanceTracker] = None):
        """
        Initialize the retrieval system.
        
        Args:
            name: Name of the retrieval system
            tracker: Performance tracker to use
        """
        self._name = name
        self.tracker = tracker
        self.config = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the retrieval system with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._initialize_implementation()
    
    def _initialize_implementation(self) -> None:
        """
        Initialize implementation-specific components.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _initialize_implementation")
    
    @property
    def name(self) -> str:
        """
        Get the name of this retrieval system implementation.
        
        Returns:
            Name of the retrieval system implementation
        """
        return self._name
