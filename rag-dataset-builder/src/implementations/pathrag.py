#!/usr/bin/env python3
"""
PathRAG Implementation

This module provides a complete implementation of PathRAG, the path-based retrieval
augmented generation system. It implements the RetrievalSystem interface and 
serves as a reference implementation for other retrieval system variants.
"""

import os
import time
import json
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import networkx as nx

from ..core.interfaces import (
    DocumentProcessor,
    TextChunker,
    Embedder,
    StorageBackend,
    PerformanceTracker,
    RetrievalSystem
)
from ..core.base import (
    BaseRetrievalSystem,
    NetworkXStorageBackend
)
from ..core.plugin import register

# Configure logging
logger = logging.getLogger("rag_dataset_builder.pathrag")

@register('retrieval_systems', 'pathrag')
class PathRAG(BaseRetrievalSystem):
    """
    Implementation of PathRAG (Path-based Retrieval Augmented Generation).
    
    PathRAG builds a knowledge graph where documents and chunks are nodes,
    and relationships between them form edges. This allows for path-based
    retrieval that can provide more context and reasoning steps than
    traditional vector-based retrieval.
    """
    
    def __init__(self, tracker: Optional[PerformanceTracker] = None):
        """
        Initialize PathRAG implementation.
        
        Args:
            tracker: Performance tracker to use
        """
        super().__init__(name="pathrag", tracker=tracker)
        self.storage_backend = None
        self.chunks_dir = None
        self.metadata_dir = None
        self.embeddings_dir = None
    
    def _initialize_implementation(self) -> None:
        """Initialize implementation-specific components."""
        # Get output directory
        output_dir = self.config.get("output_dir")
        if not output_dir:
            raise ValueError("Output directory not specified in configuration")
        
        # Create directory structure
        self.chunks_dir = os.path.join(output_dir, "chunks")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.embeddings_dir = os.path.join(output_dir, "embeddings")
        self.graph_dir = os.path.join(output_dir, "graph")
        
        for directory in [self.chunks_dir, self.metadata_dir, self.embeddings_dir, self.graph_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Get storage backend configuration
        storage_backend_type = self.config.get("storage_backend", "networkx")
        
        # Create storage backend instance
        if storage_backend_type == "networkx":
            self.storage_backend = NetworkXStorageBackend()
        else:
            # Use plugin system to get the appropriate storage backend
            from ..core.plugin import create_storage_backend
            self.storage_backend = create_storage_backend(storage_backend_type)
        
        # Initialize or load existing data
        data_path = os.path.join(self.graph_dir, "knowledge_graph")
        if os.path.exists(f"{data_path}.pickle"):
            try:
                self.storage_backend.load(data_path)
                logger.info(f"Loaded existing knowledge graph")
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
                raise
    
    def process_document(self, 
                        file_path: str, 
                        processor: DocumentProcessor,
                        chunker: TextChunker,
                        embedder: Embedder) -> None:
        """
        Process a document and add it to the PathRAG dataset.
        
        Args:
            file_path: Path to the document
            processor: Document processor to use
            chunker: Text chunker to use
            embedder: Embedder to use
            
        Returns:
            None
        """
        start_time = time.time()
        file_path = os.path.abspath(file_path)
        
        try:
            # Step 1: Process document
            logger.info(f"Processing document: {file_path}")
            text, metadata = processor.process(file_path)
            document_id = metadata.get("id")
            
            # Track document processing if a tracker is available
            if self.tracker:
                self.tracker.track_document_processing(
                    document_id=document_id,
                    document_path=file_path,
                    processing_time=time.time() - start_time,
                    metadata=metadata,
                    success=True
                )
            
            # Step 2: Chunk document
            chunk_start_time = time.time()
            logger.info(f"Chunking document: {file_path}")
            chunks = chunker.chunk_text(text, metadata)
            
            # Track chunking if a tracker is available
            if self.tracker:
                self.tracker.track_chunking(
                    document_id=document_id,
                    chunker_type=chunker.__class__.__name__,
                    num_chunks=len(chunks),
                    chunking_time=time.time() - chunk_start_time,
                    success=True
                )
            
            # Step 3: Generate embeddings
            embed_start_time = time.time()
            logger.info(f"Generating embeddings for document: {file_path}")
            texts = [chunk.get("content", "") for chunk in chunks]
            embeddings = embedder.embed_texts(texts)
            
            # Step 4: Add document to the knowledge graph
            graph_start_time = time.time()
            self._add_to_knowledge_graph(document_id, chunks, embeddings, metadata)
            
            # Step 5: Save document metadata
            self._save_document_metadata(document_id, metadata)
            
            # Step 6: Save chunks
            self._save_chunks(chunks)
            
            # Step 7: Save embeddings
            self._save_embeddings(document_id, chunks, embeddings)
            
            logger.info(f"Successfully processed document: {file_path}")
            
            # Track output generation if a tracker is available
            if self.tracker:
                self.tracker.track_output_generation(
                    output_type="pathrag",
                    num_documents=1,
                    num_chunks=len(chunks),
                    output_time=time.time() - graph_start_time,
                    success=True
                )
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            
            # Track error if a tracker is available
            if self.tracker:
                self.tracker.track_document_processing(
                    document_id=metadata.get("id", "unknown"),
                    document_path=file_path,
                    processing_time=time.time() - start_time,
                    metadata=metadata,
                    success=False,
                    error=str(e)
                )
            
            raise
    
    def _add_to_knowledge_graph(self, 
                                document_id: str, 
                                chunks: List[Dict[str, Any]], 
                                embeddings: List[List[float]], 
                                metadata: Dict[str, Any]) -> None:
        """
        Add a document and its chunks to the knowledge graph.
        
        Args:
            document_id: ID of the document
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Document metadata
        """
        # Add document node
        self.storage_backend.add_item(
            item_id=document_id,
            item_type="document",
            title=metadata.get("filename", "Unknown"),
            category=metadata.get("metadata", {}).get("category", "unknown")
        )
        
        # Add chunk nodes and document -> chunk relationships
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["id"]
            
            # Add chunk node
            self.storage_backend.add_item(
                item_id=chunk_id,
                item_type="chunk",
                position=i,
                # Store the embedding with the chunk if enabled in config
                embedding=embeddings[i] if self.config.get("path_retrieval", {}).get("store_embeddings_with_chunks", True) else None
            )
            
            # Add document -> chunk relationship
            self.storage_backend.add_relationship(
                from_item=document_id,
                to_item=chunk_id,
                relationship_type="contains"
            )
        
        # Add category node and relationship if available
        category = metadata.get("metadata", {}).get("category")
        if category and category != "unknown":
            category_id = f"category:{category}"
            
            # We can't check directly if the item exists in the storage backend
            # as the interface doesn't require this capability, so we'll add it safely
            try:
                self.storage_backend.add_item(
                    item_id=category_id,
                    item_type="category",
                    name=category
                )
            except Exception as e:
                logger.debug(f"Category node may already exist: {e}")
            
            # Add document -> category relationship
            self.storage_backend.add_relationship(
                from_item=document_id,
                to_item=category_id,
                relationship_type="belongs_to"
            )
        
        # Add connections between sequential chunks
        for i in range(len(chunks) - 1):
            self.storage_backend.add_relationship(
                from_item=chunks[i]["id"],
                to_item=chunks[i + 1]["id"],
                relationship_type="next"
            )
            
        # Add similarity connections between chunks if configured
        if self.config.get("path_retrieval", {}).get("add_similarity_edges", True):
            similarity_threshold = self.config.get("path_retrieval", {}).get("similarity_threshold", 0.7)
            
            # Calculate similarities between chunks
            import numpy as np
            for i in range(len(chunks)):
                for j in range(i+1, len(chunks)):
                    # Calculate cosine similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    
                    # Add edge if similarity is above threshold
                    if similarity > similarity_threshold:
                        self.storage_backend.add_relationship(
                            from_item=chunks[i]["id"],
                            to_item=chunks[j]["id"],
                            relationship_type="similar_to",
                            similarity=float(similarity)
                        )
    
    def _save_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save document metadata to disk.
        
        Args:
            document_id: ID of the document
            metadata: Document metadata
        """
        metadata_path = os.path.join(self.metadata_dir, f"{document_id}.json")
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}")
            raise
    
    def _save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Save chunks to disk.
        
        Args:
            chunks: List of text chunks
        """
        for chunk in chunks:
            chunk_id = chunk["id"]
            chunk_path = os.path.join(self.chunks_dir, f"{chunk_id}.json")
            
            try:
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error saving chunk to {chunk_path}: {e}")
                raise
    
    def _save_embeddings(self, 
                         document_id: str, 
                         chunks: List[Dict[str, Any]], 
                         embeddings: List[List[float]]) -> None:
        """
        Save embeddings to disk.
        
        Args:
            document_id: ID of the document
            chunks: List of text chunks
            embeddings: List of embedding vectors
        """
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["id"]
            embedding_data = {
                "id": chunk_id,
                "doc_id": document_id,
                "embedding": embedding
            }
            
            embedding_path = os.path.join(self.embeddings_dir, f"{chunk_id}.json")
            
            try:
                with open(embedding_path, 'w', encoding='utf-8') as f:
                    json.dump(embedding_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error saving embedding to {embedding_path}: {e}")
                raise
    
    def save(self, output_dir: Optional[str] = None) -> None:
        """
        Save the PathRAG dataset.
        
        Args:
            output_dir: Directory to save the dataset (if different from initialization)
        """
        # Use provided output directory or the one from configuration
        if output_dir:
            # Create new directory structure
            chunks_dir = os.path.join(output_dir, "chunks")
            metadata_dir = os.path.join(output_dir, "metadata")
            embeddings_dir = os.path.join(output_dir, "embeddings")
            graph_dir = os.path.join(output_dir, "graph")
            
            for directory in [chunks_dir, metadata_dir, embeddings_dir, graph_dir]:
                os.makedirs(directory, exist_ok=True)
            
            # Copy files to new location
            if os.path.exists(self.chunks_dir):
                for file in os.listdir(self.chunks_dir):
                    shutil.copy2(
                        os.path.join(self.chunks_dir, file),
                        os.path.join(chunks_dir, file)
                    )
            
            if os.path.exists(self.metadata_dir):
                for file in os.listdir(self.metadata_dir):
                    shutil.copy2(
                        os.path.join(self.metadata_dir, file),
                        os.path.join(metadata_dir, file)
                    )
            
            if os.path.exists(self.embeddings_dir):
                for file in os.listdir(self.embeddings_dir):
                    shutil.copy2(
                        os.path.join(self.embeddings_dir, file),
                        os.path.join(embeddings_dir, file)
                    )
            
            # Save graph to new location
            graph_path = os.path.join(graph_dir, "knowledge_graph")
            self.storage_backend.save(graph_path)
            
        else:
            # Save graph to existing location
            graph_path = os.path.join(self.graph_dir, "knowledge_graph")
            self.storage_backend.save(graph_path)
        
        logger.info("PathRAG dataset saved successfully")
    
    def get_paths(self, query_embedding: List[float], max_paths: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant paths from the knowledge graph for a query.
        
        Args:
            query_embedding: Query embedding vector
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths, each containing nodes and edges
        """
        import numpy as np
        paths = []
        
        # Get path retrieval settings from config
        path_config = self.config.get("path_retrieval", {})
        max_path_length = path_config.get("max_path_length", 3)
        similarity_threshold = path_config.get("similarity_threshold", 0.7)
        consider_bidirectional = path_config.get("consider_bidirectional", True)
        path_ranking = path_config.get("path_ranking", "combined")
        max_start_nodes = min(path_config.get("max_start_nodes", 10), max_paths * 2)
        
        # Step 1: Find the most similar chunks to the query
        similar_chunks = self.storage_backend.query({
            "item_type": "chunk",
            "embedding": query_embedding,
            "similarity_threshold": similarity_threshold
        }, limit=max_start_nodes)
        
        if not similar_chunks:
            logger.warning("No similar chunks found for query")
            return []
        
        # TODO: Implement actual path finding algorithm here
        # This is still a placeholder but with better structure
        # A complete implementation would use a graph traversal algorithm
        # to find paths through the knowledge graph
        
        # For now, return a more structured placeholder
        if similar_chunks:
            logger.info(f"Found {len(similar_chunks)} similar chunks")
            # Just create a simple path with the first similar chunk for now
            paths.append({
                "nodes": [similar_chunks[0]],
                "edges": [],
                "similarity": similar_chunks[0].get("similarity", 0.0)
            })
        
        logger.warning("get_paths is still a placeholder implementation with limited functionality")
        return paths
