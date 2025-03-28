#!/usr/bin/env python3
"""
RAG Dataset Builder - Output Formatters

This module contains implementations of output formatters for the RAG Dataset Builder.
"""

import os
import json
import logging
import pickle
import shutil
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from .builder import BaseOutputFormatter

# Configure logging
logger = logging.getLogger("rag_dataset_builder.formatters")


class BaseFormatter(BaseOutputFormatter):
    """Base class for output formatters with common functionality."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the formatter.
        
        Args:
            output_dir: Directory to save output
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def format_output(self, chunks: List[Dict[str, Any]], 
                    embeddings: List[List[float]], 
                    metadata: Dict[str, Any]) -> None:
        """
        Format and save output.
        
        Args:
            chunks: List of chunks
            embeddings: List of embeddings
            metadata: Document metadata
        """
        raise NotImplementedError("Subclasses must implement format_output")
    
    def save_json(self, data: Dict[str, Any], path: str) -> None:
        """
        Save data as JSON.
        
        Args:
            data: Data to save
            path: Path to save to
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving JSON to {path}: {e}")


class PathRAGFormatter(BaseFormatter):
    """Output formatter for PathRAG format."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the formatter.
        
        Args:
            output_dir: Directory to save output
        """
        super().__init__(output_dir)
        
        # Create subdirectories
        self.chunks_dir = os.path.join(output_dir, "chunks")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.embeddings_dir = os.path.join(output_dir, "embeddings")
        self.graph_dir = os.path.join(output_dir, "graph")
        
        for directory in [self.chunks_dir, self.metadata_dir, self.embeddings_dir, self.graph_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.DiGraph()
        self.graph_path = os.path.join(self.graph_dir, "knowledge_graph")
        
        # Load existing graph if it exists
        if os.path.exists(f"{self.graph_path}.pickle"):
            try:
                with open(f"{self.graph_path}.pickle", 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                logger.info(f"Loaded existing knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes")
            except Exception as e:
                logger.error(f"Error loading existing graph: {e}")
    
    def format_output(self, chunks: List[Dict[str, Any]], 
                    embeddings: List[List[float]], 
                    metadata: Dict[str, Any]) -> None:
        """
        Format and save output in PathRAG format.
        
        Args:
            chunks: List of chunks
            embeddings: List of embeddings
            metadata: Document metadata
        """
        if not chunks:
            logger.warning("No chunks to format")
            return
        
        if len(chunks) != len(embeddings):
            logger.error(f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})")
            return
        
        try:
            # Save document metadata
            doc_id = metadata["id"]
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            self.save_json(metadata, metadata_path)
            
            # Add document node to knowledge graph
            self.knowledge_graph.add_node(
                doc_id,
                type="document",
                title=metadata.get("metadata", {}).get("filename", "Unknown"),
                category=metadata.get("metadata", {}).get("category", "unknown")
            )
            
            # Process each chunk and its embedding
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = chunk["id"]
                
                # Save chunk
                chunk_path = os.path.join(self.chunks_dir, f"{chunk_id}.json")
                self.save_json(chunk, chunk_path)
                
                # Save embedding
                embedding_data = {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "embedding": embedding
                }
                embedding_path = os.path.join(self.embeddings_dir, f"{chunk_id}.json")
                self.save_json(embedding_data, embedding_path)
                
                # Add chunk node to knowledge graph
                self.knowledge_graph.add_node(
                    chunk_id,
                    type="chunk",
                    position=chunk["metadata"].get("position", 0)
                )
                
                # Add document -> chunk relationship
                self.knowledge_graph.add_edge(
                    doc_id,
                    chunk_id,
                    type="contains"
                )
            
            # Add category node and relationship if available
            category = metadata.get("metadata", {}).get("category")
            if category and category != "unknown":
                category_id = f"category:{category}"
                
                # Add category node if it doesn't exist
                if not self.knowledge_graph.has_node(category_id):
                    self.knowledge_graph.add_node(
                        category_id,
                        type="category",
                        name=category
                    )
                
                # Add document -> category relationship
                self.knowledge_graph.add_edge(
                    doc_id,
                    category_id,
                    type="belongs_to"
                )
            
            # Save knowledge graph
            self._save_graph()
            
            logger.info(f"Formatted document {doc_id} with {len(chunks)} chunks in PathRAG format")
        
        except Exception as e:
            logger.error(f"Error formatting output: {e}")
    
    def _save_graph(self) -> None:
        """Save the knowledge graph to disk."""
        try:
            # Save as pickle for efficient loading
            with open(f"{self.graph_path}.pickle", 'wb') as f:
                pickle.dump(self.knowledge_graph, f)
            
            # Save as JSON for interoperability
            graph_data = {
                "nodes": [{"id": node, **data} for node, data in self.knowledge_graph.nodes(data=True)],
                "edges": [{"source": u, "target": v, **data} for u, v, data in self.knowledge_graph.edges(data=True)]
            }
            
            with open(f"{self.graph_path}.json", 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False)
            
            logger.info(f"Saved knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes and {self.knowledge_graph.number_of_edges()} edges")
        
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")


class VectorDBFormatter(BaseFormatter):
    """Output formatter for vector database format."""
    
    def __init__(self, output_dir: str, vector_db_type: str = "faiss"):
        """
        Initialize the formatter.
        
        Args:
            output_dir: Directory to save output
            vector_db_type: Type of vector database (faiss, chroma, etc.)
        """
        super().__init__(output_dir)
        self.vector_db_type = vector_db_type
        
        # Create subdirectories
        self.texts_dir = os.path.join(output_dir, "texts")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.vectors_dir = os.path.join(output_dir, "vectors")
        
        for directory in [self.texts_dir, self.metadata_dir, self.vectors_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def format_output(self, chunks: List[Dict[str, Any]], 
                    embeddings: List[List[float]], 
                    metadata: Dict[str, Any]) -> None:
        """
        Format and save output in vector database format.
        
        Args:
            chunks: List of chunks
            embeddings: List of embeddings
            metadata: Document metadata
        """
        if not chunks:
            logger.warning("No chunks to format")
            return
        
        if len(chunks) != len(embeddings):
            logger.error(f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})")
            return
        
        try:
            # Save document metadata
            doc_id = metadata["id"]
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            self.save_json(metadata, metadata_path)
            
            # Process each chunk and its embedding
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = chunk["id"]
                
                # Save chunk text
                text_path = os.path.join(self.texts_dir, f"{chunk_id}.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(chunk["text"])
                
                # Save chunk metadata
                chunk_metadata = {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "position": chunk["metadata"].get("position", 0),
                    "source": chunk["metadata"].get("source", ""),
                    "category": metadata.get("metadata", {}).get("category", "unknown")
                }
                chunk_metadata_path = os.path.join(self.metadata_dir, f"{chunk_id}.json")
                self.save_json(chunk_metadata, chunk_metadata_path)
            
            # Save embeddings in numpy format
            embeddings_array = np.array(embeddings, dtype=np.float32)
            chunk_ids = [chunk["id"] for chunk in chunks]
            
            embeddings_data = {
                "ids": chunk_ids,
                "doc_id": doc_id,
                "embeddings": embeddings_array.tolist()
            }
            
            vector_path = os.path.join(self.vectors_dir, f"{doc_id}.json")
            self.save_json(embeddings_data, vector_path)
            
            logger.info(f"Formatted document {doc_id} with {len(chunks)} chunks in vector database format")
        
        except Exception as e:
            logger.error(f"Error formatting output: {e}")


class HuggingFaceDatasetFormatter(BaseFormatter):
    """Output formatter for Hugging Face dataset format."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the formatter.
        
        Args:
            output_dir: Directory to save output
        """
        super().__init__(output_dir)
        self.dataset_file = os.path.join(output_dir, "dataset.jsonl")
        
        # Create empty dataset file if it doesn't exist
        if not os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'w', encoding='utf-8') as f:
                pass
    
    def format_output(self, chunks: List[Dict[str, Any]], 
                    embeddings: List[List[float]], 
                    metadata: Dict[str, Any]) -> None:
        """
        Format and save output in Hugging Face dataset format.
        
        Args:
            chunks: List of chunks
            embeddings: List of embeddings
            metadata: Document metadata
        """
        if not chunks:
            logger.warning("No chunks to format")
            return
        
        if len(chunks) != len(embeddings):
            logger.error(f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})")
            return
        
        try:
            doc_id = metadata["id"]
            category = metadata.get("metadata", {}).get("category", "unknown")
            filename = metadata.get("metadata", {}).get("filename", "Unknown")
            
            # Process each chunk and its embedding
            with open(self.dataset_file, 'a', encoding='utf-8') as f:
                for chunk, embedding in zip(chunks, embeddings):
                    chunk_id = chunk["id"]
                    position = chunk["metadata"].get("position", 0)
                    
                    # Create dataset entry
                    entry = {
                        "id": chunk_id,
                        "text": chunk["text"],
                        "metadata": {
                            "doc_id": doc_id,
                            "chunk_position": position,
                            "category": category,
                            "source": filename
                        },
                        "embedding": embedding
                    }
                    
                    # Write as JSON line
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks to Hugging Face dataset")
        
        except Exception as e:
            logger.error(f"Error formatting output: {e}")
