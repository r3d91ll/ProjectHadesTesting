#!/usr/bin/env python3
"""
RAG Dataset Builder

A flexible, memory-efficient tool for building datasets for 
Retrieval-Augmented Generation (RAG) systems.

This is the core builder module that handles document processing,
chunking, embedding generation, and output formatting.
"""

import os
import sys
import json
import time
import logging
import argparse
import gc
import glob
import uuid
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional, Generator, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_dataset_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_dataset_builder")


class BaseProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    def process_document(self, doc_path: str) -> Dict[str, Any]:
        """
        Process a single document.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Processed document data
        """
        pass


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        pass


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        pass


class BaseOutputFormatter(ABC):
    """Abstract base class for output formatters."""
    
    @abstractmethod
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
        pass


class SimpleTextProcessor(BaseProcessor):
    """Simple text processor for plain text documents."""
    
    def __init__(self):
        """Initialize the text processor."""
        pass
    
    def process_document(self, doc_path: str) -> Dict[str, Any]:
        """
        Process a plain text document.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Processed document data
        """
        try:
            # Read text file
            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract basic metadata
            filename = os.path.basename(doc_path)
            file_size = os.path.getsize(doc_path)
            category = self._determine_category(doc_path)
            
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "path": doc_path,
                "text": text,
                "metadata": {
                    "filename": filename,
                    "file_size": file_size,
                    "category": category,
                    "extension": os.path.splitext(filename)[1],
                    "created_at": datetime.fromtimestamp(os.path.getctime(doc_path)).isoformat(),
                    "modified_at": datetime.fromtimestamp(os.path.getmtime(doc_path)).isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "path": doc_path,
                "text": "",
                "metadata": {
                    "filename": os.path.basename(doc_path),
                    "error": str(e)
                }
            }
    
    def _determine_category(self, doc_path: str) -> str:
        """
        Determine document category based on path.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Document category
        """
        path_parts = doc_path.split(os.sep)
        for part in path_parts:
            if part in ["papers", "documentation", "code"]:
                return part
            
            # Handle more specific categories
            if part in ["actor_network_theory", "sts_digital_sociology", "value_studies"]:
                return part
        
        return "unknown"


class SlidingWindowChunker(BaseChunker):
    """Chunker that uses a sliding window approach."""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text using a sliding window.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        if not text:
            return []
        
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        
        if len(tokens) <= self.chunk_size:
            # Text is shorter than chunk size, return as single chunk
            return [{
                "id": self._generate_chunk_id(text, metadata["id"], 0),
                "text": text,
                "metadata": {
                    "doc_id": metadata["id"],
                    "position": 0,
                    "source": metadata["path"]
                }
            }]
        
        chunks = []
        position = 0
        
        # Create chunks with sliding window
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            if len(chunk_tokens) < self.chunk_size / 2 and chunks:
                # If the last chunk is too small, merge with previous chunk
                continue
            
            chunk_text = " ".join(chunk_tokens)
            chunks.append({
                "id": self._generate_chunk_id(chunk_text, metadata["id"], position),
                "text": chunk_text,
                "metadata": {
                    "doc_id": metadata["id"],
                    "position": position,
                    "source": metadata["path"]
                }
            })
            position += 1
        
        return chunks
    
    def _generate_chunk_id(self, text: str, doc_id: str, position: int) -> str:
        """
        Generate a unique ID for a chunk.
        
        Args:
            text: Chunk text
            doc_id: Document ID
            position: Chunk position
            
        Returns:
            Chunk ID
        """
        hash_input = f"{doc_id}:{position}:{text[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()
