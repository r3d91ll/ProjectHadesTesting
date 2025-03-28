#!/usr/bin/env python3
"""
RAG Dataset Builder - Text Chunkers

This module contains implementations of text chunkers for the RAG Dataset Builder.
"""

import os
import logging
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize

from builder import BaseChunker

# Configure logging
logger = logging.getLogger("rag_dataset_builder.chunkers")

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class SlidingWindowChunker(BaseChunker):
    """Chunker that uses a sliding window approach."""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50, 
                 strategy: str = "word", separator: str = " "):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            strategy: Chunking strategy ("word" or "character")
            separator: Token separator when rejoining chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.separator = separator
    
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
            logger.warning("Empty text provided to chunker")
            return []
        
        # Tokenize text based on strategy
        if self.strategy == "word":
            tokens = text.split()
        elif self.strategy == "character":
            tokens = list(text)
        else:
            logger.warning(f"Unknown chunking strategy: {self.strategy}, falling back to word")
            tokens = text.split()
        
        # Handle case where text is shorter than chunk size
        if len(tokens) <= self.chunk_size:
            return [{
                "id": self._generate_chunk_id(text, metadata["id"], 0),
                "text": text,
                "metadata": {
                    "doc_id": metadata["id"],
                    "position": 0,
                    "source": metadata.get("path", ""),
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", "unknown")
                }
            }]
        
        chunks = []
        position = 0
        
        # Create chunks with sliding window
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            # If this is the last chunk and it's too small, skip it
            if i > 0 and len(chunk_tokens) < self.chunk_size / 3:
                continue
                
            # Rejoin tokens into text
            chunk_text = self.separator.join(chunk_tokens) if self.strategy == "word" else "".join(chunk_tokens)
            
            # Generate chunk
            chunk = {
                "id": self._generate_chunk_id(chunk_text, metadata["id"], position),
                "text": chunk_text,
                "metadata": {
                    "doc_id": metadata["id"],
                    "position": position,
                    "start_idx": i,
                    "end_idx": i + len(chunk_tokens),
                    "source": metadata.get("path", ""),
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", "unknown")
                }
            }
            
            chunks.append(chunk)
            position += 1
        
        logger.info(f"Generated {len(chunks)} chunks using sliding window approach")
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


class SemanticChunker(BaseChunker):
    """Chunker that tries to preserve semantic units like paragraphs and sentences."""
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text preserving semantic units.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        if not text:
            logger.warning("Empty text provided to chunker")
            return []
        
        # If text is already small enough, return as a single chunk
        if len(text) <= self.max_chunk_size:
            return [{
                "id": self._generate_chunk_id(text, metadata["id"], 0),
                "text": text,
                "metadata": {
                    "doc_id": metadata["id"],
                    "position": 0,
                    "source": metadata.get("path", ""),
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", "unknown")
                }
            }]
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_paragraphs = []
        position = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                # Create chunk
                chunk_text = current_chunk.strip()
                chunks.append({
                    "id": self._generate_chunk_id(chunk_text, metadata["id"], position),
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": metadata["id"],
                        "position": position,
                        "source": metadata.get("path", ""),
                        "title": metadata.get("title", ""),
                        "category": metadata.get("category", "unknown"),
                        "paragraphs": len(current_paragraphs)
                    }
                })
                
                # Reset for next chunk
                current_chunk = ""
                current_paragraphs = []
                position += 1
            
            # Add paragraph to current chunk
            current_chunk += paragraph + "\n\n"
            current_paragraphs.append(paragraph)
            
            # Handle very long paragraphs by further splitting into sentences
            if len(paragraph) > self.max_chunk_size:
                # Create chunks from current paragraph
                sentence_chunks = self._split_paragraph_into_sentences(
                    paragraph, metadata["id"], position, metadata
                )
                
                if sentence_chunks:
                    # Add chunks and update position
                    chunks.extend(sentence_chunks)
                    position += len(sentence_chunks)
                    
                    # Reset current chunk since we've handled this paragraph separately
                    current_chunk = ""
                    current_paragraphs = []
        
        # Add the last chunk if there's anything left
        if current_chunk.strip() and len(current_paragraphs) > 0:
            chunk_text = current_chunk.strip()
            chunks.append({
                "id": self._generate_chunk_id(chunk_text, metadata["id"], position),
                "text": chunk_text,
                "metadata": {
                    "doc_id": metadata["id"],
                    "position": position,
                    "source": metadata.get("path", ""),
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", "unknown"),
                    "paragraphs": len(current_paragraphs)
                }
            })
        
        logger.info(f"Generated {len(chunks)} chunks using semantic chunking approach")
        return chunks
    
    def _split_paragraph_into_sentences(self, paragraph: str, doc_id: str, 
                                      start_position: int, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a long paragraph into sentence-based chunks.
        
        Args:
            paragraph: Long paragraph text
            doc_id: Document ID
            start_position: Starting position for chunks
            metadata: Document metadata
            
        Returns:
            List of chunks
        """
        # Use NLTK to split into sentences
        sentences = sent_tokenize(paragraph)
        
        chunks = []
        current_chunk = ""
        position = start_position
        
        for sentence in sentences:
            # If adding this sentence would exceed max size, finalize current chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                # Create chunk
                chunk_text = current_chunk.strip()
                chunks.append({
                    "id": self._generate_chunk_id(chunk_text, doc_id, position),
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "position": position,
                        "source": metadata.get("path", ""),
                        "title": metadata.get("title", ""),
                        "category": metadata.get("category", "unknown"),
                        "chunk_type": "sentence_group"
                    }
                })
                
                # Reset for next chunk
                current_chunk = ""
                position += 1
            
            # Add sentence to current chunk
            current_chunk += sentence + " "
        
        # Add the last chunk if there's anything left
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            chunks.append({
                "id": self._generate_chunk_id(chunk_text, doc_id, position),
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "position": position,
                    "source": metadata.get("path", ""),
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", "unknown"),
                    "chunk_type": "sentence_group"
                }
            })
        
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


class FixedSizeChunker(BaseChunker):
    """Chunker that creates fixed-size chunks with no sentence truncation."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in words
            overlap: Overlap between chunks in words
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text into fixed-size pieces.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        if not text:
            logger.warning("Empty text provided to chunker")
            return []
        
        # First split into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        position = 0
        
        for sentence in sentences:
            # Count words in this sentence
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence would exceed max size, finalize current chunk
            if current_word_count + sentence_word_count > self.chunk_size and current_word_count > 0:
                # Create chunk
                chunk_text = " ".join([" ".join(words) for words in current_chunk])
                chunks.append({
                    "id": self._generate_chunk_id(chunk_text, metadata["id"], position),
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": metadata["id"],
                        "position": position,
                        "source": metadata.get("path", ""),
                        "title": metadata.get("title", ""),
                        "category": metadata.get("category", "unknown"),
                        "word_count": current_word_count
                    }
                })
                
                # Keep overlap words for next chunk
                overlap_start = max(0, len(current_chunk) - 1)
                overlap_sentences = current_chunk[overlap_start:]
                
                # Reset for next chunk
                current_chunk = overlap_sentences
                current_word_count = sum(len(words) for words in overlap_sentences)
                position += 1
            
            # Add sentence to current chunk
            current_chunk.append(sentence_words)
            current_word_count += sentence_word_count
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = " ".join([" ".join(words) for words in current_chunk])
            chunks.append({
                "id": self._generate_chunk_id(chunk_text, metadata["id"], position),
                "text": chunk_text,
                "metadata": {
                    "doc_id": metadata["id"],
                    "position": position,
                    "source": metadata.get("path", ""),
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", "unknown"),
                    "word_count": current_word_count
                }
            })
        
        logger.info(f"Generated {len(chunks)} chunks using fixed-size chunking approach")
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
