#!/usr/bin/env python3
"""
PathRAG Database Builder

This script processes collected datasets (papers, documentation, code samples)
and builds a PathRAG database incrementally with very low memory usage.
The script is designed to work with your specific research interests in
Actor-Network Theory, Code/Software Architecture, Machine Learning, etc.

It integrates with Arize Phoenix for performance tracking.
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
import random
import shutil
from typing import List, Dict, Any, Tuple, Optional, Generator
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import necessary for local embeddings
from sentence_transformers import SentenceTransformer
import torch
import requests
import networkx as nx
from tqdm import tqdm
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pathrag_db_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pathrag_db_builder")

# Load environment variables
load_dotenv()

# Import Arize Phoenix adapter (optional)
try:
    from pathrag_db_arize_adapter import PathRAGDBArizeAdapter
    ARIZE_AVAILABLE = True
except ImportError:
    ARIZE_AVAILABLE = False
    logger.warning("Arize Phoenix adapter not available. Performance tracking will be disabled.")

# Constants
CHUNK_SIZE = 300  # Smaller chunks for more detailed graph
CHUNK_OVERLAP = 50
BATCH_SIZE = 5  # Process documents in small batches to limit memory usage


class PathRAGDatabaseBuilder:
    """
    Low-memory implementation of PathRAG database builder.
    Processes documents incrementally and saves progress to disk.
    """
    
    def __init__(self, data_dir: str, output_dir: str, track_performance: bool = True):
        """
        Initialize the PathRAGDatabaseBuilder.
        
        Args:
            data_dir: Directory containing the collected datasets
            output_dir: Directory to save the PathRAG database
            track_performance: Whether to track performance with Arize Phoenix
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.chunks_dir = os.path.join(output_dir, "chunks")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.embeddings_dir = os.path.join(output_dir, "embeddings")
        self.graph_dir = os.path.join(output_dir, "graph")
        
        for directory in [self.chunks_dir, self.metadata_dir, 
                         self.embeddings_dir, self.graph_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up local embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load a model suitable for embeddings (smaller than BERT but still effective)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        logger.info(f"Loaded embedding model: all-MiniLM-L6-v2")
        
        # Set up Arize Phoenix tracking if available
        self.track_performance = track_performance and ARIZE_AVAILABLE
        if self.track_performance:
            try:
                self.arize_adapter = PathRAGDBArizeAdapter({
                    "phoenix_host": os.environ.get("PHOENIX_HOST", "localhost"),
                    "phoenix_port": int(os.environ.get("PHOENIX_PORT", "8080")),
                    "track_performance": True
                })
                logger.info("Connected to Arize Phoenix for performance tracking")
            except Exception as e:
                logger.error(f"Failed to initialize Arize Phoenix adapter: {e}")
                self.track_performance = False
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Checkpoint file for tracking progress
        self.checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        self.checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint from disk if it exists.
        
        Returns:
            Checkpoint data
        """
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted checkpoint file. Starting fresh.")
        
        # Initialize new checkpoint
        return {
            "processed_files": [],
            "processed_chunks": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        self.checkpoint["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def _generate_chunk_id(self, text: str, source: str) -> str:
        """
        Generate a unique ID for a chunk based on its content and source.
        
        Args:
            text: Chunk text
            source: Source document
            
        Returns:
            Chunk ID
        """
        hash_input = f"{source}:{text}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def find_all_documents(self) -> List[str]:
        """
        Find all documents in the data directory.
        
        Returns:
            List of document paths
        """
        document_paths = []
        
        # Find all PDF files
        pdf_files = glob.glob(os.path.join(self.data_dir, "**/*.pdf"), recursive=True)
        document_paths.extend(pdf_files)
        
        # Find all text files
        text_files = glob.glob(os.path.join(self.data_dir, "**/*.txt"), recursive=True)
        document_paths.extend(text_files)
        
        # Filter out already processed files
        document_paths = [
            path for path in document_paths 
            if path not in self.checkpoint["processed_files"]
        ]
        
        logger.info(f"Found {len(document_paths)} unprocessed documents")
        return document_paths
    
    def process_documents_in_batches(self, document_paths: List[str], batch_size: int = BATCH_SIZE):
        """
        Process documents in small batches to limit memory usage.
        
        Args:
            document_paths: List of document paths
            batch_size: Number of documents to process in each batch
        """
        total_documents = len(document_paths)
        
        for i in range(0, total_documents, batch_size):
            batch = document_paths[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_documents + batch_size - 1)//batch_size}")
            
            for doc_path in batch:
                try:
                    self.process_document(doc_path)
                    self.checkpoint["processed_files"].append(doc_path)
                    self._save_checkpoint()
                except Exception as e:
                    logger.error(f"Error processing document {doc_path}: {e}")
            
            # Force garbage collection after each batch
            gc.collect()
    
    def process_document(self, doc_path: str):
        """
        Process a single document.
        
        Args:
            doc_path: Path to the document
        """
        logger.info(f"Processing document: {doc_path}")
        
        # Initialize tracking if available
        tracking_info = None
        if self.track_performance:
            tracking_info = self.arize_adapter.track_document_processing(doc_path)
        
        try:
            # Extract text from document
            text = self._extract_text(doc_path)
            if not text:
                logger.warning(f"Failed to extract text from {doc_path}")
                if self.track_performance and tracking_info:
                    self.arize_adapter.complete_document_tracking(tracking_info, 0, 0)
                return
            
            # Get document metadata
            metadata = self._extract_metadata(doc_path, text)
            
            # Generate chunks
            chunks = self._chunk_text(text, metadata)
            logger.info(f"Generated {len(chunks)} chunks from {doc_path}")
            
            # Save chunks to disk
            for chunk in chunks:
                chunk_id = chunk["id"]
                chunk_file = os.path.join(self.chunks_dir, f"{chunk_id}.json")
                with open(chunk_file, 'w') as f:
                    json.dump(chunk, f, indent=2)
            
            # Save document metadata
            doc_id = os.path.basename(doc_path)
            metadata_file = os.path.join(self.metadata_dir, f"{doc_id}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update checkpoint
            self.checkpoint["processed_chunks"] += len(chunks)
            self.checkpoint["total_chunks"] += len(chunks)
            
            # Process embeddings for chunks
            self._process_embeddings_for_chunks(chunks)
            
            # Update graph with chunks and relationships
            self._update_graph_with_chunks(chunks, metadata)
            
            # Save graph
            self._save_graph()
            
            # Complete tracking if available
            if self.track_performance and tracking_info:
                # Calculate total output size
                output_size = sum(os.path.getsize(os.path.join(self.chunks_dir, f"{chunk['id']}.json")) 
                               for chunk in chunks)
                self.arize_adapter.complete_document_tracking(tracking_info, len(chunks), output_size)
                
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            # Complete tracking with error if available
            if self.track_performance and tracking_info:
                self.arize_adapter.complete_document_tracking(tracking_info, 0, 0)
            raise  # Re-raise the exception
    
    def _extract_text(self, doc_path: str) -> str:
        """
        Extract text from a document.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Extracted text
        """
        # Handle PDF files
        if doc_path.endswith('.pdf'):
            # Use pdftotext if available
            try:
                import subprocess
                result = subprocess.run(
                    ['pdftotext', '-layout', doc_path, '-'],
                    capture_output=True, text=True, check=True
                )
                return result.stdout
            except (subprocess.SubprocessError, ImportError):
                logger.warning(f"Failed to extract text from PDF {doc_path}")
                return ""
        
        # Handle text files
        elif doc_path.endswith('.txt'):
            try:
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read text file {doc_path}: {e}")
                return ""
        
        # Unsupported file type
        else:
            logger.warning(f"Unsupported file type: {doc_path}")
            return ""
    
    def _extract_metadata(self, doc_path: str, text: str) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            doc_path: Path to the document
            text: Document text
            
        Returns:
            Document metadata
        """
        filename = os.path.basename(doc_path)
        file_path = doc_path
        
        # Try to extract title from text
        title = filename
        title_match = re.search(r'^(?:#|Title:)\s*(.+?)$', text, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        
        # Determine category based on path
        category = "unknown"
        for category_name in ["actor_network_theory", "sts_digital_sociology", 
                            "knowledge_graphs_retrieval", "computational_linguistics",
                            "ethics_bias_ai", "graph_reasoning_ml", 
                            "semiotics_linguistic_anthropology"]:
            if category_name in doc_path:
                category = category_name
                break
        
        # Extract document type
        doc_type = "unknown"
        if "papers" in doc_path:
            doc_type = "research_paper"
        elif "documentation" in doc_path:
            doc_type = "documentation"
        elif "code_samples" in doc_path:
            doc_type = "code"
        
        return {
            "id": filename,
            "title": title,
            "path": file_path,
            "category": category,
            "doc_type": doc_type,
            "creation_time": datetime.now().isoformat(),
            "file_size": os.path.getsize(doc_path),
            "character_count": len(text)
        }
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        # Split text into smaller chunks
        words = text.split()
        chunks = []
        
        doc_id = metadata["id"]
        doc_path = metadata["path"]
        doc_title = metadata["title"]
        doc_category = metadata["category"]
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            # Get chunk words with overlap
            end_idx = min(i + CHUNK_SIZE, len(words))
            chunk_words = words[i:end_idx]
            chunk_text = " ".join(chunk_words)
            
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(chunk_text, doc_id)
            
            # Create chunk
            chunk = {
                "id": chunk_id,
                "text": chunk_text,
                "start_idx": i,
                "end_idx": end_idx,
                "doc_id": doc_id,
                "doc_path": doc_path,
                "doc_title": doc_title,
                "doc_category": doc_category,
                "position": len(chunks),  # Position in the document
                "creation_time": datetime.now().isoformat()
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _process_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Generate embeddings for chunks using local sentence-transformers model.
        
        Args:
            chunks: List of chunks
        """
        # Process in batches to manage memory usage
        batch_size = 32  # Larger batch size is fine for local processing
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:min(i+batch_size, len(chunks))]
            
            # Initialize tracking if available
            tracking_info = None
            if self.track_performance:
                texts = [chunk["text"] for chunk in batch]
                tracking_info = self.arize_adapter.track_embedding_generation(
                    texts=texts,
                    model_name="all-MiniLM-L6-v2",
                    embedding_dim=self.embedding_model.get_sentence_embedding_dimension(),
                    batch_size=batch_size
                )
            
            try:
                # Extract texts
                texts = [chunk["text"] for chunk in batch]
                
                # Generate embeddings using sentence-transformers (returns numpy array)
                with torch.no_grad():
                    embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
                
                # Save embeddings
                for j, chunk in enumerate(batch):
                    chunk_id = chunk["id"]
                    embedding = embeddings[j].tolist()  # Convert numpy array to list for JSON serialization
                    
                    # Save embedding to disk
                    embedding_file = os.path.join(self.embeddings_dir, f"{chunk_id}.json")
                    with open(embedding_file, 'w') as f:
                        json.dump({
                            "id": chunk_id,
                            "embedding": embedding
                        }, f)
                
                logger.info(f"Generated embeddings for {len(batch)} chunks using local model")
                
                # Complete tracking if available
                if self.track_performance and tracking_info:
                    self.arize_adapter.complete_embedding_tracking(tracking_info)
                
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Complete tracking with error if available
                if self.track_performance and tracking_info:
                    self.arize_adapter.complete_embedding_tracking(tracking_info)
                raise  # Re-raise the exception for debugging
    
    def _update_graph_with_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """
        Update graph with chunks and relationships.
        
        Args:
            chunks: List of chunks
            metadata: Document metadata
        """
        # Initialize tracking if available
        tracking_info = None
        if self.track_performance:
            tracking_info = self.arize_adapter.track_graph_update(
                num_nodes=self.graph.number_of_nodes(),
                num_edges=self.graph.number_of_edges()
            )
        
        # Count new entities and relations
        initial_nodes = self.graph.number_of_nodes()
        initial_edges = self.graph.number_of_edges()
        
        doc_id = metadata["id"]
        doc_title = metadata["title"]
        doc_category = metadata["category"]
        
        # Add document node
        self.graph.add_node(
            doc_id,
            type="document",
            title=doc_title,
            category=doc_category,
            path=metadata["path"]
        )
        
        # Add chunks as nodes
        for chunk in chunks:
            chunk_id = chunk["id"]
            
            # Add chunk node
            self.graph.add_node(
                chunk_id,
                type="chunk",
                text=chunk["text"][:100] + "...",  # Short preview
                position=chunk["position"],
                doc_id=doc_id
            )
            
            # Add edge from document to chunk
            self.graph.add_edge(
                doc_id, chunk_id,
                type="contains",
                weight=1.0
            )
            
            # Add neighboring chunk relationships
            if chunk["position"] > 0:
                prev_chunk_id = chunks[chunk["position"] - 1]["id"]
                self.graph.add_edge(
                    prev_chunk_id, chunk_id,
                    type="next",
                    weight=1.0
                )
        
        # Add category node if it doesn't exist
        if not self.graph.has_node(doc_category):
            self.graph.add_node(
                doc_category,
                type="category"
            )
        
        # Add edge from category to document
        self.graph.add_edge(
            doc_category, doc_id,
            type="contains",
            weight=1.0
        )
        
        # Calculate new nodes and edges
        new_nodes = self.graph.number_of_nodes() - initial_nodes
        new_edges = self.graph.number_of_edges() - initial_edges
        
        # Update checkpoint
        self.checkpoint["total_entities"] = self.graph.number_of_nodes()
        self.checkpoint["total_relationships"] = self.graph.number_of_edges()
        
        # Complete tracking if available
        if self.track_performance and tracking_info:
            self.arize_adapter.complete_graph_tracking(tracking_info, new_nodes, new_edges)
    
    def _save_graph(self):
        """Save graph to disk."""
        # Initialize tracking if available
        tracking_info = None
        if self.track_performance:
            tracking_info = self.arize_adapter.track_graph_update(
                num_nodes=self.graph.number_of_nodes(),
                num_edges=self.graph.number_of_edges(),
                operation="save"
            )
        
        # Save graph using pickle instead of write_gpickle
        graph_file = os.path.join(self.graph_dir, "knowledge_graph.pickle")
        import pickle
        with open(graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
            
        # Complete tracking if available
        if self.track_performance and tracking_info:
            self.arize_adapter.complete_graph_tracking(tracking_info, 0, 0)
        
        # Also save as JSON for easier inspection
        graph_json = os.path.join(self.graph_dir, "knowledge_graph.json")
        
        # Convert to serializable format
        graph_data = {
            "nodes": [
                {
                    "id": node,
                    **attr
                }
                for node, attr in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **attr
                }
                for u, v, attr in self.graph.edges(data=True)
            ]
        }
        
        with open(graph_json, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def build_database(self):
        """Build the PathRAG database."""
        # Find all documents
        document_paths = self.find_all_documents()
        
        if not document_paths:
            logger.info("No new documents to process")
            return
        
        # Process documents in batches
        self.process_documents_in_batches(document_paths)
        
        # Final save
        self._save_checkpoint()
        self._save_graph()
        
        logger.info(f"Database build complete. Processed {len(self.checkpoint['processed_files'])} documents "
                   f"with {self.checkpoint['total_chunks']} chunks, "
                   f"{self.checkpoint['total_entities']} entities, and "
                   f"{self.checkpoint['total_relationships']} relationships.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build PathRAG database with low memory usage")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing the collected datasets")
    parser.add_argument("--output-dir", type=str, default="./pathrag_database",
                        help="Directory to save the PathRAG database")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of documents to process in each batch")
    parser.add_argument("--clean", action="store_true",
                        help="Clean existing database before building")
    parser.add_argument("--track-performance", action="store_true", default=True,
                        help="Track performance with Arize Phoenix")
    
    args = parser.parse_args()
    
    # Clean database if requested
    if args.clean and os.path.exists(args.output_dir):
        logger.info(f"Cleaning existing database at {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # Build database
    builder = PathRAGDatabaseBuilder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        track_performance=args.track_performance
    )
    
    logger.info(f"Building PathRAG database from {args.data_dir} to {args.output_dir}")
    builder.build_database()


if __name__ == "__main__":
    main()
