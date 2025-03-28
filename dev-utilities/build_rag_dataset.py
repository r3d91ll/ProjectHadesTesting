#!/usr/bin/env python3
"""
Build RAG Dataset

A streamlined utility to build a RAG dataset using the fixed RAG dataset builder components.
This script bypasses the problematic areas while still leveraging the core functionality.
"""

import os
import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Setup paths
project_root = Path(__file__).parent.parent.absolute()
rag_builder_path = os.path.join(project_root, "rag-dataset-builder")

# Add rag-dataset-builder to Python path
sys.path.insert(0, rag_builder_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("build_rag_dataset")

# Import components we need
try:
    from src.processors import SimpleTextProcessor
    from src.chunkers import SemanticChunker
    from src.embedders import SentenceTransformerEmbedder
    from src.formatters import PathRAGFormatter
except ImportError as e:
    logger.error(f"Error importing RAG dataset builder components: {e}")
    sys.exit(1)

class RAGDatasetBuilder:
    """Simplified RAG dataset builder that bypasses problematic code."""
    
    def __init__(self, source_dir, output_dir, config=None):
        """Initialize the builder with source and output directories."""
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.processor = SimpleTextProcessor()
        self.chunker = SemanticChunker(
            max_chunk_size=self.config.get("chunker", {}).get("chunk_size", 1000),
            min_chunk_size=self.config.get("chunker", {}).get("min_chunk_size", 100)
        )
        self.embedder = SentenceTransformerEmbedder(
            model_name=self.config.get("embedder", {}).get("model_name", "all-MiniLM-L6-v2")
        )
        self.formatter = PathRAGFormatter(
            output_dir=self.output_dir
        )
        
        # Create checkpoint file
        self.checkpoint_file = os.path.join(self.output_dir, "checkpoint.json")
        self.checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint from file or create a new one."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Corrupted checkpoint file. Starting fresh.")
        
        return {
            "processed_files": [],
            "processed_chunks": 0,
            "total_chunks": 0,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _save_checkpoint(self):
        """Save checkpoint to file."""
        self.checkpoint["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def find_documents(self):
        """Find all documents in the source directory."""
        document_paths = []
        patterns = ["**/*.md", "**/*.txt", "**/*.pdf"]  # Add more as needed
        
        for pattern in patterns:
            matched_files = list(Path(self.source_dir).glob(pattern))
            document_paths.extend([str(f) for f in matched_files])
        
        # Filter out already processed files
        document_paths = [
            path for path in document_paths 
            if path not in self.checkpoint["processed_files"]
        ]
        
        logger.info(f"Found {len(document_paths)} unprocessed documents")
        return document_paths
    
    def process_document(self, doc_path):
        """Process a single document."""
        try:
            logger.info(f"Processing document: {doc_path}")
            
            # Process document - this part works as confirmed by our debug script
            doc_data = self.processor.process_document(doc_path)
            
            # Skip if processing failed
            if not doc_data or not doc_data.get("text"):
                logger.warning(f"Failed to process document: {doc_path}")
                return
            
            # Ensure document has an ID - this is redundant but just to be safe
            if "id" not in doc_data:
                doc_data["id"] = hashlib.md5(doc_path.encode()).hexdigest()
            
            # Generate chunks
            chunks = self.chunker.chunk_text(doc_data["text"], doc_data["metadata"])
            logger.info(f"Generated {len(chunks)} chunks from {doc_path}")
            
            # Skip if no chunks were generated
            if not chunks:
                logger.warning(f"No chunks generated from document: {doc_path}")
                return
            
            # Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(chunk_texts)
            
            # Format and save output
            self.formatter.format_output(chunks, embeddings, doc_data)
            
            # Update checkpoint
            self.checkpoint["processed_files"].append(doc_path)
            self.checkpoint["processed_chunks"] += len(chunks)
            self.checkpoint["total_chunks"] += len(chunks)
            self._save_checkpoint()
            
            logger.info(f"Successfully processed document: {doc_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return False
    
    def build_dataset(self):
        """Build the RAG dataset."""
        logger.info("Starting RAG dataset build")
        
        # Find documents
        document_paths = self.find_documents()
        if not document_paths:
            logger.info("No new documents to process")
            return
        
        # Process documents
        processed_count = 0
        for doc_path in document_paths:
            success = self.process_document(doc_path)
            if success:
                processed_count += 1
        
        # Finalize
        if hasattr(self.formatter, 'finalize'):
            self.formatter.finalize()
        
        logger.info(f"Dataset build complete. Processed {processed_count} documents with {self.checkpoint['processed_chunks']} chunks.")

def load_config(config_path):
    """Load configuration from file."""
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def main():
    """Main entry point."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Build a RAG dataset")
    parser.add_argument("--config", help="Path to configuration file", default=None)
    parser.add_argument("--source", help="Source directory", default=None)
    parser.add_argument("--output", help="Output directory", default=None)
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all documents")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before processing")
    args = parser.parse_args()
    
    # Set up paths
    source_dir = args.source or os.path.join(project_root, "source_documents")
    output_dir = args.output or os.path.join(project_root, "rag_databases", "current")
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    else:
        config_path = os.path.join(rag_builder_path, "config", "config.yaml")
        if os.path.exists(config_path):
            config = load_config(config_path)
    
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Handle force reprocessing
    if args.force:
        checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        if os.path.exists(checkpoint_file):
            logger.info(f"Resetting checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "processed_files": [],
                    "processed_chunks": 0,
                    "total_chunks": 0,
                    "last_updated": ""
                }, f, indent=2)
    
    # Handle clean option
    if args.clean:
        if os.path.exists(output_dir):
            logger.info(f"Cleaning output directory: {output_dir}")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
            logger.info("Output directory cleaned")
    
    # Build dataset
    builder = RAGDatasetBuilder(source_dir, output_dir, config)
    builder.build_dataset()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
