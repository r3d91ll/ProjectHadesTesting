#!/usr/bin/env python3
"""
Direct RAG Dataset Builder

A streamlined utility that directly builds a RAG dataset without relying on the main builder.
This script bypasses all the problematic code and directly uses the components we know work.
"""

import os
import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path
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
logger = logging.getLogger("direct_dataset_builder")

# Import components
try:
    # Import directly from source - avoid the main builder
    from src.processors import SimpleTextProcessor
    from src.chunkers import SemanticChunker
    from src.embedders import SentenceTransformerEmbedder
    from src.formatters import PathRAGFormatter
except ImportError as e:
    logger.error(f"Error importing RAG dataset builder components: {e}")
    sys.exit(1)

def find_documents(source_dir):
    """Find all documents in the source directory."""
    document_paths = []
    
    # List of file extensions to include
    patterns = ["**/*.md", "**/*.txt", "**/*.pdf", "**/*.py", "**/*.js", "**/*.java"]
    
    for pattern in patterns:
        logger.info(f"Searching for pattern: {pattern}")
        found_paths = list(Path(source_dir).glob(pattern))
        logger.info(f"Found {len(found_paths)} files matching pattern {pattern}")
        document_paths.extend([str(p) for p in found_paths])
    
    logger.info(f"Found {len(document_paths)} total documents")
    return document_paths

def process_documents(source_dir, output_dir, force=False, clean=False):
    """Process all documents in the source directory and generate a RAG dataset."""
    logger.info(f"Processing documents from {source_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean output directory if requested
    if clean:
        logger.info(f"Cleaning output directory: {output_dir}")
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            else:
                import shutil
                shutil.rmtree(item_path)
        logger.info("Output directory cleaned")
    
    # Initialize components
    processor = SimpleTextProcessor()
    chunker = SemanticChunker(max_chunk_size=1000, min_chunk_size=100)
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    formatter = PathRAGFormatter(output_dir=output_dir)
    
    # Find documents
    document_paths = find_documents(source_dir)
    if not document_paths:
        logger.info("No documents found")
        return False
    
    # Process each document
    all_chunks = []
    all_embeddings = []
    processed_count = 0
    
    for doc_path in document_paths:
        logger.info(f"Processing document: {doc_path}")
        
        try:
            # Process document
            doc_data = processor.process_document(doc_path)
            
            # Print diagnostic info
            if 'id' in doc_data:
                logger.info(f"Document ID: {doc_data['id']}")
            else:
                logger.warning(f"No ID found in document data, adding one")
                doc_data['id'] = hashlib.md5(doc_path.encode()).hexdigest()
            
            # Generate chunks
            chunks = chunker.chunk_text(doc_data["text"], doc_data["metadata"])
            logger.info(f"Generated {len(chunks)} chunks")
            
            # Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embedder.generate_embeddings(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Add to collection
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            
            # Format output for this document
            formatter.format_output(chunks, embeddings, doc_data)
            
            logger.info(f"Successfully processed document: {doc_path}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
    
    # Finalize
    logger.info("Finalizing output...")
    formatter.finalize()
    
    logger.info(f"Processing complete. Processed {processed_count} documents with {len(all_chunks)} chunks.")
    return processed_count > 0

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Direct RAG Dataset Builder")
    parser.add_argument("--source", help="Source directory", default=None)
    parser.add_argument("--output", help="Output directory", default=None)
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all documents")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before processing")
    args = parser.parse_args()
    
    # Set up paths
    source_dir = args.source or os.path.join(project_root, "source_documents")
    output_dir = args.output or os.path.join(project_root, "rag_databases", "current")
    
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process documents
    success = process_documents(source_dir, output_dir, args.force, args.clean)
    
    if success:
        logger.info("RAG dataset build completed successfully")
        return 0
    else:
        logger.warning("RAG dataset build completed with warnings")
        return 1

if __name__ == "__main__":
    sys.exit(main())
