#!/usr/bin/env python3
"""
Minimal RAG Dataset Builder

A bare-minimum implementation that directly processes markdown files and 
generates a RAG dataset without relying on potentially buggy components.
"""

import os
import sys
import json
import hashlib
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

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
logger = logging.getLogger("minimal_dataset_builder")

def directly_process_document(doc_path):
    """Process a document directly, without using the processor classes."""
    try:
        # Read text file
        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Extract basic metadata
        filename = os.path.basename(doc_path)
        file_size = os.path.getsize(doc_path)
        
        # Generate ID
        doc_id = hashlib.md5(doc_path.encode()).hexdigest()
        
        # Try to extract title from text
        title = filename
        title_match = re.search(r'^(?:#|Title:)\s*(.+?)$', text, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        
        # Create document data
        doc_data = {
            "id": doc_id,
            "text": text,
            "metadata": {
                "filename": filename,
                "title": title,
                "path": doc_path,
                "file_size": file_size,
                "category": "documentation",
                "extension": os.path.splitext(filename)[1],
                "created_at": datetime.fromtimestamp(os.path.getctime(doc_path)).isoformat(),
                "modified_at": datetime.fromtimestamp(os.path.getmtime(doc_path)).isoformat(),
                "character_count": len(text)
            }
        }
        
        logger.info(f"Successfully processed document: {filename}")
        logger.info(f"Document ID: {doc_id}")
        return doc_data
    
    except Exception as e:
        logger.error(f"Error processing document {doc_path}: {e}")
        return None

def simple_chunk_text(text, metadata, chunk_size=1000, overlap=200):
    """Simple text chunking without using the chunker classes."""
    chunks = []
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    # Initialize the first chunk
    current_chunk = ""
    current_chunk_id = 0
    
    for para in paragraphs:
        # If adding this paragraph would exceed the chunk size, save current chunk and start a new one
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            # Create chunk
            chunk = {
                "id": f"{metadata.get('id', 'doc')}_chunk_{current_chunk_id}",
                "text": current_chunk.strip(),
                "metadata": {
                    **metadata,
                    "chunk_id": current_chunk_id,
                    "chunk_text_length": len(current_chunk)
                }
            }
            chunks.append(chunk)
            
            # Start new chunk with overlap
            words = current_chunk.split()
            overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
            current_chunk = overlap_text + '\n\n' + para
            current_chunk_id += 1
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += '\n\n' + para
            else:
                current_chunk = para
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunk = {
            "id": f"{metadata.get('id', 'doc')}_chunk_{current_chunk_id}",
            "text": current_chunk.strip(),
            "metadata": {
                **metadata,
                "chunk_id": current_chunk_id,
                "chunk_text_length": len(current_chunk)
            }
        }
        chunks.append(chunk)
    
    logger.info(f"Generated {len(chunks)} chunks")
    return chunks

def simple_pathrag_output(chunks, output_dir):
    """Generate a simple PathRAG output without using the formatter classes."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chunks to JSON file
    chunks_file = os.path.join(output_dir, "chunks.json")
    with open(chunks_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")
    
    # Create a simple knowledge graph structure
    nodes = []
    edges = []
    
    # Create nodes for each chunk
    for chunk in chunks:
        node = {
            "id": chunk["id"],
            "type": "chunk",
            "content": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
            "metadata": chunk["metadata"]
        }
        nodes.append(node)
    
    # Create document nodes and connect them to chunks
    doc_ids = set()
    for chunk in chunks:
        doc_id = chunk["metadata"].get("id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.add(doc_id)
            
            # Create document node
            doc_node = {
                "id": doc_id,
                "type": "document",
                "content": chunk["metadata"].get("title", ""),
                "metadata": {k: v for k, v in chunk["metadata"].items() if k != "chunk_id" and k != "chunk_text_length"}
            }
            nodes.append(doc_node)
            
            # Create edge from document to chunk
            edge = {
                "source": doc_id,
                "target": chunk["id"],
                "type": "contains"
            }
            edges.append(edge)
    
    # Save knowledge graph to JSON file
    graph_file = os.path.join(output_dir, "knowledge_graph.json")
    with open(graph_file, 'w') as f:
        json.dump({
            "nodes": nodes,
            "edges": edges
        }, f, indent=2)
    
    logger.info(f"Saved knowledge graph with {len(nodes)} nodes and {len(edges)} edges to {graph_file}")
    
    return True

def find_documents(source_dir):
    """Find all documents in the source directory."""
    document_paths = []
    
    # List of file extensions to include
    patterns = ["**/*.md", "**/*.txt"]
    
    for pattern in patterns:
        logger.info(f"Searching for pattern: {pattern}")
        found_paths = list(Path(source_dir).glob(pattern))
        logger.info(f"Found {len(found_paths)} files matching pattern {pattern}")
        document_paths.extend([str(p) for p in found_paths])
    
    logger.info(f"Found {len(document_paths)} total documents")
    return document_paths

def build_minimal_dataset(source_dir, output_dir, force=False, clean=False):
    """Build a minimal RAG dataset directly."""
    logger.info(f"Building minimal RAG dataset from {source_dir} to {output_dir}")
    
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
    
    # Find documents
    document_paths = find_documents(source_dir)
    if not document_paths:
        logger.info("No documents found")
        return False
    
    # Process documents and collect all chunks
    all_chunks = []
    processed_count = 0
    
    for doc_path in document_paths:
        logger.info(f"Processing document: {doc_path}")
        
        # Directly process document
        doc_data = directly_process_document(doc_path)
        if not doc_data:
            logger.warning(f"Skipping document: {doc_path}")
            continue
        
        # Create chunks
        chunks = simple_chunk_text(doc_data["text"], doc_data["metadata"])
        
        # Add document ID to each chunk's metadata
        for chunk in chunks:
            chunk["metadata"]["document_id"] = doc_data["id"]
        
        # Add to all chunks
        all_chunks.extend(chunks)
        processed_count += 1
    
    # Generate output
    if all_chunks:
        simple_pathrag_output(all_chunks, output_dir)
    
    logger.info(f"Processing complete. Processed {processed_count} documents with {len(all_chunks)} chunks.")
    return processed_count > 0

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Minimal RAG Dataset Builder")
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
    
    # Build dataset
    success = build_minimal_dataset(source_dir, output_dir, args.force, args.clean)
    
    if success:
        logger.info("RAG dataset build completed successfully")
        return 0
    else:
        logger.warning("RAG dataset build completed with warnings")
        return 1

if __name__ == "__main__":
    sys.exit(main())
