#!/usr/bin/env python3
"""
Ultra-Minimal PathRAG Database Builder

This script creates a basic PathRAG database structure by processing documents 
in tiny batches with minimal memory footprint. It avoids any operations that
would consume large amounts of memory.
"""

import os
import sys
import json
import logging
import gc
import time
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f'pathrag_minimal_{time.strftime("%Y%m%d_%H%M%S")}.log')
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(os.path.dirname(BASE_DIR), "docs")
DATA_DIR = os.path.join(BASE_DIR, "data")

def setup_database(db_path: str, clean: bool = False):
    """Set up the database directory structure."""
    if clean and os.path.exists(db_path):
        logger.info(f"Cleaning existing database at {db_path}")
        import shutil
        shutil.rmtree(db_path)
    
    # Create database directory structure
    os.makedirs(db_path, exist_ok=True)
    os.makedirs(os.path.join(db_path, "documents"), exist_ok=True)
    os.makedirs(os.path.join(db_path, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(db_path, "graph"), exist_ok=True)
    
    # Initialize files
    edges_file = os.path.join(db_path, "graph", "edges.jsonl")
    nodes_file = os.path.join(db_path, "graph", "nodes.jsonl")
    
    if not os.path.exists(edges_file):
        with open(edges_file, 'w') as f:
            pass
    
    if not os.path.exists(nodes_file):
        with open(nodes_file, 'w') as f:
            pass
    
    return {
        "documents_dir": os.path.join(db_path, "documents"),
        "chunks_dir": os.path.join(db_path, "chunks"),
        "graph_dir": os.path.join(db_path, "graph"),
        "edges_file": edges_file,
        "nodes_file": nodes_file
    }

def find_documents(docs_dir: str, extensions: List[str] = ['.md', '.txt']) -> List[str]:
    """Find all documents with the specified extensions."""
    documents = []
    for ext in extensions:
        for file_path in Path(docs_dir).rglob(f"*{ext}"):
            documents.append(str(file_path))
    return documents

def extract_document_metadata(file_path: str) -> Dict[str, Any]:
    """Extract basic metadata from a document without loading its full content."""
    try:
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        
        # For very large files, just get the metadata
        if file_size > 1000000:  # 1MB
            return {
                "file_path": file_path,
                "file_name": file_name,
                "file_ext": file_ext,
                "file_size": file_size,
                "title": os.path.splitext(file_name)[0].replace('_', ' ').title(),
                "too_large": True
            }
        
        # For smaller files, extract a title from content
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = []
            for _ in range(10):  # Read just first 10 lines
                line = f.readline()
                if not line:
                    break
                first_lines.append(line)
        
        title = os.path.splitext(file_name)[0].replace('_', ' ').title()
        
        # Try to extract title from markdown heading
        if file_ext == '.md':
            for line in first_lines:
                if line.startswith('# '):
                    title = line.replace('# ', '').strip()
                    break
        
        return {
            "file_path": file_path,
            "file_name": file_name,
            "file_ext": file_ext,
            "file_size": file_size,
            "title": title,
            "too_large": False
        }
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}")
        return None

def chunk_document(file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chunk a document into smaller pieces with minimal memory usage.
    Read and process the file in small chunks rather than loading it all at once.
    """
    chunk_data = []
    doc_id = str(uuid.uuid4())
    
    try:
        file_size = os.path.getsize(file_path)
        
        # Skip extremely large files
        if file_size > 5000000:  # 5MB
            logger.warning(f"Skipping large file: {file_path} ({file_size} bytes)")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get document metadata
        file_name = os.path.basename(file_path)
        title = os.path.splitext(file_name)[0].replace('_', ' ').title()
        
        # Try to extract title from markdown heading
        if file_path.endswith('.md'):
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    title = line.replace('# ', '').strip()
                    break
        
        # Split content into chunks
        chunks = []
        
        if len(content) <= chunk_size:
            chunks.append(content)
        else:
            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                
                # Try to find a good split point
                if end < len(content):
                    candidates = [
                        content.rfind('. ', start, end),
                        content.rfind('\n', start, end),
                        content.rfind(' ', start, end)
                    ]
                    split_point = max(filter(lambda x: x > start, candidates), default=end)
                    if split_point > start:
                        end = split_point + 1
                
                chunks.append(content[start:end])
                start = end - overlap
                
                # Explicit garbage collection
                gc.collect()
        
        # Create chunk data
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_data.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "index": i,
                "total_chunks": len(chunks),
                "text": chunk_text,
                "title": title,
                "file_path": file_path
            })
            
            # Explicit garbage collection after each chunk
            gc.collect()
        
        # Free memory
        del content
        del chunks
        gc.collect()
        
        return chunk_data
    
    except Exception as e:
        logger.error(f"Error chunking document {file_path}: {e}")
        return []

def save_document_metadata(doc_id: str, metadata: Dict[str, Any], documents_dir: str):
    """Save document metadata to a file."""
    output_path = os.path.join(documents_dir, f"{doc_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

def save_chunk(chunk: Dict[str, Any], chunks_dir: str):
    """Save a chunk to a file."""
    chunk_id = chunk["chunk_id"]
    
    # Save chunk text
    with open(os.path.join(chunks_dir, f"{chunk_id}.txt"), 'w', encoding='utf-8') as f:
        f.write(chunk["text"])
    
    # Save chunk metadata (without the text to save space)
    chunk_metadata = {k: v for k, v in chunk.items() if k != "text"}
    with open(os.path.join(chunks_dir, f"{chunk_id}.json"), 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, indent=2)

def add_node(node_id: str, metadata: Dict[str, Any], nodes_file: str):
    """Add a node to the graph."""
    node_data = {"id": node_id, "metadata": metadata}
    with open(nodes_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(node_data) + '\n')

def add_edge(source: str, target: str, edge_type: str, weight: float, edges_file: str):
    """Add an edge to the graph."""
    edge_data = {
        "source": source,
        "target": target,
        "type": edge_type,
        "weight": weight
    }
    with open(edges_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(edge_data) + '\n')

def create_qa_pairs(docs_dir: str, output_path: str):
    """Create sample QA pairs from documents with minimal memory usage."""
    qa_pairs = []
    
    # Find documents
    documents = find_documents(docs_dir)
    
    for doc_path in documents:
        try:
            # Get basic metadata
            metadata = extract_document_metadata(doc_path)
            if not metadata:
                continue
            
            title = metadata["title"]
            
            # For markdown files, try to extract some content for QA pairs
            if doc_path.endswith('.md') and not metadata.get("too_large", False):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read(10000)  # Read only first 10KB
                
                # Create a simple QA pair
                qa_pairs.append({
                    "question": f"What is {title} about?",
                    "answer": f"{title} is about {' '.join(content.split()[:50])}...",
                    "document": doc_path
                })
                
                # Try to extract sections for more QA pairs
                lines = content.split('\n')
                sections = []
                current_section = {"title": "", "content": []}
                
                for line in lines:
                    if line.startswith('## '):
                        # Save previous section
                        if current_section["content"]:
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            "title": line.replace('## ', '').strip(),
                            "content": []
                        }
                    elif line.startswith('# '):
                        # Document title, skip
                        continue
                    else:
                        # Add to current section
                        current_section["content"].append(line)
                
                # Add last section
                if current_section["content"]:
                    sections.append(current_section)
                
                # Create QA pairs for sections
                for section in sections:
                    if len(section["content"]) < 3:
                        continue
                    
                    section_content = "\n".join(section["content"])
                    qa_pairs.append({
                        "question": f"What does the section '{section['title']}' in {title} describe?",
                        "answer": f"The section '{section['title']}' in {title} describes {' '.join(section_content.split()[:30])}...",
                        "document": doc_path,
                        "section": section["title"]
                    })
            else:
                # For non-markdown or large files, just create a simple question
                qa_pairs.append({
                    "question": f"What is {title}?",
                    "answer": f"{title} is a document located at {doc_path}.",
                    "document": doc_path
                })
            
            # Explicit garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error creating QA pairs for {doc_path}: {e}")
    
    # Save QA pairs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2)
    
    logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
    return qa_pairs

def create_checkpoint(db_path: str, processed_docs: List[str]):
    """Create a checkpoint file to track progress."""
    checkpoint_path = os.path.join(db_path, "checkpoint.json")
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.time(),
            "processed_docs": processed_docs
        }, f, indent=2)

def load_checkpoint(db_path: str) -> List[str]:
    """Load checkpoint data if it exists."""
    checkpoint_path = os.path.join(db_path, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("processed_docs", [])
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    return []

def main():
    """Main function to build a minimal PathRAG database."""
    parser = argparse.ArgumentParser(description="Ultra-Minimal PathRAG Database Builder")
    parser.add_argument("--docs-dir", type=str, default=DOCS_DIR, help="Directory containing documents")
    parser.add_argument("--db-path", type=str, default=os.path.join(DATA_DIR, "minimal_pathrag"), help="Path to database")
    parser.add_argument("--qa-pairs", type=str, default=os.path.join(DATA_DIR, "qa_pairs.json"), help="Path for QA pairs")
    parser.add_argument("--chunk-size", type=int, default=300, help="Maximum chunk size")
    parser.add_argument("--overlap", type=int, default=30, help="Chunk overlap size")
    parser.add_argument("--clean", action="store_true", help="Clean existing database")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Set up database structure
    logger.info(f"Setting up database at {args.db_path}")
    db_paths = setup_database(args.db_path, args.clean and not args.resume)
    
    # Create QA pairs
    logger.info("Creating QA pairs")
    create_qa_pairs(args.docs_dir, args.qa_pairs)
    
    # Find documents
    logger.info(f"Finding documents in {args.docs_dir}")
    documents = find_documents(args.docs_dir)
    logger.info(f"Found {len(documents)} documents")
    
    # Load checkpoint if resuming
    processed_docs = []
    if args.resume:
        processed_docs = load_checkpoint(args.db_path)
        logger.info(f"Resuming from checkpoint, {len(processed_docs)} documents already processed")
        
        # Filter out already processed documents
        documents = [doc for doc in documents if doc not in processed_docs]
        logger.info(f"{len(documents)} documents remaining to process")
    
    # Process documents
    total_chunks = 0
    total_edges = 0
    
    for i, doc_path in enumerate(documents):
        try:
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc_path}")
            
            # Extract document metadata
            metadata = extract_document_metadata(doc_path)
            if not metadata:
                logger.warning(f"Skipping document with no metadata: {doc_path}")
                continue
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Save document metadata
            metadata["doc_id"] = doc_id
            save_document_metadata(doc_id, metadata, db_paths["documents_dir"])
            
            # Chunk document
            logger.info(f"Chunking document: {metadata['title']}")
            chunks = chunk_document(
                doc_path, 
                chunk_size=args.chunk_size, 
                overlap=args.overlap
            )
            
            if not chunks:
                logger.warning(f"No chunks created for document: {doc_path}")
                continue
                
            logger.info(f"Created {len(chunks)} chunks")
            
            # Process chunks
            chunk_ids = []
            for chunk in chunks:
                # Save chunk
                save_chunk(chunk, db_paths["chunks_dir"])
                
                # Add node to graph
                node_metadata = {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["index"],
                    "total_chunks": chunk["total_chunks"],
                    "title": metadata["title"],
                    "file_path": doc_path
                }
                add_node(chunk["chunk_id"], node_metadata, db_paths["nodes_file"])
                
                # Keep track of chunk IDs
                chunk_ids.append(chunk["chunk_id"])
                
                # Explicit garbage collection
                del chunk
                gc.collect()
            
            # Create edges between consecutive chunks
            for j in range(len(chunk_ids) - 1):
                add_edge(
                    chunk_ids[j], 
                    chunk_ids[j + 1], 
                    "sequential", 
                    1.0,
                    db_paths["edges_file"]
                )
                total_edges += 1
            
            total_chunks += len(chunk_ids)
            
            # Update processed documents
            processed_docs.append(doc_path)
            
            # Create checkpoint every few documents
            if (i + 1) % 5 == 0 or i == len(documents) - 1:
                create_checkpoint(args.db_path, processed_docs)
                logger.info(f"Checkpoint created after {len(processed_docs)} documents")
            
            # Explicit garbage collection
            del chunks
            del chunk_ids
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
    
    # Final stats
    logger.info(f"PathRAG database creation complete")
    logger.info(f"Total documents processed: {len(processed_docs)}")
    logger.info(f"Total chunks created: {total_chunks}")
    logger.info(f"Total edges created: {total_edges}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
