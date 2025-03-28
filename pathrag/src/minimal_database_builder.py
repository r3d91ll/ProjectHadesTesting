#!/usr/bin/env python3
"""
Minimal Memory PathRAG Database Preparation Script

This script prepares a PathRAG database by ingesting documents from the
HADES repository with minimal memory usage and no large models loaded.
It uses a file-based approach and processes one document at a time.
"""

import os
import sys
import json
import logging
import argparse
import time
import uuid
import gc
import shutil
from pathlib import Path
from typing import List, Dict, Any, Generator

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f'pathrag_db_prep_{time.strftime("%Y%m%d_%H%M%S")}.log')
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "document_store")

# Import minimal required libraries
try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Failed to import required library: {e}")
    logger.error("Please install the required libraries with:")
    logger.error("pip install python-dotenv requests")
    sys.exit(1)

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
    sys.exit(1)

class MinimalPathRAGBuilder:
    """
    Ultra-minimal PathRAG database builder with focus on memory efficiency.
    No large models loaded in memory, uses disk operations extensively.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the minimal PathRAG database builder."""
        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.db_path = config.get("db_path", DB_PATH)
        self.graph_edges_file = os.path.join(self.db_path, "edges.jsonl")
        self.nodes_metadata_file = os.path.join(self.db_path, "nodes.jsonl")
        
        # Create database directory structure
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(os.path.join(self.db_path, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(self.db_path, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(self.db_path, "documents"), exist_ok=True)
        
        # Initialize graph edges and nodes files
        if not os.path.exists(self.graph_edges_file):
            with open(self.graph_edges_file, 'w') as f:
                pass
                
        if not os.path.exists(self.nodes_metadata_file):
            with open(self.nodes_metadata_file, 'w') as f:
                pass
    
    def find_documents(self, directory: str, extensions: List[str] = ['.md', '.txt']) -> List[str]:
        """Find all documents with the specified extensions."""
        documents = []
        for ext in extensions:
            documents.extend(list(Path(directory).rglob(f"*{ext}")))
        return [str(doc) for doc in documents]
    
    def chunk_document(self, content: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        
        if len(content) <= self.chunk_size:
            chunks.append(content)
        else:
            start = 0
            while start < len(content):
                end = min(start + self.chunk_size, len(content))
                
                # Try to find a good split point
                if end < len(content):
                    split_candidates = [
                        content.rfind('. ', start, end),
                        content.rfind('\n', start, end),
                        content.rfind('. ', start, end - 10),
                        content.rfind(' ', start, end)
                    ]
                    # Use the latest good split point
                    split_point = max(filter(lambda x: x > start, split_candidates), default=end)
                    end = split_point + 1  # Include the split character
                
                chunks.append(content[start:end])
                start = end - self.chunk_overlap
        
        return chunks
    
    def get_openai_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for a chunk of text."""
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
            
        # Truncate long text
        if len(text) > 8000:
            text = text[:8000]
            
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": "text-embedding-3-small",
                    "dimensions": 1536
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error getting embedding: {response.text}")
                return []
                
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and extract its content and metadata."""
        file_path = os.path.abspath(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
            
        # Basic metadata
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1]
        title = os.path.splitext(filename)[0].replace('_', ' ').title()
        
        # Extract title from markdown if possible
        if file_ext == '.md':
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    title = line.replace('# ', '').strip()
                    break
        
        return {
            "file_path": file_path,
            "title": title,
            "content": content,
            "doc_type": file_ext.replace('.', '')
        }
    
    def save_chunk(self, chunk_id: str, text: str):
        """Save a chunk to disk."""
        chunk_path = os.path.join(self.db_path, "chunks", f"{chunk_id}.txt")
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def save_embedding(self, chunk_id: str, embedding: List[float]):
        """Save an embedding to disk."""
        if not embedding:
            return
            
        embedding_path = os.path.join(self.db_path, "embeddings", f"{chunk_id}.json")
        with open(embedding_path, 'w') as f:
            json.dump(embedding, f)
    
    def add_node_metadata(self, node_id: str, metadata: Dict[str, Any]):
        """Add node metadata to the nodes file."""
        node_data = {"id": node_id, "metadata": metadata}
        with open(self.nodes_metadata_file, 'a') as f:
            f.write(json.dumps(node_data) + '\n')
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float = 1.0):
        """Add an edge to the edges file."""
        edge_data = {
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "weight": weight
        }
        with open(self.graph_edges_file, 'a') as f:
            f.write(json.dumps(edge_data) + '\n')
    
    def ingest_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a document into the database.
        
        Args:
            doc_data: Document data dictionary with title, content, etc.
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not doc_data or not doc_data.get("content"):
            logger.error("Invalid document data")
            return {"nodes_created": 0, "edges_created": 0}
            
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Save document data
        doc_path = os.path.join(self.db_path, "documents", f"{doc_id}.json")
        doc_metadata = {
            "id": doc_id,
            "title": doc_data["title"],
            "file_path": doc_data["file_path"],
            "doc_type": doc_data.get("doc_type", "unknown")
        }
        
        with open(doc_path, 'w') as f:
            json.dump(doc_metadata, f, indent=2)
        
        # Chunk the document
        logger.info(f"Chunking document: {doc_data['title']}")
        chunks = self.chunk_document(doc_data["content"])
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process chunks one by one to minimize memory usage
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            # Generate chunk ID
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Create chunk metadata
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "title": doc_data["title"],
                "file_path": doc_data["file_path"]
            }
            
            # Save chunk text
            self.save_chunk(chunk_id, chunk)
            
            # Get and save embedding
            logger.info(f"Getting embedding for chunk {i+1}/{len(chunks)}")
            embedding = self.get_openai_embedding(chunk)
            if embedding:
                self.save_embedding(chunk_id, embedding)
                chunk_metadata["has_embedding"] = True
            else:
                chunk_metadata["has_embedding"] = False
            
            # Add node to graph
            self.add_node_metadata(chunk_id, chunk_metadata)
            
            # Force garbage collection
            del chunk
            del embedding
            gc.collect()
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
        
        # Create edges between consecutive chunks
        logger.info("Creating sequential edges between chunks")
        edges_created = 0
        for i in range(len(chunk_ids) - 1):
            self.add_edge(chunk_ids[i], chunk_ids[i + 1], "sequential")
            edges_created += 1
        
        # Garbage collection
        del chunks
        del chunk_ids
        gc.collect()
        
        return {
            "doc_id": doc_id,
            "nodes_created": len(chunks),
            "edges_created": edges_created
        }
    
    def create_qa_pairs(self, document_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Create sample QA pairs from documents without keeping them all in memory.
        Process one document at a time.
        """
        qa_pairs = []
        
        for doc_path in document_paths:
            if not doc_path.endswith('.md'):
                continue
                
            # Process document
            doc_data = self.process_document(doc_path)
            if not doc_data:
                continue
            
            # Create basic QA pairs
            qa_pairs.append({
                "question": f"What is {doc_data['title']} about?",
                "answer": f"{doc_data['title']} is about {' '.join(doc_data['content'].split()[:30])}...",
                "document": doc_data["file_path"]
            })
            
            # Extract sections (for markdown only)
            if doc_path.endswith('.md'):
                lines = doc_data["content"].split('\n')
                sections = []
                current_section = {"title": "", "content": []}
                
                for line in lines:
                    if line.startswith('## '):
                        # Save previous section
                        if current_section["content"]:
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            "title": line.replace('## ', ''),
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
                    if len(section["content"]) < 3:  # Skip very small sections
                        continue
                        
                    section_content = "\n".join(section["content"])
                    qa_pairs.append({
                        "question": f"What does the section '{section['title']}' in {doc_data['title']} describe?",
                        "answer": f"The section '{section['title']}' in {doc_data['title']} describes {' '.join(section_content.split()[:30])}...",
                        "document": doc_data["file_path"],
                        "section": section["title"]
                    })
            
            # Garbage collection
            del doc_data
            gc.collect()
        
        return qa_pairs
    
    def create_similarity_edges(self, similarity_threshold: float = 0.75, batch_size: int = 10):
        """
        Create edges between similar chunks.
        Process in small batches to minimize memory usage.
        """
        # Get all embeddings
        embedding_dir = os.path.join(self.db_path, "embeddings")
        embedding_files = list(Path(embedding_dir).glob("*.json"))
        
        if not embedding_files:
            logger.warning("No embeddings found to create similarity edges")
            return 0
            
        logger.info(f"Creating similarity edges from {len(embedding_files)} embeddings")
        
        # Function to read embedding
        def read_embedding(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading embedding {file_path}: {e}")
                return None
        
        # Function to compute cosine similarity
        def cosine_similarity(v1, v2):
            import numpy as np
            if not v1 or not v2:
                return 0
            dot_product = sum(a*b for a, b in zip(v1, v2))
            norm_a = sum(a*a for a in v1) ** 0.5
            norm_b = sum(b*b for b in v2) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
        
        # Process in batches
        edges_created = 0
        processed = 0
        total_comparisons = (len(embedding_files) * (len(embedding_files) - 1)) // 2
        
        # Check every 10% for progress reporting
        report_interval = max(1, total_comparisons // 10)
        
        logger.info(f"Processing {total_comparisons} potential connections in batches")
        
        for i, file1 in enumerate(embedding_files):
            chunk_id1 = file1.stem
            embedding1 = read_embedding(file1)
            
            if not embedding1:
                continue
                
            for j in range(i+1, len(embedding_files)):
                file2 = embedding_files[j]
                chunk_id2 = file2.stem
                
                # Skip if chunks are from the same document and sequential
                if chunk_id1.split('_chunk_')[0] == chunk_id2.split('_chunk_')[0]:
                    try:
                        idx1 = int(chunk_id1.split('_chunk_')[1])
                        idx2 = int(chunk_id2.split('_chunk_')[1])
                        if abs(idx1 - idx2) <= 1:
                            # Skip since we already have sequential edges
                            processed += 1
                            continue
                    except:
                        pass
                
                embedding2 = read_embedding(file2)
                
                if embedding2:
                    similarity = cosine_similarity(embedding1, embedding2)
                    
                    if similarity > similarity_threshold:
                        # Add bidirectional edges
                        self.add_edge(chunk_id1, chunk_id2, "similarity", similarity)
                        self.add_edge(chunk_id2, chunk_id1, "similarity", similarity)
                        edges_created += 2
                
                processed += 1
                
                # Report progress
                if processed % report_interval == 0:
                    logger.info(f"Processed {processed}/{total_comparisons} comparisons, created {edges_created} edges")
                
                # Garbage collection every batch
                if processed % batch_size == 0:
                    gc.collect()
            
            # Explicit cleanup
            del embedding1
            gc.collect()
        
        logger.info(f"Created {edges_created} similarity edges")
        return edges_created

def main():
    """Main function to prepare the PathRAG database."""
    parser = argparse.ArgumentParser(description="Minimal Memory PathRAG Database Preparation")
    parser.add_argument("--docs-dir", type=str, default=DOCS_DIR, help="Directory containing documents to ingest")
    parser.add_argument("--db-path", type=str, default=DB_PATH, help="Path to store document database")
    parser.add_argument("--qa-pairs", type=str, default=os.path.join(DATA_DIR, "qa_pairs.json"), help="Path to save generated QA pairs")
    parser.add_argument("--clean", action="store_true", help="Clean existing database before ingestion")
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Clean existing database if requested
    if args.clean and os.path.exists(args.db_path):
        logger.info(f"Cleaning existing database at {args.db_path}")
        shutil.rmtree(args.db_path)
    
    # Initialize database builder
    config = {
        "db_path": args.db_path,
        "chunk_size": 500,  # Smaller chunks to save memory
        "chunk_overlap": 100
    }
    
    logger.info("Initializing minimal memory PathRAG database builder")
    pathrag_builder = MinimalPathRAGBuilder(config)
    
    # Find documents
    logger.info(f"Finding documents in {args.docs_dir}")
    documents = pathrag_builder.find_documents(args.docs_dir)
    logger.info(f"Found {len(documents)} documents")
    
    # Create sample QA pairs
    logger.info("Creating sample QA pairs")
    qa_pairs = pathrag_builder.create_qa_pairs(documents)
    
    # Save QA pairs
    with open(args.qa_pairs, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2)
    logger.info(f"Saved {len(qa_pairs)} QA pairs to {args.qa_pairs}")
    
    # Process each document
    logger.info("Ingesting documents into PathRAG")
    total_nodes = 0
    total_edges = 0
    
    for doc_path in documents:
        logger.info(f"Processing document: {doc_path}")
        doc_data = pathrag_builder.process_document(doc_path)
        
        if doc_data:
            result = pathrag_builder.ingest_document(doc_data)
            total_nodes += result.get("nodes_created", 0)
            total_edges += result.get("edges_created", 0)
            
            logger.info(f"Ingested document: {doc_data['title']} - Created {result.get('nodes_created', 0)} nodes and {result.get('edges_created', 0)} edges")
            
            # Force garbage collection after each document
            del doc_data
            del result
            gc.collect()
    
    logger.info(f"Total nodes created: {total_nodes}")
    logger.info(f"Total sequential edges created: {total_edges}")
    
    # Create similarity edges
    logger.info("Creating similarity edges between chunks")
    similarity_edges = pathrag_builder.create_similarity_edges(similarity_threshold=0.7)
    logger.info(f"Created {similarity_edges} similarity edges")
    
    logger.info("Database preparation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
