#!/usr/bin/env python3
"""
Low Memory PathRAG Database Preparation Script

This script prepares a PathRAG database by ingesting documents from the
HADES repository, creating embeddings, and setting up the system for testing.
This version is optimized for lower memory usage.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any
import time
import uuid
import gc

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'logs',
                f'pathrag_db_prep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "chroma_db")

# Import required libraries
try:
    import numpy as np
    import openai
    import networkx as nx
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Failed to import required library: {e}")
    logger.error("Please install the required libraries with:")
    logger.error("pip install python-dotenv openai networkx numpy")
    sys.exit(1)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class LowMemoryPathRAGBuilder:
    """
    Memory-efficient PathRAG database builder.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PathRAG database builder.
        
        Args:
            config: Configuration settings
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.db_path = config.get("db_path", DB_PATH)
        self.use_chroma = config.get("use_chroma", False)
        
        # Initialize OpenAI API key
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        else:
            logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
            sys.exit(1)
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
        
        # Initialize ChromaDB if needed
        if self.use_chroma:
            self.initialize_chromadb()
    
    def initialize_chromadb(self):
        """Initialize ChromaDB for document storage."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            logger.error("Failed to import ChromaDB. Please install it with: pip install chromadb")
            sys.exit(1)

        logger.info(f"Initializing ChromaDB at {self.db_path}")
        os.makedirs(self.db_path, exist_ok=True)
        
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )
            self.collection = self.chroma_client.create_collection(
                name="pathrag_docs",
                embedding_function=self.embedding_function,
                get_or_create=True
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            sys.exit(1)
    
    def find_documents(self, directory: str, extensions: List[str] = ['.md', '.txt']) -> List[str]:
        """
        Find all documents with the specified extensions in the given directory and its subdirectories.
        
        Args:
            directory: Directory to search
            extensions: List of file extensions to include
            
        Returns:
            List of paths to documents
        """
        documents = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    documents.append(os.path.join(root, file))
        
        return documents
    
    def process_markdown(self, file_path: str) -> Dict[str, Any]:
        """
        Process a Markdown file to extract metadata and content.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            Dictionary with metadata and content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract basic metadata
        filename = os.path.basename(file_path)
        title = filename.replace('.md', '').replace('_', ' ').title()
        
        # Try to extract title from the first heading
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                title = line.replace('# ', '')
                break
        
        # Extract sections
        sections = []
        current_section = {"title": "", "content": []}
        
        for line in lines:
            if line.startswith('## '):
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": line.replace('## ', ''),
                    "content": []
                }
            elif line.startswith('# '):
                # This is the document title, skip
                continue
            else:
                # Add to current section
                current_section["content"].append(line)
        
        # Add the last section
        if current_section["content"]:
            sections.append(current_section)
        
        return {
            "title": title,
            "file_path": file_path,
            "content": content,
            "sections": sections
        }
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        if len(text) <= self.chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                
                # Try to find a period or newline to split on
                if end < len(text):
                    # Look for a good split point
                    split_candidates = [text.rfind('. ', start, end), text.rfind('\n', start, end)]
                    split_point = max(split_candidates)
                    
                    if split_point > start:
                        end = split_point + 1  # Include the period or newline
                
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Truncate text if too long
            if len(text) > 8000:
                text = text[:8000]
                
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def add_to_graph(self, node_id: str, metadata: Dict[str, Any], text: str = None, embedding=None):
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique ID for the node
            metadata: Node metadata
            text: Node text content
            embedding: Node embedding vector
        """
        # If embedding not provided but text is, get embedding
        if embedding is None and text is not None and OPENAI_API_KEY:
            embedding = self.get_embedding(text)
        
        node_data = {**metadata}
        if embedding:
            node_data["embedding"] = embedding
        
        self.graph.add_node(node_id, **node_data)
    
    def save_document_json(self, doc_id: str, chunks: List[str], metadata: Dict[str, Any], 
                         chunk_ids: List[str], embeddings: List[List[float]] = None):
        """
        Save document chunks and metadata to disk as JSON.
        
        Args:
            doc_id: Document ID
            chunks: List of text chunks
            metadata: Document metadata
            chunk_ids: List of chunk IDs
            embeddings: List of embeddings for each chunk
        """
        doc_data = {
            "id": doc_id,
            "metadata": metadata,
            "chunks": []
        }
        
        for i, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks)):
            chunk_data = {
                "id": chunk_id,
                "text": chunk,
                "metadata": {**metadata, "chunk_index": i, "chunk_count": len(chunks)}
            }
            
            if embeddings and i < len(embeddings):
                chunk_data["embedding"] = embeddings[i]
                
            doc_data["chunks"].append(chunk_data)
        
        # Save to JSON file
        os.makedirs(os.path.join(DATA_DIR, "documents"), exist_ok=True)
        output_path = os.path.join(DATA_DIR, "documents", f"{doc_id}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2)
            
        logger.info(f"Saved document data to {output_path}")
    
    def ingest_document(self, content: str, metadata: Dict[str, Any], get_embedding: bool = True) -> Dict[str, Any]:
        """
        Ingest a document into the knowledge base.
        
        Args:
            content: Document content
            metadata: Document metadata
            get_embedding: Whether to compute embeddings
            
        Returns:
            Dict containing ingestion results
        """
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Chunk the document
        chunks = self.chunk_text(content)
        
        # Generate chunk IDs
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Get embeddings if needed
        embeddings = None
        if get_embedding and OPENAI_API_KEY:
            logger.info(f"Getting embeddings for {len(chunks)} chunks")
            embeddings = []
            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                embeddings.append(embedding)
                # Sleep a bit to avoid rate limiting
                time.sleep(0.1)
        
        # Add to ChromaDB if enabled
        if self.use_chroma and embeddings:
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_ids = chunk_ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size] if embeddings else None
                batch_metadatas = [{**metadata, "chunk_index": i+j, "chunk_count": len(chunks), "document_id": doc_id} 
                                 for j in range(len(batch_chunks))]
                
                if batch_embeddings:
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_chunks,
                        metadatas=batch_metadatas,
                        embeddings=batch_embeddings
                    )
                else:
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_chunks,
                        metadatas=batch_metadatas
                    )
                
                logger.info(f"Added batch {i//batch_size + 1} of {(len(chunks) + batch_size - 1) // batch_size} to ChromaDB")
        
        # Add to graph
        for i, chunk_id in enumerate(chunk_ids):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "document_id": doc_id
            }
            
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None
            self.add_to_graph(chunk_id, chunk_metadata, embedding=embedding)
        
        # Create edges between consecutive chunks
        for i in range(len(chunk_ids) - 1):
            self.graph.add_edge(chunk_ids[i], chunk_ids[i + 1], weight=1.0, edge_type="sequential")
        
        # Save document data to JSON file
        self.save_document_json(doc_id, chunks, metadata, chunk_ids, embeddings)
        
        # Free memory
        del chunks
        del embeddings
        gc.collect()
        
        return {
            "document_id": doc_id,
            "nodes_created": len(chunk_ids),
            "edges_created": len(chunk_ids) - 1 if len(chunk_ids) > 1 else 0
        }
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a file into the knowledge base.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing ingestion results
        """
        logger.info(f"Processing file: {file_path}")
        
        if file_path.endswith('.md'):
            # Process Markdown files specially
            doc_data = self.process_markdown(file_path)
            
            # Ingest the full document
            result = self.ingest_document(
                doc_data["content"],
                metadata={
                    "title": doc_data["title"],
                    "file_path": file_path,
                    "document_type": "markdown"
                }
            )
            
            logger.info(f"Ingested document: {doc_data['title']} - Created {result['nodes_created']} nodes and {result['edges_created']} edges")
            
            # Free memory
            gc.collect()
            
            # This is where we would ingest sections, but to save memory we'll skip this for now
            
            return result
        else:
            # Ingest other file types directly
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get metadata from the file path
            filename = os.path.basename(file_path)
            title = filename.replace('.txt', '').replace('_', ' ').title()
            
            # Ingest the document
            result = self.ingest_document(
                content,
                metadata={
                    "title": title,
                    "file_path": file_path,
                    "document_type": "text"
                }
            )
            
            logger.info(f"Ingested file: {filename} - Created {result['nodes_created']} nodes and {result['edges_created']} edges")
            
            # Free memory
            gc.collect()
            
            return result
    
    def create_qa_pairs(self, document_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create sample QA pairs from document data.
        
        Args:
            document_data: List of processed document data
            
        Returns:
            List of QA pairs
        """
        qa_pairs = []
        
        for doc in document_data:
            # Create a general question about the document
            qa_pairs.append({
                "question": f"What is {doc['title']} about?",
                "answer": f"{doc['title']} is about {' '.join(doc['content'].split()[:50])}...",
                "document": doc["file_path"]
            })
            
            # Create questions for each section
            for section in doc["sections"]:
                if len(section["content"]) > 100:  # Skip very small sections
                    section_content = "\n".join(section["content"])
                    qa_pairs.append({
                        "question": f"What does the section '{section['title']}' in {doc['title']} describe?",
                        "answer": f"The section '{section['title']}' in {doc['title']} describes {' '.join(section_content.split()[:50])}...",
                        "document": doc["file_path"],
                        "section": section["title"]
                    })
        
        return qa_pairs
    
    def save_graph(self, output_path: str = None):
        """
        Save the NetworkX graph to a file.
        
        Args:
            output_path: Path to save the graph
        """
        if output_path is None:
            output_path = os.path.join(DATA_DIR, "pathrag_graph.gpickle")
        
        logger.info(f"Saving graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges to {output_path}")
        
        try:
            nx.write_gpickle(self.graph, output_path)
            logger.info(f"Graph saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

def main():
    """Main function to prepare the PathRAG database."""
    parser = argparse.ArgumentParser(description="Low Memory PathRAG Database Preparation")
    parser.add_argument("--docs-dir", type=str, default=DOCS_DIR, help="Directory containing documents to ingest")
    parser.add_argument("--db-path", type=str, default=DB_PATH, help="Path to store document database")
    parser.add_argument("--qa-pairs", type=str, default=os.path.join(DATA_DIR, "qa_pairs.json"), help="Path to save generated QA pairs")
    parser.add_argument("--use-chroma", action="store_true", help="Use ChromaDB for storage (more memory intensive)")
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize PathRAG builder
    config = {
        "db_path": args.db_path,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "use_chroma": args.use_chroma
    }
    
    logger.info("Initializing low-memory PathRAG database builder")
    pathrag_builder = LowMemoryPathRAGBuilder(config)
    
    # Find documents
    logger.info(f"Finding documents in {args.docs_dir}")
    documents = pathrag_builder.find_documents(args.docs_dir)
    logger.info(f"Found {len(documents)} documents")
    
    # Process documents to extract data for QA pairs
    document_data = []
    for doc_path in documents:
        if doc_path.endswith('.md'):
            document_data.append(pathrag_builder.process_markdown(doc_path))
    
    # Create sample QA pairs
    logger.info("Creating sample QA pairs")
    qa_pairs = pathrag_builder.create_qa_pairs(document_data)
    
    # Save QA pairs
    with open(args.qa_pairs, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2)
    logger.info(f"Saved {len(qa_pairs)} QA pairs to {args.qa_pairs}")
    
    # Free memory
    del document_data
    gc.collect()
    
    # Ingest documents
    logger.info("Ingesting documents into PathRAG")
    total_nodes = 0
    total_edges = 0
    
    for doc_path in documents:
        result = pathrag_builder.ingest_file(doc_path)
        total_nodes += result["nodes_created"]
        total_edges += result["edges_created"]
        
        # Force garbage collection after each file
        gc.collect()
    
    logger.info(f"Total nodes created: {total_nodes}")
    logger.info(f"Total edges created: {total_edges}")
    
    # Save the graph
    pathrag_builder.save_graph()
    
    logger.info("Database preparation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
