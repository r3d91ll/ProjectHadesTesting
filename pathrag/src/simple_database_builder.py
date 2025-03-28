#!/usr/bin/env python3
"""
Simple PathRAG Database Preparation Script

This script prepares a PathRAG database by ingesting documents from the
HADES repository, creating embeddings, and setting up the system for testing.
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

# Import required libraries
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.utils import embedding_functions
    from dotenv import load_dotenv
    import openai
    import networkx as nx
except ImportError as e:
    logger.error(f"Failed to import required library: {e}")
    logger.error("Please install the required libraries with:")
    logger.error("pip install sentence-transformers chromadb python-dotenv openai networkx numpy")
    sys.exit(1)

# Define paths
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "chroma_db")

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class PathRAGSimpleBuilder:
    """
    Simple PathRAG database builder without Arize Phoenix dependency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PathRAG database builder.
        
        Args:
            config: Configuration settings
        """
        self.config = config
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.use_openai = self.embedding_model_name == "openai"
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.db_path = config.get("db_path", DB_PATH)
        
        # Initialize embedding model
        self.initialize_embedding_model()
        
        # Initialize ChromaDB
        self.initialize_chromadb()
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
    
    def initialize_embedding_model(self):
        """Initialize the embedding model."""
        if self.use_openai:
            logger.info("Using OpenAI embeddings model")
            if not OPENAI_API_KEY:
                logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
                sys.exit(1)
            openai.api_key = OPENAI_API_KEY
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )
        else:
            logger.info(f"Loading SentenceTransformer model: {self.embedding_model_name}")
            try:
                self.model = SentenceTransformer(self.embedding_model_name)
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                sys.exit(1)
    
    def initialize_chromadb(self):
        """Initialize ChromaDB for document storage."""
        logger.info(f"Initializing ChromaDB at {self.db_path}")
        os.makedirs(self.db_path, exist_ok=True)
        
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
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
    
    def add_to_graph(self, node_id: str, metadata: Dict[str, Any], embedding=None):
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique ID for the node
            metadata: Node metadata
            embedding: Node embedding vector
        """
        self.graph.add_node(node_id, **metadata)
    
    def create_edges(self, similarity_threshold: float = 0.7):
        """
        Create edges between nodes based on embedding similarity.
        
        Args:
            similarity_threshold: Threshold for creating edges
        """
        logger.info("Creating edges between nodes based on similarity")
        
        # Get all documents from ChromaDB
        all_docs = self.collection.get()
        
        if not all_docs or not all_docs['embeddings']:
            logger.warning("No documents found in the collection")
            return
        
        # Calculate similarity between all pairs of documents
        for i, (id_i, embedding_i, metadata_i) in enumerate(zip(all_docs['ids'], all_docs['embeddings'], all_docs['metadatas'])):
            for j, (id_j, embedding_j, metadata_j) in enumerate(zip(all_docs['ids'], all_docs['embeddings'], all_docs['metadatas'])):
                if i != j:  # Don't compare a document with itself
                    # Calculate cosine similarity
                    similarity = np.dot(embedding_i, embedding_j) / (np.linalg.norm(embedding_i) * np.linalg.norm(embedding_j))
                    
                    if similarity > similarity_threshold:
                        # Add edge to graph
                        self.graph.add_edge(id_i, id_j, weight=similarity)
                        
                        logger.debug(f"Added edge: {id_i} -> {id_j} (similarity: {similarity:.4f})")
        
        logger.info(f"Created {self.graph.number_of_edges()} edges between {self.graph.number_of_nodes()} nodes")
    
    def ingest_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a document into the knowledge base.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Dict containing ingestion results
        """
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Chunk the document
        chunks = self.chunk_text(content)
        
        # Add each chunk to ChromaDB and the graph
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "document_id": doc_id
            }
            
            # Add to ChromaDB
            self.collection.add(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
            
            # Add to graph
            self.add_to_graph(chunk_id, chunk_metadata)
            chunk_ids.append(chunk_id)
        
        # Create edges between consecutive chunks
        for i in range(len(chunk_ids) - 1):
            self.graph.add_edge(chunk_ids[i], chunk_ids[i + 1], weight=1.0, edge_type="sequential")
        
        return {
            "document_id": doc_id,
            "nodes_created": len(chunks),
            "edges_created": len(chunks) - 1 if len(chunks) > 1 else 0
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
            
            # Also ingest each section separately for better retrieval
            nodes_created = result['nodes_created']
            edges_created = result['edges_created']
            
            for section in doc_data["sections"]:
                section_content = "\n".join(section["content"])
                if len(section_content.strip()) > 100:  # Skip very small sections
                    section_result = self.ingest_document(
                        section_content,
                        metadata={
                            "title": f"{doc_data['title']} - {section['title']}",
                            "section": section["title"],
                            "parent_document": doc_data["title"],
                            "file_path": file_path,
                            "document_type": "markdown_section"
                        }
                    )
                    
                    nodes_created += section_result['nodes_created']
                    edges_created += section_result['edges_created']
                    
                    logger.info(f"Ingested section: {section['title']} - Created {section_result['nodes_created']} nodes and {section_result['edges_created']} edges")
            
            return {
                "document_id": result["document_id"],
                "nodes_created": nodes_created,
                "edges_created": edges_created
            }
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
    parser = argparse.ArgumentParser(description="Simple PathRAG Database Preparation")
    parser.add_argument("--docs-dir", type=str, default=DOCS_DIR, help="Directory containing documents to ingest")
    parser.add_argument("--db-path", type=str, default=DB_PATH, help="Path to store ChromaDB")
    parser.add_argument("--qa-pairs", type=str, default=os.path.join(DATA_DIR, "qa_pairs.json"), help="Path to save generated QA pairs")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model to use")
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize PathRAG builder
    config = {
        "embedding_model": args.embedding_model,
        "db_path": args.db_path,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    logger.info("Initializing PathRAG database builder")
    pathrag_builder = PathRAGSimpleBuilder(config)
    
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
    
    # Ingest documents
    logger.info("Ingesting documents into PathRAG")
    total_nodes = 0
    total_edges = 0
    
    for doc_path in documents:
        result = pathrag_builder.ingest_file(doc_path)
        total_nodes += result["nodes_created"]
        total_edges += result["edges_created"]
    
    logger.info(f"Total nodes created: {total_nodes}")
    logger.info(f"Total edges created: {total_edges}")
    
    # Create additional edges based on similarity
    logger.info("Creating additional edges based on similarity")
    pathrag_builder.create_edges(similarity_threshold=0.7)
    
    # Save the graph
    pathrag_builder.save_graph()
    
    logger.info("Database preparation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
