#!/usr/bin/env python3
"""
PathRAG Database Preparation Script

This script prepares the PathRAG database by ingesting documents from the
HADES repository, creating a knowledge graph, and setting up the system
for question answering.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add parent directories to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the PathRAG configuration and adapter
from config.pathrag_config import get_config, validate_config
from implementations.pathrag.arize_integration.adapter import PathRAGArizeAdapter

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

def find_documents(directory: str, extensions: List[str] = ['.md', '.txt']) -> List[str]:
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

def process_markdown(file_path: str) -> Dict[str, Any]:
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

def ingest_documents(pathrag: PathRAGArizeAdapter, documents: List[str]) -> None:
    """
    Ingest documents into PathRAG.
    
    Args:
        pathrag: PathRAG adapter instance
        documents: List of document paths to ingest
    """
    for doc_path in documents:
        logger.info(f"Processing document: {doc_path}")
        
        if doc_path.endswith('.md'):
            # Process Markdown files specially
            doc_data = process_markdown(doc_path)
            
            # Ingest the full document
            result = pathrag.ingest_document(
                doc_data["content"],
                metadata={
                    "title": doc_data["title"],
                    "file_path": doc_data["file_path"],
                    "document_type": "markdown"
                }
            )
            
            logger.info(f"Ingested document: {doc_data['title']} - Created {result['nodes_created']} nodes and {result['edges_created']} edges")
            
            # Also ingest each section separately for better retrieval
            for section in doc_data["sections"]:
                section_content = "\n".join(section["content"])
                if len(section_content.strip()) > 100:  # Skip very small sections
                    section_result = pathrag.ingest_document(
                        section_content,
                        metadata={
                            "title": f"{doc_data['title']} - {section['title']}",
                            "section": section["title"],
                            "parent_document": doc_data["title"],
                            "file_path": doc_data["file_path"],
                            "document_type": "markdown_section"
                        }
                    )
                    
                    logger.info(f"Ingested section: {section['title']} - Created {section_result['nodes_created']} nodes and {section_result['edges_created']} edges")
        else:
            # Ingest other document types directly
            result = pathrag.ingest_file(doc_path)
            logger.info(f"Ingested file: {doc_path} - Created {result['nodes_created']} nodes and {result['edges_created']} edges")

def create_qa_pairs(document_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

def main():
    """Main function to prepare the PathRAG database."""
    parser = argparse.ArgumentParser(description="PathRAG Database Preparation")
    parser.add_argument("--docs-dir", type=str, default=DOCS_DIR, help="Directory containing documents to ingest")
    parser.add_argument("--qa-pairs", type=str, default=os.path.join(DATA_DIR, "qa_pairs.json"), help="Path to save generated QA pairs")
    args = parser.parse_args()
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed. Please check your .env file and settings.")
        return 1
    
    # Get configuration
    config = get_config()
    
    # Initialize PathRAG
    logger.info("Initializing PathRAG")
    pathrag = PathRAGArizeAdapter(config)
    pathrag.initialize()
    
    # Find documents
    logger.info(f"Finding documents in {args.docs_dir}")
    documents = find_documents(args.docs_dir)
    logger.info(f"Found {len(documents)} documents")
    
    # Process documents to extract data for QA pairs
    document_data = []
    for doc_path in documents:
        if doc_path.endswith('.md'):
            document_data.append(process_markdown(doc_path))
    
    # Create sample QA pairs
    logger.info("Creating sample QA pairs")
    qa_pairs = create_qa_pairs(document_data)
    
    # Save QA pairs
    with open(args.qa_pairs, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2)
    logger.info(f"Saved {len(qa_pairs)} QA pairs to {args.qa_pairs}")
    
    # Ingest documents
    logger.info("Ingesting documents into PathRAG")
    ingest_documents(pathrag, documents)
    
    logger.info("Database preparation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
