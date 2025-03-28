#!/usr/bin/env python3
"""
Debug Document Processor

A utility script to debug the document processor in RAG dataset builder.
This script tests the processor directly on source documents.
"""

import os
import sys
import logging
import hashlib
from pathlib import Path

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
logger = logging.getLogger("debug_processor")

# Import necessary components
try:
    from src.processors import SimpleTextProcessor
except ImportError as e:
    logger.error(f"Error importing RAG dataset builder: {e}")
    sys.exit(1)

def directly_process_document(doc_path):
    """Process a document directly, bypassing the RAG dataset builder."""
    logger.info(f"Processing document: {doc_path}")
    
    processor = SimpleTextProcessor()
    
    # Try manually opening and processing the file
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Basic document metadata
        filename = os.path.basename(doc_path)
        doc_id = hashlib.md5(doc_path.encode()).hexdigest()
        
        logger.info(f"Successfully read {len(text)} characters from {filename}")
        logger.info(f"Generated ID: {doc_id}")
        
        # Now try the processor
        doc_data = processor.process_document(doc_path)
        if doc_data:
            logger.info(f"Processor returned document with ID: {doc_data.get('id', 'MISSING')}")
            if 'text' in doc_data:
                logger.info(f"Document text length: {len(doc_data['text'])} characters")
            if 'metadata' in doc_data:
                logger.info(f"Document metadata: {list(doc_data['metadata'].keys())}")
            return doc_data
        else:
            logger.error(f"Processor returned None for {doc_path}")
            return None
    except Exception as e:
        logger.error(f"Error directly processing {doc_path}: {e}")
        return None

def add_fixed_processor(doc_path):
    """Add a fixed version of the processor output."""
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Extract basic metadata
        filename = os.path.basename(doc_path)
        file_size = os.path.getsize(doc_path)
        doc_id = hashlib.md5(doc_path.encode()).hexdigest()
        
        # Try to extract title from text
        title = filename
        import re
        title_match = re.search(r'^(?:#|Title:)\s*(.+?)$', text, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        
        # Create a document with all required fields
        from datetime import datetime
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
        
        logger.info(f"Created fixed document data for {filename}")
        return doc_data
    except Exception as e:
        logger.error(f"Error creating fixed document data for {doc_path}: {e}")
        return None

def main():
    """Main entry point for the debug script."""
    source_docs_dir = os.path.join(project_root, "source_documents")
    if not os.path.exists(source_docs_dir):
        logger.error(f"Source documents directory not found: {source_docs_dir}")
        return 1
    
    # Find all markdown files
    markdown_files = list(Path(source_docs_dir).glob("**/*.md"))
    if not markdown_files:
        logger.error("No markdown files found in source_documents directory")
        return 1
    
    logger.info(f"Found {len(markdown_files)} markdown files")
    
    # Process each document
    for doc_path in markdown_files:
        logger.info(f"Testing document: {doc_path}")
        # Try the existing processor
        existing_result = directly_process_document(str(doc_path))
        if not existing_result:
            # Try our fixed processor
            fixed_result = add_fixed_processor(str(doc_path))
            if fixed_result:
                logger.info("Fixed processor succeeded where original failed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
