#!/usr/bin/env python3
"""
Run Dataset Builder

A utility script to run the rag-dataset-builder with the correct configuration.
This script directly uses the rag-dataset-builder module without modifying it.
It also provides options to force reprocessing of all documents.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.absolute()
rag_builder_path = os.path.join(project_root, "rag-dataset-builder")
config_path = os.path.join(rag_builder_path, "config", "config.yaml")

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
logger = logging.getLogger("run_dataset_builder")

def main():
    """Main entry point to run the dataset builder."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the RAG dataset builder")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all documents")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before processing")
    args = parser.parse_args()
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"RAG builder path: {rag_builder_path}")
    logger.info(f"Config path: {config_path}")
    
    # Handle force reprocessing option
    if args.force:
        checkpoint_file = os.path.join(project_root, "rag_databases", "current", "checkpoint.json")
        if os.path.exists(checkpoint_file):
            logger.info(f"Resetting checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "processed_files": [],
                    "processed_chunks": 0,
                    "total_chunks": 0,
                    "last_updated": ""
                }, f, indent=2)
        else:
            logger.info("No checkpoint file found to reset")
    
    # Handle clean option
    if args.clean:
        output_dir = os.path.join(project_root, "rag_databases", "current")
        if os.path.exists(output_dir):
            logger.info(f"Cleaning output directory: {output_dir}")
            # Only remove files, not the directory itself
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
            logger.info("Output directory cleaned")
    
    # Ensure source_documents directory exists
    source_docs_dir = os.path.join(project_root, "source_documents")
    if not os.path.exists(source_docs_dir):
        logger.info(f"Creating source documents directory: {source_docs_dir}")
        os.makedirs(source_docs_dir, exist_ok=True)
    
    # Check if we have any test documents
    source_doc_files = list(Path(source_docs_dir).glob("**/*.*"))
    if not source_doc_files:
        logger.warning("No source documents found. The dataset builder will not have anything to process.")
        create_test_doc = input("Would you like to create a test document? (y/n): ")
        if create_test_doc.lower() == 'y':
            test_doc_path = os.path.join(source_docs_dir, "test_document.md")
            with open(test_doc_path, 'w') as f:
                f.write("# Test Document\n\nThis is a simple test document for the RAG dataset builder.")
            logger.info(f"Created a test document at {test_doc_path}")
    else:
        logger.info(f"Found {len(source_doc_files)} source documents to process")
        for doc in source_doc_files:
            logger.info(f"  - {doc.relative_to(source_docs_dir)}")
    
    # Ensure output directory exists
    output_dir = os.path.join(project_root, "rag_databases", "current")
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Import and run the dataset builder
    try:
        logger.info("Importing RAG dataset builder...")
        from src.main import RAGDatasetBuilder
        
        # Create an instance of the builder and debug paths
        logger.info("Initializing RAG dataset builder...")
        builder = RAGDatasetBuilder(config_path)
        
        logger.info(f"Actual data directory path: {builder.data_dir}")
        logger.info(f"Actual output directory path: {builder.output_dir}")
        
        # Debug document discovery
        include_patterns = builder.config.get("input", {}).get("include", [])
        logger.info(f"Include patterns: {include_patterns}")
        
        # Manually check if files exist
        import glob
        for pattern in include_patterns:
            pattern_path = os.path.join(builder.data_dir, pattern)
            logger.info(f"Checking pattern: {pattern_path}")
            files = glob.glob(pattern_path, recursive=True)
            logger.info(f"  Found {len(files)} files")
            for file in files:
                logger.info(f"  - {file}")
        
        # Run the builder
        logger.info("\nRunning dataset builder...")
        builder.build_dataset()
        
        # Make sure to finalize the output formatter
        if hasattr(builder.formatter, 'finalize'):
            logger.info("Finalizing output formatter...")
            builder.formatter.finalize()
        
        logger.info("Dataset builder completed successfully")
    except Exception as e:
        logger.error(f"Error running dataset builder: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
