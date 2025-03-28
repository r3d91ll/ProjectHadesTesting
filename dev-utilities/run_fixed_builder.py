#!/usr/bin/env python3
"""
Runner script for the fixed RAG dataset builder.
This script directly executes the fixed_main.py without modifying the original source files.
"""

import os
import sys
import subprocess
from pathlib import Path

# Get the absolute path of the project root
project_root = Path(__file__).parent.parent.absolute()
dev_utilities_dir = os.path.join(project_root, "dev-utilities")
config_path = os.path.join(project_root, "rag-dataset-builder", "config", "config.yaml")

# Ensure the source documents directory exists
source_docs_dir = os.path.join(project_root, "source_documents")
if not os.path.exists(source_docs_dir):
    os.makedirs(source_docs_dir)
    print(f"Created source documents directory at {source_docs_dir}")

# Check if we have any test documents
if not any(os.path.isfile(os.path.join(source_docs_dir, f)) for f in os.listdir(source_docs_dir)):
    print("WARNING: No source documents found. The dataset builder may not have anything to process.")
    # Create a simple test document
    test_doc_path = os.path.join(source_docs_dir, "test_document.md")
    with open(test_doc_path, 'w') as f:
        f.write("# Test Document\n\nThis is a simple test document for the RAG dataset builder.")
    print(f"Created a test document at {test_doc_path}")

# Ensure output directory exists
output_dir = os.path.join(project_root, "rag_databases", "current")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory at {output_dir}")

# Run the fixed main script as a module
print(f"Running fixed RAG dataset builder with config: {config_path}")
print(f"Working directory: {project_root}")
print(f"Source documents directory: {source_docs_dir}")
print(f"Output directory: {output_dir}")

try:
    # Execute the fixed_main.py directly
    cmd = [sys.executable, os.path.join(dev_utilities_dir, "fixed_main.py"), "--config", config_path]
    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.run(
        cmd,
        cwd=project_root,  # Run from project root
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("Output:")
    print(process.stdout)
    
    if process.stderr:
        print("Errors:")
        print(process.stderr)
    
    print("RAG dataset builder completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error running RAG dataset builder: {e}")
    print("Output:")
    print(e.stdout)
    print("Errors:")
    print(e.stderr)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
