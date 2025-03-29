#!/usr/bin/env python3
"""
Temporary utility script to run the rag-dataset-builder with proper imports.
This script ensures that the Python import system works correctly with the module structure.
"""

import os
import sys
import subprocess

# Add the rag-dataset-builder to the Python path
rag_builder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "rag-dataset-builder")
sys.path.insert(0, rag_builder_path)

# Change directory to rag-dataset-builder
os.chdir(rag_builder_path)

# Fix imports by modifying the files temporarily
def fix_imports():
    files_to_fix = [
        os.path.join(rag_builder_path, "src", "embedders.py"),
        os.path.join(rag_builder_path, "src", "chunkers.py"),
        os.path.join(rag_builder_path, "src", "processors.py"),
        os.path.join(rag_builder_path, "src", "formatters.py"),
        os.path.join(rag_builder_path, "src", "main.py")
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace relative imports with absolute imports
            content = content.replace("from .builder", "from src.builder")
            content = content.replace("from .core", "from src.core")
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"Fixed imports in {file_path}")

# Run the fix_imports function
fix_imports()

# Now run the dataset builder
config_path = os.path.join(rag_builder_path, "config", "config.yaml")
print(f"Running rag-dataset-builder with config: {config_path}")

# Execute the main module
from src.main import main
main(config_path=config_path)
