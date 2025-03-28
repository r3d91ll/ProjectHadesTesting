#!/usr/bin/env python3
"""
PDF Processor Patch

This utility script creates a temporary workaround for the NLTK punkt_tab/english issue
in the PDF processor of the RAG dataset builder.

This is a one-time fix script placed in dev-utilities/ as per project conventions.
"""

import os
import sys
import shutil
import logging
import nltk
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pdf_processor_patch")

def patch_pdf_processor():
    """Apply a patch to fix the PDF processor's NLTK dependency issues."""
    # Get the main NLTK data directory we established earlier
    nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')
    
    # Paths to relevant directories
    punkt_tab_english_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
    punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    punkt_pickle_path = os.path.join(punkt_dir, 'english.pickle')
    punkt_py3_pickle_path = os.path.join(punkt_dir, 'PY3', 'english.pickle')
    
    # Target file path
    target_pickle_path = os.path.join(punkt_tab_english_dir, 'punkt.pickle')
    
    # Ensure the directory exists
    os.makedirs(punkt_tab_english_dir, exist_ok=True)
    
    # Create an empty __init__.py file to make it a valid module
    with open(os.path.join(punkt_tab_english_dir, '__init__.py'), 'w') as f:
        pass
    logger.info(f"Created __init__.py file in {punkt_tab_english_dir}")
    
    # Copy the most appropriate English punkt.pickle file to punkt_tab/english/punkt.pickle
    source_pickle_path = None
    if os.path.exists(punkt_py3_pickle_path):
        source_pickle_path = punkt_py3_pickle_path
        logger.info(f"Using PY3 version of english.pickle")
    elif os.path.exists(punkt_pickle_path):
        source_pickle_path = punkt_pickle_path
        logger.info(f"Using standard version of english.pickle")
    
    if source_pickle_path:
        # Copy the pickle file with the right name
        shutil.copy2(source_pickle_path, target_pickle_path)
        logger.info(f"Copied {source_pickle_path} to {target_pickle_path}")
        
        # Also copy the pickle file with the original name in case it's needed
        original_name_path = os.path.join(punkt_tab_english_dir, 'english.pickle')
        shutil.copy2(source_pickle_path, original_name_path)
        logger.info(f"Copied {source_pickle_path} to {original_name_path}")
        
        print(f"✓ Successfully patched NLTK punkt_tab/english directory at {punkt_tab_english_dir}")
        print(f"  Created pickle files:\n  - {target_pickle_path}\n  - {original_name_path}")
        return True
    else:
        logger.error("Could not find any suitable English pickle file")
        print("✗ Failed to patch PDF processor. Could not find English pickle files.")
        return False

if __name__ == "__main__":
    print("Applying PDF processor patch for RAG dataset builder...")
    success = patch_pdf_processor()
    
    if success:
        print("\nThe patch has been applied. You should now be able to process PDF files.")
        print("This is a temporary fix. If you update the NLTK library, you may need to run this script again.")
    else:
        print("\nFailed to apply the patch. Please make sure you've run the install_nltk_dependencies.py script first.")
