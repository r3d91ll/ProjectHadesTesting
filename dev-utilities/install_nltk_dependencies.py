#!/usr/bin/env python3
"""
NLTK Dependencies Installer

This utility script downloads required NLTK resources needed by the RAG dataset builder.
Place in dev-utilities/ as it's a one-time setup operation.
"""

import nltk
import os
import sys

def install_nltk_dependencies():
    """Install all required NLTK dependencies for the RAG dataset builder."""
    print("Installing NLTK dependencies...")
    
    # Create NLTK data directory in the project root to ensure it's found
    nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Configure NLTK to use our project directory
    nltk.data.path.insert(0, nltk_data_dir)
    
    # Required resources for PDF processing and text extraction
    resources = [
        'punkt',
        'punkt_tab',  # This specific resource is needed for PDF processing
        'averaged_perceptron_tagger',
        'stopwords',
        'wordnet',
        'vader_lexicon'
    ]
    
    # Download each resource
    for resource in resources:
        print(f"Downloading {resource}...")
        try:
            nltk.download(resource, download_dir=nltk_data_dir)
            print(f"✓ Successfully downloaded {resource}")
        except Exception as e:
            print(f"✗ Failed to download {resource}: {str(e)}")
    
    print(f"\nNLTK resources installed to: {nltk_data_dir}")
    print("Setup complete!")

if __name__ == "__main__":
    install_nltk_dependencies()
