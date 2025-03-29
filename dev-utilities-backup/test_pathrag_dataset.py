#!/usr/bin/env python3
"""
Test script to verify PathRAG's connection to the RAG dataset and its retrieval capabilities.
"""

import os
import sys
import json
from pathlib import Path

# Add the pathrag directory to the Python path
pathrag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "pathrag")
sys.path.insert(0, pathrag_path)

# Import PathRAG modules
from config.pathrag_config import get_config, validate_config

# Add the src directory to the Python path
src_path = os.path.join(pathrag_path, "src")
sys.path.insert(0, src_path)

# Import the PathRAG adapter directly from the pathrag_runner module
from pathrag_runner import PathRAGArizeAdapter

def test_pathrag_dataset():
    """Test PathRAG's connection to the dataset and its retrieval capabilities."""
    print("🔍 Testing PathRAG's connection to the RAG dataset...")
    
    # Get the configuration
    config = get_config()
    print(f"📂 Document store path: {config['document_store_path']}")
    
    # Initialize PathRAG
    if not validate_config():
        print("❌ Configuration validation failed. Please check your .env file and settings.")
        return
        
    # Initialize PathRAG with Arize Phoenix integration
    print("🔄 Initializing PathRAG with Arize Phoenix integration")
    pathrag = PathRAGArizeAdapter(config)
    pathrag.initialize()
    
    # Check if the dataset exists
    dataset_path = config['document_store_path']
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    print(f"✅ Dataset found at {dataset_path}")
    
    # List the contents of the dataset directory
    print("\n📊 Dataset contents:")
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item} (directory with {len(os.listdir(item_path))} items)")
        else:
            print(f"  📄 {item} ({os.path.getsize(item_path)} bytes)")
    
    # Test queries
    test_queries = [
        "What are the key principles of RAG systems?",
        "How does PathRAG improve retrieval relevance?",
        "What is the transformer architecture?",
        "Explain the XnX notation system"
    ]
    
    print("\n🔍 Testing queries:")
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        
        try:
            # Get the response from PathRAG
            response = pathrag.query(query, session_id="test_dataset_debug")
            
            # Print the response
            print(f"🔄 Response: {response}")
            
            # Try to get the retrieved documents
            if hasattr(pathrag, "get_retrieved_documents"):
                docs = pathrag.get_retrieved_documents()
                print(f"📚 Retrieved {len(docs)} documents")
                for i, doc in enumerate(docs):
                    print(f"  📄 Document {i+1}: {doc.get('title', 'Untitled')} ({len(doc.get('content', ''))} chars)")
            else:
                print("❌ PathRAG does not have a get_retrieved_documents method")
                
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
    
    print("\n✅ PathRAG dataset test complete")

if __name__ == "__main__":
    test_pathrag_dataset()
