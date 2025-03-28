#!/usr/bin/env python3
"""
PathRAG Runner with Arize Phoenix Integration

This script demonstrates how to use PathRAG with Arize Phoenix integration
for performance tracking and evaluation.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the PathRAG configuration
from config.pathrag_config import get_config, validate_config

# Import the PathRAG Arize Phoenix adapter
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
                f'pathrag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ingest_document(pathrag, document_path, **kwargs):
    """
    Ingest a document into PathRAG.
    
    Args:
        pathrag: PathRAG adapter instance
        document_path: Path to the document to ingest
        **kwargs: Additional arguments for ingestion
    """
    logger.info(f"Ingesting document: {document_path}")
    result = pathrag.ingest_file(document_path, **kwargs)
    logger.info(f"Ingestion result: {json.dumps(result, indent=2)}")
    return result

def query_pathrag(pathrag, query, **kwargs):
    """
    Query PathRAG and display the results.
    
    Args:
        pathrag: PathRAG adapter instance
        query: Query string
        **kwargs: Additional arguments for the query
    """
    logger.info(f"Processing query: {query}")
    result = pathrag.query(query, **kwargs)
    
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("-"*80)
    print(f"Answer: {result['answer']}")
    print("-"*80)
    print("Paths:")
    for i, path in enumerate(result['paths']):
        print(f"  Path {i+1}: {' -> '.join(path)}")
    print("-"*80)
    print(f"Latency: {result['metrics']['latency_ms']} ms")
    print(f"Trace ID: {result['metrics']['trace_id']}")
    print("="*80 + "\n")
    
    return result

def evaluate_query(pathrag, query, ground_truth, **kwargs):
    """
    Evaluate a query against ground truth.
    
    Args:
        pathrag: PathRAG adapter instance
        query: Query string
        ground_truth: Ground truth answer
        **kwargs: Additional arguments for evaluation
    """
    logger.info(f"Evaluating query: {query}")
    result = pathrag.evaluate_query(query, ground_truth, **kwargs)
    
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("-"*80)
    print(f"Answer: {result['answer']}")
    print("-"*80)
    print(f"Ground Truth: {ground_truth}")
    print("-"*80)
    print("Evaluation Metrics:")
    print(f"  BLEU Score: {result['evaluation']['bleu_score']}")
    print(f"  ROUGE-1 F1: {result['evaluation']['rouge_scores']['rouge-1']['f']}")
    print(f"  ROUGE-2 F1: {result['evaluation']['rouge_scores']['rouge-2']['f']}")
    print(f"  ROUGE-L F1: {result['evaluation']['rouge_scores']['rouge-l']['f']}")
    print("-"*80)
    print(f"Latency: {result['metrics']['latency_ms']} ms")
    print("="*80 + "\n")
    
    return result

def main():
    """Main function to run PathRAG with Arize Phoenix integration."""
    parser = argparse.ArgumentParser(description="PathRAG with Arize Phoenix Integration")
    parser.add_argument("--ingest", type=str, help="Path to document to ingest")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate query against ground truth")
    parser.add_argument("--ground-truth", type=str, help="Ground truth for evaluation")
    parser.add_argument("--session-id", type=str, default=None, help="Session ID for tracking")
    parser.add_argument("--user-id", type=str, default="anonymous", help="User ID for tracking")
    args = parser.parse_args()
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed. Please check your .env file and settings.")
        return 1
    
    # Get configuration
    config = get_config()
    
    # Initialize PathRAG with Arize Phoenix integration
    logger.info("Initializing PathRAG with Arize Phoenix integration")
    pathrag = PathRAGArizeAdapter(config)
    pathrag.initialize()
    
    # Set session ID if not provided
    if args.session_id is None:
        args.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Process commands
    if args.ingest:
        ingest_document(pathrag, args.ingest, 
                       user_id=args.user_id, session_id=args.session_id)
    
    if args.query:
        if args.evaluate and args.ground_truth:
            evaluate_query(pathrag, args.query, args.ground_truth,
                         user_id=args.user_id, session_id=args.session_id)
        else:
            query_pathrag(pathrag, args.query, 
                         user_id=args.user_id, session_id=args.session_id)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
