#!/usr/bin/env python3
"""
PathRAG Database Arize Phoenix Loader

This script loads the completed PathRAG database into Arize Phoenix
for visualization and analytics.
"""

import os
import sys
import json
import time
import logging
import argparse
import pickle
import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pathrag_arize_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pathrag_arize_loader")

# Load environment variables
load_dotenv()

# Import Arize Phoenix integration
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                               "implementations", "pathrag", "arize_integration"))
try:
    from adapter import PathRAGArizeAdapter
    ARIZE_AVAILABLE = True
except ImportError:
    ARIZE_AVAILABLE = False
    logger.warning("Arize Phoenix adapter not available. Please ensure the adapter is available.")
    logger.warning("Check in: /implementations/pathrag/arize_integration/adapter.py")
    sys.exit(1)


class PathRAGDatabaseLoader:
    """
    Utility for loading PathRAG database into Arize Phoenix for visualization and analytics.
    """
    
    def __init__(self, database_dir: str, phoenix_host: str = "localhost", phoenix_port: int = 8084):
        """
        Initialize the PathRAG database loader.
        
        Args:
            database_dir: Path to the PathRAG database directory
            phoenix_host: Arize Phoenix host
            phoenix_port: Arize Phoenix port
        """
        self.database_dir = database_dir
        self.phoenix_host = phoenix_host
        self.phoenix_port = phoenix_port
        
        # Directory paths
        self.chunks_dir = os.path.join(database_dir, "chunks")
        self.metadata_dir = os.path.join(database_dir, "metadata")
        self.embeddings_dir = os.path.join(database_dir, "embeddings")
        self.graph_dir = os.path.join(database_dir, "graph")
        
        # Check if database exists
        for directory in [self.chunks_dir, self.metadata_dir, self.embeddings_dir, self.graph_dir]:
            if not os.path.exists(directory):
                raise ValueError(f"Database directory {directory} does not exist")
        
        # Initialize Arize Phoenix connection
        try:
            self.arize_adapter = PathRAGArizeAdapter({
                "phoenix_host": phoenix_host,
                "phoenix_port": phoenix_port,
                "track_performance": True,
                # General configuration for PathRAG
                "model_name": "pathrag-database-loader",
                "embedding_dim": 384  # Using all-MiniLM-L6-v2 model
            })
            logger.info(f"Connected to Arize Phoenix at {phoenix_host}:{phoenix_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Arize Phoenix: {e}")
            raise
    
    def load_graph(self) -> nx.DiGraph:
        """
        Load the knowledge graph from disk.
        
        Returns:
            Knowledge graph
        """
        graph_file = os.path.join(self.graph_dir, "knowledge_graph.pickle")
        logger.info(f"Loading knowledge graph from {graph_file}")
        
        try:
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f)
            logger.info(f"Loaded knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            raise
    
    def log_graph_summary(self, graph: nx.DiGraph) -> None:
        """
        Log knowledge graph summary to Arize Phoenix.
        
        Args:
            graph: Knowledge graph
        """
        # Get node types
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Get edge types
        edge_types = {}
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Log summary trace
        trace_id = f"graph-summary-{datetime.now().isoformat()}"
        trace_data = {
            "id": trace_id,
            "name": "Knowledge Graph Summary",
            "model": "pathrag-database-loader",
            "input": "Generate graph summary",
            "output": f"Graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "metadata": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "node_types": json.dumps(node_types),
                "edge_types": json.dumps(edge_types),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        self.arize_adapter._log_to_phoenix(trace_data)
        logger.info(f"Logged graph summary to Arize Phoenix with trace ID {trace_id}")
    
    def log_embedding_clusters(self, sample_size: int = 1000) -> None:
        """
        Log embedding clusters to Arize Phoenix.
        
        Args:
            sample_size: Number of embeddings to sample
        """
        import random
        import numpy as np
        from sklearn.manifold import TSNE
        
        # Get list of embedding files
        embedding_files = os.listdir(self.embeddings_dir)
        if len(embedding_files) == 0:
            logger.warning("No embeddings found")
            return
        
        # Sample embedding files if there are too many
        if len(embedding_files) > sample_size:
            embedding_files = random.sample(embedding_files, sample_size)
        
        # Load embeddings
        embeddings = []
        chunk_ids = []
        
        logger.info(f"Loading {len(embedding_files)} embeddings")
        for file_name in tqdm(embedding_files):
            if not file_name.endswith('.json'):
                continue
                
            file_path = os.path.join(self.embeddings_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    embeddings.append(data["embedding"])
                    chunk_ids.append(data["id"])
            except Exception as e:
                logger.error(f"Failed to load embedding {file_path}: {e}")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply dimensionality reduction for visualization
        logger.info("Applying t-SNE for dimensionality reduction")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Create DataFrame for Arize Phoenix
        df = pd.DataFrame({
            "chunk_id": chunk_ids,
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1]
        })
        
        # Log embeddings to Arize Phoenix
        trace_id = f"embedding-clusters-{datetime.now().isoformat()}"
        trace_data = {
            "id": trace_id,
            "name": "Embedding Clusters",
            "model": "pathrag-database-loader",
            "input": "Generate embedding clusters",
            "output": f"Generated 2D projection of {len(embeddings)} embeddings",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "metadata": {
                "num_embeddings": len(embeddings),
                "clusters_data": df.to_json(orient="records"),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        self.arize_adapter._log_to_phoenix(trace_data)
        logger.info(f"Logged embedding clusters to Arize Phoenix with trace ID {trace_id}")
    
    def log_document_metadata(self) -> None:
        """
        Log document metadata to Arize Phoenix.
        """
        # Get list of metadata files
        metadata_files = os.listdir(self.metadata_dir)
        if len(metadata_files) == 0:
            logger.warning("No metadata found")
            return
        
        # Count documents by category
        categories = {}
        
        logger.info(f"Processing metadata for {len(metadata_files)} documents")
        for file_name in tqdm(metadata_files):
            if not file_name.endswith('.json'):
                continue
                
            file_path = os.path.join(self.metadata_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    category = data.get("category", "unknown")
                    categories[category] = categories.get(category, 0) + 1
            except Exception as e:
                logger.error(f"Failed to load metadata {file_path}: {e}")
        
        # Log document metadata to Arize Phoenix
        trace_id = f"document-metadata-{datetime.now().isoformat()}"
        trace_data = {
            "id": trace_id,
            "name": "Document Metadata",
            "model": "pathrag-database-loader",
            "input": "Generate document metadata summary",
            "output": f"Processed metadata for {len(metadata_files)} documents",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "metadata": {
                "num_documents": len(metadata_files),
                "categories": json.dumps(categories),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        self.arize_adapter._log_to_phoenix(trace_data)
        logger.info(f"Logged document metadata to Arize Phoenix with trace ID {trace_id}")
    
    def run(self) -> None:
        """
        Run the database loader.
        """
        # Load knowledge graph
        graph = self.load_graph()
        
        # Log graph summary
        self.log_graph_summary(graph)
        
        # Log embedding clusters
        self.log_embedding_clusters()
        
        # Log document metadata
        self.log_document_metadata()
        
        logger.info("Database loading complete")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Load PathRAG database into Arize Phoenix")
    parser.add_argument("--database-dir", type=str, default="../database",
                      help="Path to the PathRAG database directory")
    parser.add_argument("--phoenix-host", type=str, default="localhost",
                      help="Arize Phoenix host")
    parser.add_argument("--phoenix-port", type=int, default=8084,
                      help="Arize Phoenix port")
    
    args = parser.parse_args()
    
    # Initialize and run loader
    try:
        loader = PathRAGDatabaseLoader(
            database_dir=args.database_dir,
            phoenix_host=args.phoenix_host,
            phoenix_port=args.phoenix_port
        )
        loader.run()
    except Exception as e:
        logger.error(f"Database loading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
