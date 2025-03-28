#!/usr/bin/env python3
"""
PathRAG Example with New Configuration System

This script demonstrates how to use the PathRAG implementation with the
new technology-agnostic configuration system.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_loader import get_configuration
from src.core.plugin import discover_plugins, register, create_retrieval_system
from src.trackers.arize_phoenix_tracker import ArizePhoenixTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pathrag_example")

def setup_environment():
    """Set up necessary environment for running PathRAG."""
    # Create data directories if they don't exist
    os.makedirs("./data/input", exist_ok=True)
    os.makedirs("./data/output", exist_ok=True)

    # Set environment variables if needed from .env file
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        logger.info(f"Loading environment variables from {env_file}")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
                    logger.debug(f"Set {key}={value}")

def create_custom_config():
    """Create a custom configuration file for this example."""
    custom_config_path = os.path.join(os.path.dirname(__file__), "pathrag_custom.yaml")
    
    if not os.path.exists(custom_config_path):
        with open(custom_config_path, "w") as f:
            f.write("""# Custom PathRAG Configuration
directories:
  input: "./data/input"
  output: "./data/output"
  logs: "./logs"
  cache: "./.cache"

# Override PathRAG settings for this example
retrieval_systems:
  pathrag:
    type: "pathrag"
    storage_backend: "networkx"
    embedder: "sentence_transformers"  # Using sentence_transformers instead of OpenAI
    chunker: "sliding_window"
    processor: "pdf"
    
    # Path generation and retrieval settings
    path_retrieval:
      max_paths: 3
      max_path_length: 2
      similarity_threshold: 0.75
      path_ranking: "combined"

# Arize Phoenix monitoring configuration
monitoring:
  arize_phoenix:
    enabled: true
    project_name: "pathrag-example"
    server_port: 8080
    track_system_resources: true
    track_gpu_metrics: true
""")
        logger.info(f"Created custom configuration at {custom_config_path}")
    
    return custom_config_path

def main():
    """Run the PathRAG example."""
    # Set up environment
    setup_environment()
    
    # Discover plugins
    discover_plugins()
    
    # Create custom configuration
    custom_config_path = create_custom_config()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = get_configuration(user_config=custom_config_path)
    
    # Initialize performance tracker (Arize Phoenix)
    logger.info("Initializing Arize Phoenix tracker...")
    tracker = None
    if config.get("monitoring", {}).get("arize_phoenix", {}).get("enabled", False):
        tracker = ArizePhoenixTracker(
            project_name=config["monitoring"]["arize_phoenix"].get("project_name", "pathrag-example"),
            server_port=config["monitoring"]["arize_phoenix"].get("server_port", 8080)
        )
    
    # Initialize PathRAG
    logger.info("Initializing PathRAG...")
    pathrag = create_retrieval_system(
        name="pathrag",
        tracker=tracker
    )
    
    # Configure PathRAG with our settings
    pathrag_config = config["retrieval_systems"]["pathrag"]
    pathrag.configure(pathrag_config)
    
    # Print configuration summary
    logger.info("PathRAG configuration summary:")
    print(f"  Storage backend: {pathrag_config['storage_backend']}")
    print(f"  Embedder: {pathrag_config['embedder']}")
    print(f"  Chunker: {pathrag_config['chunker']}")
    print(f"  Processor: {pathrag_config['processor']}")
    print(f"  Max paths: {pathrag_config.get('path_retrieval', {}).get('max_paths', 5)}")
    print(f"  Max path length: {pathrag_config.get('path_retrieval', {}).get('max_path_length', 3)}")
    
    logger.info("PathRAG has been initialized with the new configuration system.")
    logger.info("This example is now ready to process documents and perform retrieval.")
    
    # Instructions for the next steps
    print("\nTo process documents:")
    print("1. Place PDF files in the './data/input' directory")
    print("2. Call pathrag.process_documents()")
    print("3. For retrieval, use pathrag.retrieve(query)")
    print("\nTo view Arize Phoenix monitoring dashboard (if enabled):")
    print("  http://localhost:8080\n")

if __name__ == "__main__":
    main()
