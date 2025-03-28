#!/usr/bin/env python3
"""
Configuration System Demo

This script demonstrates how to use the new configuration system in the
RAG Dataset Builder framework.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_loader import get_configuration
from src.core.plugin import discover_plugins

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("config_demo")

def main():
    """Run the configuration demo."""
    # Discover plugins
    discover_plugins()
    
    # Load configuration from config.d directory
    logger.info("Loading default configuration from config.d/")
    config = get_configuration()
    
    # Print some configuration sections
    logger.info("Directories configuration:")
    print(f"  Input directory: {config['directories']['input']}")
    print(f"  Output directory: {config['directories']['output']}")
    print(f"  Cache directory: {config['directories']['cache']}")
    print()
    
    # Print available processors
    logger.info("Configured document processors:")
    for processor_name, processor_config in config.get("processors", {}).items():
        print(f"  {processor_name}: {processor_config['type']}")
    print()
    
    # Load a user-provided configuration
    logger.info("Loading user configuration and merging with defaults")
    user_config_path = os.path.join(
        os.path.dirname(__file__), 
        "user_config_example.yaml"
    )
    
    # Create a simple user config if it doesn't exist
    if not os.path.exists(user_config_path):
        with open(user_config_path, "w") as f:
            f.write("""# User Configuration Example
directories:
  input: "./custom_data/input"
  output: "./custom_data/output"

processors:
  custom_pdf:
    type: "pdf"
    extract_metadata: true
    ocr_enabled: true
    max_pages: 100
""")
        logger.info(f"Created example user config at {user_config_path}")
    
    # Load with user config
    merged_config = get_configuration(user_config=user_config_path)
    
    # Print the overridden values
    logger.info("Configuration after user overrides:")
    print(f"  Input directory: {merged_config['directories']['input']}")
    print(f"  Output directory: {merged_config['directories']['output']}")
    print()
    
    # Print retrieval system configs
    logger.info("Available retrieval systems:")
    for system_name, system_config in merged_config.get("retrieval_systems", {}).items():
        print(f"  {system_name}:")
        print(f"    Type: {system_config['type']}")
        print(f"    Storage backend: {system_config.get('storage_backend', 'default')}")
        print(f"    Embedder: {system_config.get('embedder', 'default')}")
        print()
    
    # Demonstrate environment variable resolution
    os.environ["CUSTOM_DATA_DIR"] = "/tmp/rag_custom_data"
    
    # Create a config with environment variable
    env_config_path = os.path.join(
        os.path.dirname(__file__), 
        "env_config_example.yaml"
    )
    
    # Create an environment variable config if it doesn't exist
    if not os.path.exists(env_config_path):
        with open(env_config_path, "w") as f:
            f.write("""# Environment Variable Configuration Example
directories:
  input: "${CUSTOM_DATA_DIR}/input"
  output: "${CUSTOM_DATA_DIR}/output"
""")
        logger.info(f"Created example env config at {env_config_path}")
    
    # Load with environment variable config
    env_config = get_configuration(user_config=env_config_path)
    
    # Print the resolved environment variables
    logger.info("Configuration with resolved environment variables:")
    print(f"  Input directory: {env_config['directories']['input']}")
    print(f"  Output directory: {env_config['directories']['output']}")
    print()
    
    logger.info("Configuration system demo complete!")

if __name__ == "__main__":
    main()
