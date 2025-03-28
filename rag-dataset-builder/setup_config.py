#!/usr/bin/env python3
"""
Configuration Setup Script for RAG Dataset Builder

This script initializes the configuration system for the RAG Dataset Builder framework.
It creates the directory structure for config.d/ if it doesn't exist and copies
example configuration files from src/ to config.d/ for customization.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_dataset_builder.setup_config")

def create_config_structure():
    """Create the config.d/ directory structure if it doesn't exist."""
    config_dir = Path("config.d")
    if not config_dir.exists():
        logger.info("Creating config.d/ directory structure...")
        config_dir.mkdir(exist_ok=True)
    else:
        logger.info("config.d/ directory already exists")
    
    return config_dir

def ensure_default_configs_exist(config_dir):
    """
    Ensure all default configuration files exist in config.d/.
    If they don't exist, copy them from the distribution.
    """
    # Define the core configuration files
    core_configs = [
        "00-default.yaml",
        "10-processors.yaml", 
        "20-chunkers.yaml",
        "30-embedders.yaml",
        "40-storage.yaml",
        "50-retrieval.yaml",
        "60-monitoring.yaml"
    ]
    
    # Check each file and copy if needed
    for config_file in core_configs:
        config_path = config_dir / config_file
        if not config_path.exists():
            logger.info(f"Creating default configuration file: {config_file}")
            
            # If running from source, copy from the current config.d directory
            src_path = Path(__file__).parent / "config.d" / config_file
            if src_path.exists():
                shutil.copy(src_path, config_path)
                logger.info(f"Copied {config_file} from distribution")
            else:
                logger.error(f"Could not find source for {config_file}")
                # Create empty file as placeholder
                with open(config_path, "w") as f:
                    f.write(f"# {config_file}\n# This is a placeholder configuration file\n")
                logger.info(f"Created empty placeholder for {config_file}")

def create_data_directories():
    """Create the necessary data directories if they don't exist."""
    data_dirs = [
        "data",
        "data/input",
        "data/output",
        "data/chunks",
        "data/embeddings",
        "logs",
        ".cache"
    ]
    
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if not dir_path.exists():
            logger.info(f"Creating directory: {data_dir}")
            dir_path.mkdir(exist_ok=True)
            
            # Add .gitkeep to maintain directory structure in git
            gitkeep_path = dir_path / ".gitkeep"
            if not gitkeep_path.exists():
                with open(gitkeep_path, "w") as f:
                    pass
                logger.debug(f"Created .gitkeep in {data_dir}")

def copy_implementation_examples():
    """Copy implementation-specific example configurations if they exist."""
    # Define implementation directories to check
    impl_dirs = [
        ("src/implementations/pathrag", "pathrag"),
        ("src/storage/neo4j", "neo4j")
    ]
    
    examples_dir = Path("config.d/examples")
    if not examples_dir.exists():
        logger.info("Creating examples directory in config.d/")
        examples_dir.mkdir(exist_ok=True)
    
    # Check each implementation directory for example configs
    for impl_dir, impl_name in impl_dirs:
        impl_path = Path(impl_dir)
        if impl_path.exists():
            for file in impl_path.glob("*config*.yaml"):
                dest_file = examples_dir / f"{impl_name}-{file.name}"
                if not dest_file.exists():
                    logger.info(f"Copying example config: {file.name} to examples/{dest_file.name}")
                    shutil.copy(file, dest_file)

def main():
    """Run the configuration setup script."""
    logger.info("Setting up RAG Dataset Builder configuration...")
    
    # Create config.d/ directory
    config_dir = create_config_structure()
    
    # Ensure default configs exist
    ensure_default_configs_exist(config_dir)
    
    # Create data directories
    create_data_directories()
    
    # Copy implementation-specific examples
    copy_implementation_examples()
    
    logger.info("Configuration setup complete!")
    logger.info("You can now edit the files in config.d/ to customize your configuration.")
    logger.info("See config.d/README.md for more information on the configuration system.")

if __name__ == "__main__":
    main()
