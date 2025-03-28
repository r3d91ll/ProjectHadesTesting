#!/usr/bin/env python3
"""
Direct fix for Phoenix integration to use the correct project name.
This script will update the environment variables to ensure the correct project name is used.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("direct_phoenix_fix")

def fix_phoenix_project(config_path):
    """Set environment variables directly to fix Phoenix project name"""
    try:
        # Load the configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract Phoenix configuration
        tracking_config = config.get("performance_tracking", {})
        enabled = tracking_config.get("enabled", False)
        project_name = tracking_config.get("project_name", "pathrag-dataset-builder")
        phoenix_url = tracking_config.get("phoenix_url", "http://localhost:8084")
        
        if not enabled:
            logger.warning("Performance tracking is disabled in the configuration")
            return
        
        # Set environment variables that Phoenix will use
        os.environ["PHOENIX_PROJECT_NAME"] = project_name
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_url
        
        logger.info(f"✅ Set Phoenix environment variables:")
        logger.info(f"  - PHOENIX_PROJECT_NAME = {project_name}")
        logger.info(f"  - PHOENIX_COLLECTOR_ENDPOINT = {phoenix_url}")
        
        # Write these to a file that can be sourced before running the dataset builder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        phoenix_env_path = os.path.join(script_dir, "phoenix_env.sh")
        
        with open(phoenix_env_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'export PHOENIX_PROJECT_NAME="{project_name}"\n')
            f.write(f'export PHOENIX_COLLECTOR_ENDPOINT="{phoenix_url}"\n')
        
        os.chmod(phoenix_env_path, 0o755)  # Make it executable
        
        logger.info(f"✅ Created Phoenix environment script at: {phoenix_env_path}")
        logger.info("To use this configuration for any dataset builder run, execute:")
        logger.info(f"source {phoenix_env_path} && cd rag-dataset-builder && python src/main.py --config config/config.yaml")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return

if __name__ == "__main__":
    # Check if config path is provided
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default config path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, "rag-dataset-builder/config/config.yaml")
    
    logger.info(f"Using config file: {config_path}")
    fix_phoenix_project(config_path)
