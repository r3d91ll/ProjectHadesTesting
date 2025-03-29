#!/usr/bin/env python3
"""
Script to fix the PathRAGArizeAdapter Phoenix connection
This updates the port in your config and ensures proper import handling
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_phoenix")

def update_pathrag_config():
    """
    Update the PathRAG configuration to use the correct Phoenix port
    """
    # Find all config files that might configure Phoenix
    project_root = Path(__file__).resolve().parent.parent
    config_paths = [
        project_root / "implementations" / "pathrag" / "config" / "config.json",
        project_root / "implementations" / "pathrag" / "config" / "config.yaml",
        project_root / "implementations" / "pathrag" / "config.json",
        project_root / "implementations" / "pathrag" / "config.yaml"
    ]
    
    updated_files = 0
    
    for config_path in config_paths:
        if not config_path.exists():
            continue
            
        logger.info(f"Checking config file: {config_path}")
        
        try:
            # Handle JSON configs
            if config_path.name.endswith('.json'):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update Phoenix port if it exists
                if "phoenix_port" in config:
                    old_port = config["phoenix_port"]
                    config["phoenix_port"] = 8084
                    logger.info(f"Updating phoenix_port from {old_port} to 8084")
                elif "arize" in config and "phoenix_port" in config["arize"]:
                    old_port = config["arize"]["phoenix_port"]
                    config["arize"]["phoenix_port"] = 8084
                    logger.info(f"Updating arize.phoenix_port from {old_port} to 8084")
                else:
                    # Add Phoenix config if it doesn't exist
                    config["phoenix_port"] = 8084
                    config["phoenix_host"] = "localhost"
                    logger.info("Adding Phoenix configuration with port 8084")
                
                # Write updated config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                updated_files += 1
                logger.info(f"Updated config file: {config_path}")
            
            # Handle YAML configs
            elif config_path.name.endswith('.yaml') or config_path.name.endswith('.yml'):
                import yaml
                
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if config is None:
                    config = {}
                
                # Update Phoenix port if it exists
                if "phoenix_port" in config:
                    old_port = config["phoenix_port"]
                    config["phoenix_port"] = 8084
                    logger.info(f"Updating phoenix_port from {old_port} to 8084")
                elif "arize" in config and "phoenix_port" in config["arize"]:
                    old_port = config["arize"]["phoenix_port"]
                    config["arize"]["phoenix_port"] = 8084
                    logger.info(f"Updating arize.phoenix_port from {old_port} to 8084")
                else:
                    # Add Phoenix config if it doesn't exist
                    config["phoenix_port"] = 8084
                    config["phoenix_host"] = "localhost"
                    logger.info("Adding Phoenix configuration with port 8084")
                
                # Write updated config
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                updated_files += 1
                logger.info(f"Updated config file: {config_path}")
        
        except Exception as e:
            logger.error(f"Error updating config {config_path}: {e}")
    
    return updated_files

def run_pathrag_with_phoenix():
    """
    Run the PathRAG system with Phoenix integration enabled on the correct port
    """
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Find the PathRAG main script
    pathrag_scripts = [
        project_root / "implementations" / "pathrag" / "pathrag_runner.py",
        project_root / "implementations" / "pathrag" / "main.py",
        project_root / "pathrag" / "pathrag_runner.py",
        project_root / "pathrag" / "main.py"
    ]
    
    main_script = None
    for script in pathrag_scripts:
        if script.exists():
            main_script = script
            break
    
    if main_script is None:
        logger.error("Could not find the PathRAG main script")
        return False
    
    logger.info(f"Found PathRAG main script: {main_script}")
    logger.info("To run PathRAG with Phoenix on port 8084, use:")
    logger.info(f"source .venv/bin/activate && python {main_script.relative_to(project_root)} --track_performance=true")
    
    return True

if __name__ == "__main__":
    logger.info("Fixing PathRAG Phoenix connection...")
    
    # Update config files
    updated = update_pathrag_config()
    logger.info(f"Updated {updated} config files")
    
    # Print instructions
    success = run_pathrag_with_phoenix()
    
    if updated > 0 and success:
        logger.info("✅ Phoenix connection fix complete!")
        logger.info("Phoenix UI is available at: http://localhost:8084")
    else:
        logger.error("⚠️ Phoenix fix partially complete - you may need to manually update your config.")
        logger.error("Make sure your Phoenix port is set to 8084 in your PathRAG config.")
        logger.error("Phoenix UI is available at: http://localhost:8084")
