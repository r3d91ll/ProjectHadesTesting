#!/usr/bin/env python3
"""
Phoenix Project Fixer

This script follows the exact pattern shown in the Phoenix UI to create a project
and send traces using OpenTelemetry with the proper project name.
"""
"""
Fix for Phoenix project name not being properly set in RAG Dataset Builder.

This script patches the dataset builder to ensure it uses the correct Phoenix project name
from the configuration file instead of the default 'default' project.
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# Add the rag-dataset-builder to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
rag_dataset_builder_dir = os.path.join(project_root, "rag-dataset-builder")
sys.path.append(rag_dataset_builder_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_phoenix_project")

# Import necessary components from the dataset builder
try:
    from src.utils.arize_integration import RAGDatasetBuilderArizeAdapter, get_arize_adapter
    import yaml
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root")
    sys.exit(1)

def initialize_phoenix_with_correct_project(config_path):
    """Initialize Phoenix with the correct project name from config"""
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
        
        # Create and initialize the adapter
        adapter = RAGDatasetBuilderArizeAdapter(
            project_name=project_name,
            enabled=True,
            phoenix_url=phoenix_url,
            batch_size=100
        )
        
        # Test by sending a simple record
        adapter.track_document_processing(
            document_id="test-document",
            document_path="test-path",
            document_type="test",
            processing_time=0.1,
            document_size=1000,
            metadata={"test": True, "config_path": config_path},
            success=True
        )
        
        # Force flush the records to Phoenix
        adapter.flush()
        
        logger.info(f"✅ Successfully connected to Phoenix with project: {project_name}")
        logger.info(f"Phoenix dashboard available at: {phoenix_url}")
        
        # Terminate any existing RAG dataset builder processes
        import psutil
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] != current_pid and 'python' in proc.info['name']:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'rag-dataset-builder' in cmdline and 'main.py' in cmdline:
                        logger.info(f"Terminating existing dataset builder process: {proc.info['pid']}")
                        try:
                            proc.terminate()
                        except Exception as e:
                            logger.warning(f"Could not terminate process: {e}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Create a command to restart the dataset builder with the correct project
        restart_cmd = f"""
        cd {rag_dataset_builder_dir} && 
        python src/main.py --config {os.path.basename(config_path)}
        """
        
        logger.info("To restart the dataset builder with the correct project name, run:")
        logger.info(restart_cmd)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return

if __name__ == "__main__":
    # Check if config path is provided
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default config path
        config_path = os.path.join(rag_dataset_builder_dir, "config/config.yaml")
    
    logger.info(f"Using config file: {config_path}")
    initialize_phoenix_with_correct_project(config_path)
