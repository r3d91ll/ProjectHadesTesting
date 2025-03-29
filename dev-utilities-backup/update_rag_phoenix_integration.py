#!/usr/bin/env python3
"""
RAG Dataset Builder Phoenix Integration Updater

This script modifies the relevant files in the RAG dataset builder to ensure
proper Phoenix project naming. It takes the approach of directly modifying
the code to use a hardcoded project name instead of relying on environment
variables that might not be properly set.
"""

import os
import re
import sys
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("update_phoenix_integration")

# Constants
PROJECT_NAME = "pathrag-dataset-builder"
PHOENIX_URL = "http://localhost:8084"

def update_arize_integration_file(file_path):
    """Update the arize_integration.py file to use hardcoded project name"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the RAGDatasetBuilderArizeAdapter class and modify its __init__ method
        pattern = r'def __init__\(self, project_name: str = "[^"]*", enabled: bool = True, phoenix_url: str = "[^"]*"\):'
        replacement = f'def __init__(self, project_name: str = "{PROJECT_NAME}", enabled: bool = True, phoenix_url: str = "{PHOENIX_URL}"):'
        
        if not re.search(pattern, content):
            logger.warning("Could not find the __init__ method pattern to replace")
            # Less specific pattern as fallback
            pattern = r'def __init__\(self, project_name:.+?, enabled:.+?, phoenix_url:.+?\):'
            if not re.search(pattern, content):
                logger.error("Could not find the __init__ method to modify")
                return False
        
        modified_content = re.sub(pattern, replacement, content)
        
        # Also update the log_records method to force the project name
        log_records_pattern = r'(\s+self\.client\.log_records\(\s*?project_name=self\.project_name)'
        log_records_replacement = f'\\1  # Using project name from instance\n            # Force project name to be consistent\n            project_name="{PROJECT_NAME}"'
        
        if re.search(log_records_pattern, modified_content):
            modified_content = re.sub(log_records_pattern, log_records_replacement, modified_content)
        else:
            logger.warning("Could not find log_records method to update")
        
        # Write the modified content back
        with open(file_path, 'w') as f:
            f.write(modified_content)
        
        logger.info(f"✅ Updated {file_path} to use project name: {PROJECT_NAME}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating {file_path}: {e}")
        return False

def create_environment_script(project_root):
    """Create a shell script to set the Phoenix environment variables"""
    try:
        script_path = os.path.join(project_root, "dev-utilities/phoenix_env.sh")
        
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('# Phoenix environment variables for RAG dataset builder\n')
            f.write(f'export PHOENIX_PROJECT_NAME="{PROJECT_NAME}"\n')
            f.write(f'export PHOENIX_COLLECTOR_ENDPOINT="{PHOENIX_URL}"\n')
            f.write('\n# Print the environment variables for confirmation\n')
            f.write('echo "Phoenix environment variables set:"\n')
            f.write('echo "  PHOENIX_PROJECT_NAME=${PHOENIX_PROJECT_NAME}"\n')
            f.write('echo "  PHOENIX_COLLECTOR_ENDPOINT=${PHOENIX_COLLECTOR_ENDPOINT}"\n')
        
        os.chmod(script_path, 0o755)  # Make it executable
        
        logger.info(f"✅ Created Phoenix environment script at: {script_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating environment script: {e}")
        return False

def create_runner_script(project_root):
    """Create a shell script to run the dataset builder with proper Phoenix setup"""
    try:
        script_path = os.path.join(project_root, "dev-utilities/run_pathrag_with_phoenix.sh")
        
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('# Run the RAG dataset builder with Phoenix integration\n\n')
            f.write('# Set environment variables\n')
            f.write(f'export PHOENIX_PROJECT_NAME="{PROJECT_NAME}"\n')
            f.write(f'export PHOENIX_COLLECTOR_ENDPOINT="{PHOENIX_URL}"\n\n')
            f.write('# Print confirmation\n')
            f.write('echo "Running RAG dataset builder with Phoenix project: ${PHOENIX_PROJECT_NAME}"\n\n')
            f.write('# Navigate to dataset builder directory and run it\n')
            f.write('cd "$(dirname "$0")/../rag-dataset-builder" || exit 1\n')
            f.write('python src/main.py --config config/config.yaml "$@"\n')
        
        os.chmod(script_path, 0o755)  # Make it executable
        
        logger.info(f"✅ Created runner script at: {script_path}")
        logger.info(f"You can now run the dataset builder with correct Phoenix integration using:")
        logger.info(f"./dev-utilities/run_pathrag_with_phoenix.sh")
        return True
        
    except Exception as e:
        logger.error(f"Error creating runner script: {e}")
        return False

def update_get_arize_adapter(file_path):
    """Update the get_arize_adapter function to use the hardcoded project name"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the get_arize_adapter function and modify it
        pattern = r'(def get_arize_adapter\(config:.+?\):.*?\n.*?project_name = [^\n]+)'
        replacement = f'\\1\n    # Override with hardcoded project name\n    project_name = "{PROJECT_NAME}"'
        
        if not re.search(pattern, content, re.DOTALL):
            logger.warning("Could not find the get_arize_adapter function pattern to replace")
            return False
        
        modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Write the modified content back
        with open(file_path, 'w') as f:
            f.write(modified_content)
        
        logger.info(f"✅ Updated get_arize_adapter in {file_path} to use project name: {PROJECT_NAME}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update the Phoenix integration"""
    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    logger.info(f"Project root: {project_root}")
    
    # Update key files
    arize_integration_path = os.path.join(project_root, "rag-dataset-builder/src/utils/arize_integration.py")
    update_arize_integration_file(arize_integration_path)
    update_get_arize_adapter(arize_integration_path)
    
    # Create helper scripts
    create_environment_script(project_root)
    create_runner_script(project_root)
    
    logger.info("Phoenix integration update complete!")
    logger.info("To run the dataset builder with proper Phoenix integration, use:")
    logger.info(f"./dev-utilities/run_pathrag_with_phoenix.sh")

if __name__ == "__main__":
    main()
