#!/usr/bin/env python3
"""
Arize Phoenix Startup Utility

This script helps with starting and managing Arize Phoenix for RAG performance tracking.
It follows the project convention of placing utilities in the dev-utilities directory.
"""

import os
import sys
import time
import socket
import argparse
import subprocess
import logging
from typing import Optional, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("arize_phoenix_utility")

def check_phoenix_installed() -> bool:
    """Check if Arize Phoenix is installed."""
    try:
        import arize.phoenix
        logger.info(f"Arize Phoenix found (version: {arize.phoenix.__version__})")
        return True
    except ImportError:
        logger.warning("Arize Phoenix not found in Python environment")
        return False

def install_phoenix() -> bool:
    """Install Arize Phoenix."""
    try:
        logger.info("Installing Arize Phoenix...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "arize-phoenix"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Arize Phoenix installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Arize Phoenix: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def check_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def is_phoenix_running(port: int = 8084) -> bool:
    """Check if Phoenix is running."""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/health", timeout=1)
        if response.status_code == 200:
            logger.info(f"Arize Phoenix is running at http://localhost:{port}")
            return True
        else:
            logger.warning(f"Arize Phoenix health check failed with status code {response.status_code}")
            return False
    except Exception as e:
        logger.debug(f"Arize Phoenix is not running: {e}")
        return False

def start_phoenix(port: int = 8084) -> Optional[subprocess.Popen]:
    """Start Arize Phoenix server."""
    if is_phoenix_running(port):
        logger.info(f"Arize Phoenix is already running on port {port}")
        return None
        
    if check_port_in_use(port):
        logger.error(f"Port {port} is in use by another process")
        return None
        
    try:
        logger.info(f"Starting Arize Phoenix on port {port}...")
        phoenix_process = subprocess.Popen(
            [sys.executable, "-m", "arize.phoenix.server", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for Phoenix to start
        max_attempts = 10
        for attempt in range(max_attempts):
            logger.info(f"Waiting for Phoenix to start (attempt {attempt+1}/{max_attempts})...")
            time.sleep(2)
            if is_phoenix_running(port):
                logger.info(f"Arize Phoenix started successfully at http://localhost:{port}")
                return phoenix_process
        
        logger.error("Failed to start Arize Phoenix within the expected time")
        phoenix_process.terminate()
        return None
    except Exception as e:
        logger.error(f"Error starting Arize Phoenix: {e}")
        return None

def update_rag_config(phoenix_host: str = "localhost", phoenix_port: int = 8084) -> Dict[str, Any]:
    """Update RAG configuration to use Arize Phoenix."""
    try:
        # Find config files that might contain Arize Phoenix settings
        config_paths = [
            os.path.join("pathrag", "config", "config.json"),
            os.path.join("pathrag", "config", "default_config.json"),
            os.path.expanduser("~/.pathrag/config.json")
        ]
        
        updated_configs = []
        
        for config_path in config_paths:
            full_path = os.path.join(os.getcwd(), config_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        config = json.load(f)
                    
                    # Update Phoenix settings
                    config["phoenix_host"] = phoenix_host
                    config["phoenix_port"] = phoenix_port
                    config["track_performance"] = True
                    
                    # Save updated config
                    with open(full_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    logger.info(f"Updated Arize Phoenix settings in {full_path}")
                    updated_configs.append(full_path)
                except Exception as e:
                    logger.error(f"Error updating config {full_path}: {e}")
        
        return {
            "success": len(updated_configs) > 0,
            "updated_files": updated_configs
        }
    except Exception as e:
        logger.error(f"Error updating RAG config: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Arize Phoenix Startup Utility")
    parser.add_argument("--port", type=int, default=8084,
                     help="Port to use for Arize Phoenix (default: 8084)")
    parser.add_argument("--update-config", action="store_true",
                     help="Update PathRAG configuration to use Arize Phoenix")
    parser.add_argument("--install", action="store_true",
                     help="Install Arize Phoenix if not already installed")
    
    args = parser.parse_args()
    
    # Check if Phoenix is installed
    phoenix_installed = check_phoenix_installed()
    
    # Install Phoenix if requested
    if not phoenix_installed and args.install:
        phoenix_installed = install_phoenix()
        
    if not phoenix_installed:
        logger.error("Arize Phoenix is not installed. Use --install to install it.")
        sys.exit(1)
    
    # Start Phoenix server
    phoenix_process = start_phoenix(args.port)
    
    # Update RAG configuration if requested
    if args.update_config:
        result = update_rag_config(phoenix_host="localhost", phoenix_port=args.port)
        if result["success"]:
            logger.info(f"Updated configuration in {len(result['updated_files'])} files")
        else:
            logger.warning("Failed to update RAG configuration")
    
    if phoenix_process:
        logger.info(f"Arize Phoenix is running at http://localhost:{args.port}")
        logger.info("Press Ctrl+C to stop the server")
        try:
            # Stream output from the Phoenix process
            for line in iter(phoenix_process.stdout.readline, ''):
                if line.strip():
                    logger.info(f"Phoenix: {line.strip()}")
        except KeyboardInterrupt:
            logger.info("Stopping Arize Phoenix...")
            phoenix_process.terminate()
            logger.info("Arize Phoenix stopped")
    else:
        if is_phoenix_running(args.port):
            logger.info(f"Arize Phoenix is already running at http://localhost:{args.port}")
        else:
            logger.error("Failed to start Arize Phoenix")

if __name__ == "__main__":
    main()
