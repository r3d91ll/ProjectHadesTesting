#!/usr/bin/env python3
"""
PathRAG Phoenix Connector

A utility script that connects PathRAG to Arize Phoenix using the
LangChainInstrumentor for simplified telemetry collection.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pathrag_phoenix")

def setup_phoenix_integration():
    """
    Set up Phoenix integration with LangChainInstrumentor.
    """
    try:
        # Import the required packages for Phoenix integration
        from openinference.instrumentation.langchain import LangChainInstrumentor
        
        # Instrument LangChain
        LangChainInstrumentor().instrument()
        
        logger.info("✅ Successfully instrumented LangChain for Phoenix telemetry")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import required packages: {e}")
        logger.error("Please run: pip install arize-phoenix-otel openinference-instrumentation-langchain")
        return False

def run_pathrag(args):
    """
    Run PathRAG with Phoenix integration.
    
    Args:
        args: Command-line arguments
    """
    # Setup Phoenix integration first
    if not setup_phoenix_integration():
        return False
    
    # Find PathRAG runner script
    project_root = Path(__file__).resolve().parent.parent
    potential_paths = [
        project_root / "pathrag" / "src" / "pathrag_runner.py",
        project_root / "implementations" / "pathrag" / "pathrag_runner.py",
        project_root / "pathrag" / "pathrag_runner.py"
    ]
    
    pathrag_script = None
    for path in potential_paths:
        if path.exists():
            pathrag_script = path
            break
    
    if pathrag_script is None:
        logger.error("❌ Could not find PathRAG runner script")
        return False
    
    logger.info(f"Running PathRAG from: {pathrag_script}")
    
    # Import and run PathRAG
    pathrag_dir = str(pathrag_script.parent)
    if pathrag_dir not in sys.path:
        sys.path.append(pathrag_dir)
    
    # Change working directory to project root
    os.chdir(project_root)
    
    try:
        # We'll use the module import rather than subprocess to ensure
        # the instrumentation applies to the same Python process
        pathrag_module = __import__(pathrag_script.stem)
        
        # If this is the main entry point, call the main function
        if hasattr(pathrag_module, "main"):
            pathrag_module.main()
        else:
            logger.warning("⚠️ PathRAG module doesn't have a 'main' function")
            logger.warning("Running the module directly...")
            # This will execute the script, assuming it has a __name__ == "__main__" check
            exec(open(str(pathrag_script)).read())
        
        logger.info("✅ PathRAG execution completed")
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to run PathRAG: {e}")
        return False

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run PathRAG with Phoenix integration")
    parser.add_argument("--phoenix-port", type=int, default=8084,
                       help="Phoenix server port (default: 8084)")
    parser.add_argument("--phoenix-host", type=str, default="localhost",
                       help="Phoenix server host (default: localhost)")
    parser.add_argument("--query", type=str,
                       help="Query to run with PathRAG")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Set up Phoenix environment variables
    os.environ["PHOENIX_HOST"] = args.phoenix_host
    os.environ["PHOENIX_PORT"] = str(args.phoenix_port)
    
    logger.info(f"Setting up Phoenix integration at {args.phoenix_host}:{args.phoenix_port}")
    
    # Install required packages if not already installed
    try:
        import importlib.util
        if importlib.util.find_spec("openinference.instrumentation.langchain") is None:
            logger.info("Installing required packages for Phoenix integration...")
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "arize-phoenix-otel", "openinference-instrumentation-langchain"
            ])
    except Exception as e:
        logger.error(f"Failed to install required packages: {e}")
    
    # Run PathRAG with Phoenix integration
    success = run_pathrag(args)
    
    if success:
        logger.info("✅ PathRAG with Phoenix integration completed successfully")
        logger.info(f"📊 View traces at http://{args.phoenix_host}:{args.phoenix_port}")
    else:
        logger.error("❌ PathRAG with Phoenix integration failed")
        sys.exit(1)
