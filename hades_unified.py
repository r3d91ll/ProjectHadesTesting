#!/usr/bin/env python3
"""
HADES Unified Startup Script

This script provides a unified interface for running the HADES system with Redis integration.
It supports both dataset creation/embedding and inference/retrieval modes.

Usage:
    # Dataset creation mode
    python hades_unified.py create --source-dir /path/to/source --output-dir /path/to/output [--gpu]
    
    # Inference mode (future)
    python hades_unified.py infer --database /path/to/database [--gpu]
    
    # Retrieval mode (future)
    python hades_unified.py retrieve --database /path/to/database --query "your query" [--gpu]
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import json
import yaml
import redis
import math
import functools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
def setup_logging():
    """Set up logging with file and console handlers."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file path with timestamp
    log_file = os.path.join(logs_dir, f"hades_unified_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:  
        root_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

# Basic logging setup for initial script execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": "",
    "prefix": "pathrag",
    "ttl": 604800  # 7 days in seconds
}

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RAG_BUILDER_DIR = os.path.join(PROJECT_ROOT, "rag-dataset-builder")
DEFAULT_SOURCE_DIR = os.path.join(PROJECT_ROOT, "source_documents")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "rag_databases")
DATABASES_DIR = os.path.join(PROJECT_ROOT, "rag_databases")

def set_redis_env_vars():
    """Set Redis environment variables."""
    os.environ["PATHRAG_REDIS_ENABLED"] = "true"
    os.environ["PATHRAG_REDIS_HOST"] = REDIS_CONFIG["host"]
    os.environ["PATHRAG_REDIS_PORT"] = str(REDIS_CONFIG["port"])
    os.environ["PATHRAG_REDIS_DB"] = str(REDIS_CONFIG["db"])
    os.environ["PATHRAG_REDIS_PREFIX"] = REDIS_CONFIG["prefix"]
    os.environ["PATHRAG_REDIS_TTL"] = str(REDIS_CONFIG["ttl"])
    if REDIS_CONFIG["password"]:
        os.environ["PATHRAG_REDIS_PASSWORD"] = REDIS_CONFIG["password"]
    
    logger.info("Redis environment variables set:")
    logger.info(f"  PATHRAG_REDIS_ENABLED: {os.environ['PATHRAG_REDIS_ENABLED']}")
    logger.info(f"  PATHRAG_REDIS_HOST: {os.environ['PATHRAG_REDIS_HOST']}")
    logger.info(f"  PATHRAG_REDIS_PORT: {os.environ['PATHRAG_REDIS_PORT']}")
    logger.info(f"  PATHRAG_REDIS_DB: {os.environ['PATHRAG_REDIS_DB']}")
    logger.info(f"  PATHRAG_REDIS_PREFIX: {os.environ['PATHRAG_REDIS_PREFIX']}")
    logger.info(f"  PATHRAG_REDIS_TTL: {os.environ['PATHRAG_REDIS_TTL']}")

def connect_to_redis():
    """Connect to Redis and return client."""
    try:
        client = redis.Redis(
            host=REDIS_CONFIG["host"],
            port=REDIS_CONFIG["port"],
            db=REDIS_CONFIG["db"],
            password=REDIS_CONFIG["password"] if REDIS_CONFIG["password"] else None
        )
        client.ping()
        logger.info(f"Successfully connected to Redis at {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")
        return client
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None

def clear_redis(redis_client):
    """Clear Redis database."""
    try:
        redis_client.flushdb()
        logger.info("Redis database cleared")
        return True
    except Exception as e:
        logger.error(f"Failed to clear Redis database: {e}")
        return False

def read_file_content(file_path):
    """Read the content of a file."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def load_document_to_redis(redis_client, file_path, redis_prefix):
    """Load a document into Redis."""
    try:
        # Read file content
        content = read_file_content(file_path)
        if content is None:
            return False
        
        # Create Redis keys
        relative_path = str(file_path)
        source_key = f"{redis_prefix}:source:{relative_path}"
        metadata_key = f"{source_key}:metadata"
        
        # Create metadata
        metadata = {
            "path": str(file_path),
            "size": len(content),
            "last_modified": os.path.getmtime(file_path),
            "created": os.path.getctime(file_path)
        }
        
        # Store in Redis
        redis_client.set(source_key, content)
        redis_client.set(metadata_key, json.dumps(metadata))
        
        return True
    except Exception as e:
        logger.error(f"Error loading document {file_path} to Redis: {e}")
        return False

def find_documents(source_dir, include_patterns=None, exclude_patterns=None):
    """Find all documents in the source directory that match the include/exclude patterns."""
    if include_patterns is None:
        include_patterns = ["**/*.pdf", "**/*.txt", "**/*.md", "**/*.py", "**/*.js", "**/*.java"]
    
    if exclude_patterns is None:
        exclude_patterns = ["**/README.md", "**/LICENSE.md", "**/.git/**", "**/node_modules/**"]
    
    documents = []
    for pattern in include_patterns:
        for file_path in Path(source_dir).glob(pattern):
            if file_path.is_file():
                # Check if file should be excluded
                exclude = False
                for exclude_pattern in exclude_patterns:
                    if file_path.match(exclude_pattern):
                        exclude = True
                        break
                
                if not exclude:
                    documents.append(file_path)
    
    return documents

def preload_documents(redis_client, source_dir, output_dir, threads=8, batch_size_mb=20000, use_gpu=False):
    """Preload documents into Redis in batches to manage memory usage.
    
    Args:
        redis_client: Redis client instance
        source_dir: Directory containing source documents
        output_dir: Directory to store the processed data
        threads: Number of threads for parallel loading
        batch_size_mb: Maximum batch size in MB (default: 20000 = 20GB)
        use_gpu: Whether to use GPU for embedding generation
    """
    # Set environment variables for CPU/GPU mode
    if use_gpu:
        os.environ["OLLAMA_USE_CPU"] = "false"
    else:
        os.environ["OLLAMA_USE_CPU"] = "true"
        
    # Set thread count environment variables for CPU processing
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    # Find all documents
    logger.info(f"Finding documents in {source_dir}...")
    all_documents = find_documents(source_dir)
    logger.info(f"Found {len(all_documents)} documents total")
    
    # Optimize batch size based on GPU/CPU mode
    if use_gpu:
        # For GPU, use larger batches to maximize throughput
        # GPU can process more documents at once efficiently
        optimal_batch_size_mb = min(batch_size_mb * 2, 50000)  # Cap at 50GB to avoid OOM
        logger.info(f"Using optimized GPU batch size: {optimal_batch_size_mb} MB")
        batch_size_bytes = optimal_batch_size_mb * 1024 * 1024
    else:
        # For CPU, use smaller batches to avoid memory pressure
        batch_size_bytes = batch_size_mb * 1024 * 1024
    
    # Sort documents by size (largest first) for more balanced batches
    all_documents.sort(key=lambda doc: os.path.getsize(doc), reverse=True)
    
    # Group documents into batches using a greedy algorithm for better balance
    batches = []
    batch_sizes = []
    
    # Determine optimal number of batches based on total document size
    total_size_bytes = sum(os.path.getsize(doc) for doc in all_documents)
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    # Calculate optimal number of batches
    if use_gpu:
        # For GPU, fewer batches with more documents is better
        num_batches = max(1, min(8, math.ceil(total_size_mb / (batch_size_bytes / (1024 * 1024)))))
    else:
        # For CPU, more batches with fewer documents per batch
        num_batches = max(1, math.ceil(total_size_mb / (batch_size_bytes / (1024 * 1024))))
    
    logger.info(f"Creating {num_batches} optimized batches for processing {len(all_documents)} documents")
    
    # Initialize batches
    for _ in range(num_batches):
        batches.append([])
        batch_sizes.append(0)
    
    # Distribute documents across batches using a greedy algorithm
    for doc in all_documents:
        try:
            doc_size = os.path.getsize(doc)
            
            # If this document alone exceeds batch size, process it individually
            if doc_size > batch_size_bytes:
                logger.warning(f"Document {doc} exceeds batch size limit ({doc_size / (1024*1024):.2f} MB). Processing individually.")
                batches.append([doc])
                batch_sizes.append(doc_size)
                continue
            
            # Find the batch with the smallest current size
            smallest_batch_idx = batch_sizes.index(min(batch_sizes))
            batches[smallest_batch_idx].append(doc)
            batch_sizes[smallest_batch_idx] += doc_size
        except Exception as e:
            logger.error(f"Error getting size for document {doc}: {e}")
        except Exception as e:
            logger.error(f"Error getting size for document {doc}: {e}")
    
    # Remove empty batches
    batches = [batch for batch in batches if batch]
    
    logger.info(f"Organized documents into {len(batches)} batches for processing")
    
    # Process each batch
    total_success_count = 0
    total_failure_count = 0
    total_documents = 0
    total_tokens = 0
    batch_start_time = time.time()
    
    for batch_num, batch_docs in enumerate(batches, 1):
        batch_size_mb = sum(os.path.getsize(doc) for doc in batch_docs) / (1024 * 1024)
        logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch_docs)} documents ({batch_size_mb:.2f} MB)")
        
        # Clear Redis before loading new batch (except for first batch)
        if batch_num > 1:
            logger.info("Clearing Redis before loading new batch...")
            redis_client.flushdb()
        
        # Preload batch documents to Redis
        start_time = time.time()
        success_count = 0
        failure_count = 0
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(load_document_to_redis, redis_client, doc, REDIS_CONFIG["prefix"]): doc for doc in batch_docs}
            
            for i, future in enumerate(as_completed(futures), 1):
                doc = futures[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                    
                    if i % 50 == 0 or i == len(batch_docs):
                        logger.info(f"Loaded {i}/{len(batch_docs)} documents into Redis for batch {batch_num}")
                except Exception as e:
                    logger.error(f"Error processing document {doc}: {e}")
                    failure_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Batch {batch_num} loading completed in {duration:.2f} seconds")
        logger.info(f"Successfully loaded {success_count} documents in batch {batch_num}")
        if failure_count > 0:
            logger.warning(f"Failed to load {failure_count} documents in batch {batch_num}")
        
        # Get Redis memory usage
        info = redis_client.info(section="memory")
        memory_used = info.get("used_memory_human", "unknown")
        logger.info(f"Redis memory usage after batch {batch_num}: {memory_used}")
        
        # Process this batch with the embedding process
        logger.info(f"Running embedding process for batch {batch_num}...")
        
        # Create a batch-specific output directory
        batch_output_dir = os.path.join(output_dir, f"batch_{batch_num}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # Set environment variables for this batch
        os.environ["PATHRAG_BATCH_NUM"] = str(batch_num)
        os.environ["PATHRAG_TOTAL_BATCHES"] = str(len(batches))
        
        # Run the embedding process for this batch
        try:
            batch_start_time = time.time()
            success = run_embedding_process(batch_output_dir, use_gpu, batch_mode=True, source_dir=source_dir, threads=threads)
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            
            # Store batch performance metrics in Redis
            doc_count = len(batch_docs)
            total_documents += doc_count
            redis_client.hset(f"pathrag:stats:batch:{batch_num}", mapping={
                "duration": batch_duration,
                "documents": doc_count,
                "docs_per_second": doc_count / batch_duration if batch_duration > 0 else 0,
                "batch_size_mb": batch_size_mb / 1024,  # Convert to GB
                "timestamp": time.time()
            })
            
            # Estimate token count (rough approximation)
            estimated_tokens = sum(os.path.getsize(doc) / 4 for doc in batch_docs)  # ~4 bytes per token
            total_tokens += estimated_tokens
            redis_client.hset(f"pathrag:stats:batch:{batch_num}", "estimated_tokens", estimated_tokens)
            
            if not success:
                logger.error(f"Failed to run embedding process for batch {batch_num}")
                # Continue with next batch even if this one fails
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt detected. Stopping batch processing.")
            return False
        
        total_success_count += success_count
        total_failure_count += failure_count
    
    batch_end_time = time.time()
    total_duration = batch_end_time - batch_start_time
    
    logger.info(f"All batches completed in {total_duration:.2f} seconds")
    logger.info(f"Successfully loaded {total_success_count} documents total")
    if total_failure_count > 0:
        logger.warning(f"Failed to load {total_failure_count} documents total")
    
    return total_success_count > 0

def run_embedding_process(output_dir, use_gpu=False, batch_mode=True, source_dir=None, threads=None):
    """Run the PathRAG embedding process.
    
    Args:
        output_dir: Directory to store the processed data
        use_gpu: Whether to use GPU for embedding generation
        batch_mode: Whether this is being called as part of batch processing
        source_dir: Source directory for documents (optional)
        threads: Number of threads to use for CPU processing (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure logs directory exists
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set environment variable for logs directory
    os.environ["PATHRAG_LOGS_DIR"] = logs_dir
    
    # Ensure Redis environment variables are set
    logger.info("Enabling Redis integration for embedding process")
    os.environ["PATHRAG_REDIS_ENABLED"] = "true"
    os.environ["PATHRAG_REDIS_HOST"] = "localhost"
    os.environ["PATHRAG_REDIS_PORT"] = "6379"
    os.environ["PATHRAG_REDIS_DB"] = "0"
    os.environ["PATHRAG_REDIS_PREFIX"] = "pathrag"
    os.environ["PATHRAG_REDIS_TTL"] = "604800"  # 1 week in seconds
    
    # Set additional performance optimization environment variables
    if use_gpu:
        # GPU optimizations
        os.environ["PATHRAG_BATCH_SIZE"] = "64"  # Larger batch size for GPU
        os.environ["PATHRAG_PARALLEL_CHUNKS"] = "true"
        os.environ["PATHRAG_CHUNK_PARALLEL_WORKERS"] = str(min(16, threads * 2) if threads else 16)
    else:
        # CPU optimizations
        os.environ["PATHRAG_BATCH_SIZE"] = "32"  # Smaller batch size for CPU
        os.environ["PATHRAG_PARALLEL_CHUNKS"] = "true"
        os.environ["PATHRAG_CHUNK_PARALLEL_WORKERS"] = str(threads if threads else 8)
    
    # Update config file with absolute paths
    config_file = os.path.join(RAG_BUILDER_DIR, "config", "config.yaml")
    
    # Properly parse the YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Update source_documents in the config if it was explicitly provided
    if source_dir is not None:
        config["source_documents"] = source_dir
    
    # Ensure we have a clean output directory path
    # Remove any nested batch_1 directories
    while "/batch_1/batch_1" in output_dir:
        logger.warning("Detected nested batch directories in output path, fixing...")
        output_dir = output_dir.replace("/batch_1/batch_1", "/batch_1")
        
    # Ensure the path doesn't end with batch_1
    if output_dir.endswith("/batch_1"):
        output_dir = output_dir[:-7]  # Remove the trailing /batch_1
        
    config["output_dir"] = output_dir
    logger.info(f"Setting output directory in config to: {output_dir}")
    
    # Ensure embedders section exists
    if "embedders" not in config:
        config["embedders"] = {}
    
    # Ensure ollama section exists
    if "ollama" not in config["embedders"]:
        config["embedders"]["ollama"] = {}
    
    # Explicitly set GPU/CPU mode for Ollama
    config["embedders"]["ollama"]["use_gpu"] = use_gpu
    
    # Add performance optimizations for Ollama
    if use_gpu:
        # GPU optimizations
        config["embedders"]["ollama"]["batch_size"] = 64
        config["embedders"]["ollama"]["max_concurrent_requests"] = min(16, threads * 2) if threads else 16
    else:
        # CPU optimizations
        config["embedders"]["ollama"]["batch_size"] = 32
        config["embedders"]["ollama"]["max_concurrent_requests"] = threads if threads else 8
    
    # Set embedder type to ollama
    if "embedder" not in config:
        config["embedder"] = {}
    config["embedder"]["type"] = "ollama"
    
    # Add processing optimizations
    if "processing" not in config:
        config["processing"] = {}
    
    # Configure parallel processing
    config["processing"]["parallel_embedding"] = True
    config["processing"]["max_workers"] = min(16, threads * 2) if use_gpu and threads else (threads if threads else 8)
    config["processing"]["batch_size"] = 64 if use_gpu else 32
    
    # Write the updated config
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Updated config file at {config_file}")
    logger.info(f"Using output directory: {config['output_dir']}")
    
    # Build the command
    cmd = [
        sys.executable,
        "-m", "src.main",
        "--config", config_file,
        "--pathrag"
    ]
    
    # Explicitly set GPU or CPU mode
    if use_gpu:
        cmd.append("--gpu")
        # Make sure Ollama knows to use GPU
        os.environ["OLLAMA_USE_CPU"] = "false"
    else:
        # We need to explicitly set CPU mode for Ollama
        os.environ["OLLAMA_USE_CPU"] = "true"
    
    # Pass threads parameter if provided
    if threads is not None:
        cmd.extend(["--threads", str(threads)])
    
    # Note: The RAG dataset builder doesn't support --parallel and --batch-size flags directly
    # We've already set these optimizations in the config file and environment variables
        
    # Note: We previously had a --redis-only flag here, but it's not recognized by src.main
    # When in batch mode, we're only processing documents already in Redis
    # We need to modify src.main to support this or handle it through environment variables
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        # Change to the rag-dataset-builder directory
        os.chdir(RAG_BUILDER_DIR)
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        if process.returncode != 0:
            logger.error(f"PathRAG embedding process failed with return code {process.returncode}")
            return False
        
        logger.info("PathRAG embedding process completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running PathRAG embedding process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def export_redis_to_disk(redis_client, output_dir):
    """Export Redis data to disk for permanent storage."""
    logger.info(f"Exporting Redis data to disk at {output_dir}...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the command
        cmd = [
            sys.executable,
            os.path.join(RAG_BUILDER_DIR, "scripts", "export_redis_to_disk.py"),
            "--output-dir", output_dir,
            "--redis-host", REDIS_CONFIG["host"],
            "--redis-port", str(REDIS_CONFIG["port"]),
            "--redis-db", str(REDIS_CONFIG["db"]),
            "--redis-prefix", REDIS_CONFIG["prefix"]
        ]
        
        if REDIS_CONFIG["password"]:
            cmd.extend(["--redis-password", REDIS_CONFIG["password"]])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        if process.returncode != 0:
            logger.error(f"Export process failed with return code {process.returncode}")
            return False
        
        logger.info(f"Redis data successfully exported to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error exporting Redis data to disk: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def list_available_databases():
    """List all available RAG databases."""
    try:
        databases = []
        for item in os.listdir(DATABASES_DIR):
            item_path = os.path.join(DATABASES_DIR, item)
            if os.path.isdir(item_path):
                # Check if this is a valid RAG database
                if os.path.exists(os.path.join(item_path, "pathrag")):
                    databases.append(item)
        
        return databases
    except Exception as e:
        logger.error(f"Error listing available databases: {e}")
        return []

def create_dataset(args):
    """Create a new RAG dataset with Redis integration."""
    # Set up proper logging
    log_file = setup_logging()
    logger.info(f"Starting dataset creation with Redis integration. Logging to {log_file}")
    
    # Set CPU/GPU environment variables early
    if args.gpu:
        logger.info("Setting up environment for GPU processing")
        os.environ["OLLAMA_USE_CPU"] = "false"
    else:
        logger.info("Setting up environment for CPU processing")
        os.environ["OLLAMA_USE_CPU"] = "true"
        
    # Set thread count environment variables
    if args.threads:
        logger.info(f"Setting thread count to {args.threads}")
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
        
    # We'll handle the clean flag after output_dir is defined
    
    # Read the config file to get the output directory if not specified in args
    config_file = os.path.join(RAG_BUILDER_DIR, "config", "config.yaml")
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            
        # Set source directory (command line takes precedence)
        source_dir = args.source_dir or config.get("source_documents", DEFAULT_SOURCE_DIR)
        
        # For output directory, use timestamp to avoid nesting issues
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if args.output_dir:
            # Command line argument takes precedence but ensure clean path
            base_dir = args.output_dir
            # Remove any batch_1 directories
            while "/batch_1" in base_dir:
                base_dir = base_dir.replace("/batch_1", "")
            # Use timestamp to ensure unique directory
            output_dir = f"{base_dir}_{timestamp}"
            logger.info(f"Using output directory from command line (with timestamp): {output_dir}")
        elif "output_dir" in config and config["output_dir"]:
            # Use config file value but ensure clean path
            base_dir = config["output_dir"]
            # Remove any batch_1 directories
            while "/batch_1" in base_dir:
                base_dir = base_dir.replace("/batch_1", "")
            # Use timestamp to ensure unique directory
            output_dir = f"{base_dir}_{timestamp}"
            logger.info(f"Using output directory from config (with timestamp): {output_dir}")
        else:
            # Fall back to default with timestamp
            output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"pathrag_redis_{timestamp}")
            logger.info(f"Using default output directory: {output_dir}")
            
        # Clean up any existing output directory if requested
        if args.clean and os.path.exists(output_dir):
            logger.info("Clean flag set, removing existing output directory")
            import shutil
            shutil.rmtree(output_dir)
            logger.info(f"Removed existing output directory: {output_dir}")
    except Exception as e:
        logger.warning(f"Error reading config file: {e}. Using default values.")
        source_dir = args.source_dir or DEFAULT_SOURCE_DIR
        output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, f"pathrag_redis_{time.strftime('%Y%m%d_%H%M%S')}")
    
    # Set environment variables
    os.environ["PATHRAG_SOURCE_DIR"] = source_dir
    os.environ["PATHRAG_OUTPUT_DIR"] = output_dir
    
    # Set Redis environment variables
    set_redis_env_vars()
    
    # Connect to Redis
    redis_client = connect_to_redis()
    if not redis_client:
        logger.error("Failed to connect to Redis. Aborting dataset creation.")
        return 1
    
    # Clear Redis database
    if not clear_redis(redis_client):
        logger.error("Failed to clear Redis database. Aborting dataset creation.")
        return 1
    
    # Log processing parameters
    logger.info(f"Using {args.threads} threads for parallel processing")
    logger.info(f"Maximum batch size: {args.max_ram} GB")
    logger.info(f"Using {'GPU' if args.gpu else 'CPU'} for embedding generation")
    
    # Preload documents to Redis in batches
    batch_size_mb = args.max_ram * 1024  # Convert GB to MB
    if not preload_documents(redis_client, source_dir, output_dir, args.threads, batch_size_mb, args.gpu):
        logger.error("Failed to preload documents to Redis. Aborting dataset creation.")
        return 1
    
    # The embedding process is now handled by the batched preloading function
    try:
        # We'll run a final merge step to combine all batch outputs
        logger.info("All batches processed. Merging results...")
        
        # Create a final merged output directory
        final_output_dir = os.path.join(output_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Find all batch directories
        batch_dirs = [d for d in os.listdir(output_dir) if d.startswith("batch_")]
        
        if not batch_dirs:
            logger.warning("No batch directories found. Dataset creation may be incomplete.")
            return 1
            
        # For now, we'll use the last completed batch as the final output
        # In the future, we could implement a more sophisticated merging strategy
        last_batch_dir = os.path.join(output_dir, sorted(batch_dirs)[-1])
        logger.info(f"Using results from {last_batch_dir} as the final output")
        
        # Copy the last batch's results to the final directory
        logger.info(f"Copying results from {last_batch_dir} to {final_output_dir}...")
        for item in os.listdir(last_batch_dir):
            src = os.path.join(last_batch_dir, item)
            dst = os.path.join(final_output_dir, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    import shutil
                    shutil.rmtree(dst)
                import shutil
                shutil.copytree(src, dst)
            else:
                import shutil
                shutil.copy2(src, dst)
        
        logger.info("Merge completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected during merge step. Dataset creation may be incomplete.")
        return 1
    
    # Export Redis data to disk to the final directory
    if not export_redis_to_disk(redis_client, final_output_dir):
        logger.error("Failed to export Redis data to disk. Dataset creation may be incomplete.")
        return 1
    
    # Gather statistics about the dataset creation process
    try:
        # Count total documents processed
        total_docs = sum(1 for _ in find_documents(source_dir))
        
        # Get Redis memory usage
        redis_info = redis_client.info(section="memory")
        memory_used = redis_info.get("used_memory_human", "unknown")
        
        # Get final database size
        import shutil
        final_db_size = shutil.disk_usage(final_output_dir).used / (1024 * 1024 * 1024)  # Convert to GB
        
        # Get performance metrics from Redis
        overall_stats = redis_client.hgetall("pathrag:stats:overall")
        
        # Print success message with detailed statistics
        logger.info("="*80)
        logger.info("Dataset creation completed successfully!")
        logger.info("-"*80)
        logger.info(f"Final output directory: {final_output_dir}")
        logger.info(f"Total documents processed: {total_docs}")
        logger.info(f"Processed in {len(batch_dirs)} batches with maximum RAM usage of {args.max_ram} GB per batch")
        
        # Performance metrics
        logger.info("-"*80)
        logger.info("Performance Metrics:")
        if overall_stats:
            total_duration = float(overall_stats.get(b"total_duration", 0))
            docs_per_second = float(overall_stats.get(b"docs_per_second", 0))
            tokens_per_second = float(overall_stats.get(b"tokens_per_second", 0))
            avg_seconds_per_doc = float(overall_stats.get(b"avg_seconds_per_doc", 0))
            
            logger.info(f"Total processing time: {total_duration:.2f} seconds")
            logger.info(f"Documents per second: {docs_per_second:.2f}")
            logger.info(f"Tokens per second (estimated): {tokens_per_second:.2f}")
            logger.info(f"Average time per document: {avg_seconds_per_doc:.4f} seconds")
        
        # Resource utilization
        logger.info("-"*80)
        logger.info("Resource Utilization:")
        logger.info(f"Peak Redis memory usage: {memory_used}")
        logger.info(f"Final database size: {final_db_size:.2f} GB")
        logger.info(f"Redis host: {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']} (DB: {REDIS_CONFIG['db']})")
        
        # Redis operations stats
        redis_cmd_stats = redis_client.info(section="commandstats")
        if redis_cmd_stats:
            logger.info("-"*80)
            logger.info("Redis Command Statistics:")
            for cmd, stats in sorted(redis_cmd_stats.items()):
                if cmd.startswith("cmdstat_"):
                    cmd_name = cmd[8:]  # Remove "cmdstat_" prefix
                    calls = stats.get("calls", 0)
                    usec = stats.get("usec", 0)
                    usec_per_call = stats.get("usec_per_call", 0)
                    if calls > 0:
                        logger.info(f"{cmd_name}: {calls} calls, {usec_per_call:.2f} μs/call")
        
        logger.info("="*80)
    except Exception as e:
        logger.error(f"Error gathering statistics: {e}")
        
    return 0

def infer(args):
    """Run inference using a specified RAG database (future implementation)."""
    # Set up proper logging
    log_file = setup_logging()
    logger.info(f"Inference mode selected with database: {args.database}. Logging to {log_file}")
    logger.info("Inference mode is not yet implemented.")
    
    # List available databases
    databases = list_available_databases()
    if databases:
        logger.info("Available databases:")
        for db in databases:
            logger.info(f"  - {db}")
    else:
        logger.info("No available databases found.")
    
    return 0

def retrieve(args):
    """Run retrieval using a specified RAG database and query (future implementation)."""
    # Set up proper logging
    log_file = setup_logging()
    logger.info(f"Retrieval mode selected with database: {args.database}. Logging to {log_file}")
    logger.info(f"Query: {args.query}")
    logger.info("Retrieval mode is not yet implemented.")
    
    # List available databases
    databases = list_available_databases()
    if databases:
        logger.info("Available databases:")
        for db in databases:
            logger.info(f"  - {db}")
    else:
        logger.info("No available databases found.")
    
    return 0

def main():
    """Main entry point for the HADES unified script."""
    # Set project paths before parsing arguments
    global PROJECT_ROOT, RAG_BUILDER_DIR, DEFAULT_SOURCE_DIR, DEFAULT_OUTPUT_DIR, DATABASES_DIR
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    RAG_BUILDER_DIR = os.path.join(PROJECT_ROOT, "rag-dataset-builder")
    DEFAULT_SOURCE_DIR = os.path.join(PROJECT_ROOT, "source_documents")
    DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "rag_databases")
    DATABASES_DIR = os.path.join(PROJECT_ROOT, "rag_databases")
    
    parser = argparse.ArgumentParser(description="HADES Unified Startup Script")
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Create dataset mode
    create_parser = subparsers.add_parser("create", help="Create a new RAG dataset")
    create_parser.add_argument("--source-dir", help="Directory containing source documents")
    create_parser.add_argument("--output-dir", help="Directory to store the processed data")
    create_parser.add_argument("--gpu", action="store_true", help="Use GPU for embedding generation")
    create_parser.add_argument("--cpu", action="store_true", help="Force CPU for embedding generation (default if --gpu not specified)")
    create_parser.add_argument("--threads", type=int, default=16, help="Number of threads for parallel loading")
    create_parser.add_argument("--max-ram", type=int, default=64, help="Maximum RAM to use in GB (default: 64)")
    create_parser.add_argument("--clean", action="store_true", help="Clean existing output directory before processing")
    
    # Inference mode (future)
    infer_parser = subparsers.add_parser("infer", help="Run inference using a specified RAG database")
    infer_parser.add_argument("--database", required=True, help="Path to the RAG database")
    infer_parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    
    # Retrieval mode (future)
    retrieve_parser = subparsers.add_parser("retrieve", help="Run retrieval using a specified RAG database and query")
    retrieve_parser.add_argument("--database", required=True, help="Path to the RAG database")
    retrieve_parser.add_argument("--query", required=True, help="Query for retrieval")
    retrieve_parser.add_argument("--gpu", action="store_true", help="Use GPU for retrieval")
    
    # Redis configuration options
    for subparser in [create_parser, infer_parser, retrieve_parser]:
        subparser.add_argument("--redis-host", default="localhost", help="Redis host")
        subparser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
        subparser.add_argument("--redis-db", type=int, default=0, help="Redis database")
        subparser.add_argument("--redis-password", default="", help="Redis password")
        subparser.add_argument("--redis-prefix", default="pathrag", help="Redis key prefix")
    
    args = parser.parse_args()
    
    # Update Redis configuration from command-line arguments
    if hasattr(args, "redis_host"):
        REDIS_CONFIG["host"] = args.redis_host
    if hasattr(args, "redis_port"):
        REDIS_CONFIG["port"] = args.redis_port
    if hasattr(args, "redis_db"):
        REDIS_CONFIG["db"] = args.redis_db
    if hasattr(args, "redis_password"):
        REDIS_CONFIG["password"] = args.redis_password
    if hasattr(args, "redis_prefix"):
        REDIS_CONFIG["prefix"] = args.redis_prefix
    
    # Run the selected mode
    if args.mode == "create":
        return create_dataset(args)
    elif args.mode == "infer":
        return infer(args)
    elif args.mode == "retrieve":
        return retrieve(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
