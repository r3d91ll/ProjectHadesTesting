#!/bin/bash

# Deep cleanup script for PathRAG data directories
# This script aggressively removes all data directories and files except the structure with .gitkeep

echo "Starting deep cleanup of PathRAG data directories..."

# Helper function to clean a directory while preserving structure
clean_directory() {
  DIR=$1
  echo "Cleaning directory: $DIR"
  
  # Remove all subdirectories and files while preserving main structure
  find "$DIR" -mindepth 1 -not -name ".gitkeep" -exec rm -rf {} \; 2>/dev/null
  
  # Create .gitkeep file to preserve the directory
  touch "$DIR/.gitkeep"
}

# Remove any existing .gitkeep directories that were created incorrectly
find /home/todd/ML-Lab/New-HADES/pathrag -name ".gitkeep" -type d -exec rm -rf {} \; 2>/dev/null

# Clean core PathRAG directories
clean_directory "/home/todd/ML-Lab/New-HADES/pathrag/data"
clean_directory "/home/todd/ML-Lab/New-HADES/pathrag/database"
clean_directory "/home/todd/ML-Lab/New-HADES/pathrag/logs"

# Create and clean key subdirectories for PathRAG
mkdir -p /home/todd/ML-Lab/New-HADES/pathrag/data/chroma_db
mkdir -p /home/todd/ML-Lab/New-HADES/pathrag/data/datasets
mkdir -p /home/todd/ML-Lab/New-HADES/pathrag/data/document_store
mkdir -p /home/todd/ML-Lab/New-HADES/pathrag/data/minimal_pathrag

clean_directory "/home/todd/ML-Lab/New-HADES/pathrag/data/chroma_db"
clean_directory "/home/todd/ML-Lab/New-HADES/pathrag/data/datasets"
clean_directory "/home/todd/ML-Lab/New-HADES/pathrag/data/document_store"
clean_directory "/home/todd/ML-Lab/New-HADES/pathrag/data/minimal_pathrag"

# Clean RAG Dataset Builder directories
mkdir -p /home/todd/ML-Lab/New-HADES/rag-dataset-builder/data/input
mkdir -p /home/todd/ML-Lab/New-HADES/rag-dataset-builder/data/output
mkdir -p /home/todd/ML-Lab/New-HADES/rag-dataset-builder/data/chunks
mkdir -p /home/todd/ML-Lab/New-HADES/rag-dataset-builder/data/embeddings
mkdir -p /home/todd/ML-Lab/New-HADES/rag-dataset-builder/logs

clean_directory "/home/todd/ML-Lab/New-HADES/rag-dataset-builder/data"
clean_directory "/home/todd/ML-Lab/New-HADES/rag-dataset-builder/data/input"
clean_directory "/home/todd/ML-Lab/New-HADES/rag-dataset-builder/data/output"
clean_directory "/home/todd/ML-Lab/New-HADES/rag-dataset-builder/data/chunks"
clean_directory "/home/todd/ML-Lab/New-HADES/rag-dataset-builder/data/embeddings"
clean_directory "/home/todd/ML-Lab/New-HADES/rag-dataset-builder/logs"

echo "Creating Arize Phoenix directories for performance tracking..."
# Create directory for Arize Phoenix tracking
mkdir -p /home/todd/ML-Lab/New-HADES/arize_phoenix_data
touch /home/todd/ML-Lab/New-HADES/arize_phoenix_data/.gitkeep

# Create GPU monitoring directories for Prometheus metrics
echo "Creating GPU monitoring directories..."
mkdir -p /home/todd/ML-Lab/New-HADES/monitoring/prometheus
touch /home/todd/ML-Lab/New-HADES/monitoring/prometheus/.gitkeep
mkdir -p /home/todd/ML-Lab/New-HADES/monitoring/textfile_collector
touch /home/todd/ML-Lab/New-HADES/monitoring/textfile_collector/.gitkeep

echo "Deep cleanup complete!"
echo "Directory structure is preserved with .gitkeep files while all nested directories have been removed."
