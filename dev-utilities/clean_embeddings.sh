#!/bin/bash
# Clean Embeddings Script
# This script removes all previous embedding outputs to prepare for a fresh comparison test

set -e

# Configuration
DB_DIR="/home/todd/ML-Lab/New-HADES/rag_databases"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/home/todd/ML-Lab/New-HADES/rag_databases_backup_${TIMESTAMP}"

# Create backup directory
echo "Creating backup of existing embeddings at ${BACKUP_DIR}..."
mkdir -p ${BACKUP_DIR}

# Backup existing embeddings
if [ -d "${DB_DIR}" ] && [ "$(ls -A ${DB_DIR})" ]; then
    echo "Backing up existing embeddings..."
    cp -r ${DB_DIR}/* ${BACKUP_DIR}/
    echo "Backup completed at ${BACKUP_DIR}"
    
    # Count existing embeddings for reference
    echo "Previous embedding statistics:"
    if [ -d "${DB_DIR}/current" ]; then
        echo "GPU embeddings (current):"
        echo "Documents: $(find ${DB_DIR}/current -name "*.json" | wc -l)"
        echo "Chunks: $(grep -o '"text":' ${DB_DIR}/current/*.json 2>/dev/null | wc -l)"
    fi
    
    if [ -d "${DB_DIR}/pathRAG_CPU" ]; then
        echo "CPU embeddings (pathRAG_CPU):"
        echo "Documents: $(find ${DB_DIR}/pathRAG_CPU -name "*.json" | wc -l)"
        echo "Chunks: $(grep -o '"text":' ${DB_DIR}/pathRAG_CPU/*.json 2>/dev/null | wc -l)"
    fi
    
    # Clean out all embeddings
    echo "Removing all existing embeddings..."
    rm -rf ${DB_DIR}/*
    echo "All embeddings have been removed."
else
    echo "No existing embeddings found in ${DB_DIR}."
fi

# Create empty directories for new runs
mkdir -p ${DB_DIR}/current
mkdir -p ${DB_DIR}/pathRAG_CPU

echo "Embedding directories have been cleaned and are ready for fresh comparison tests."
echo "Source documents have been preserved."
echo "You can now run the GPU and CPU embedding scripts for a fair comparison."
