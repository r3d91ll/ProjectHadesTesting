#!/bin/bash

# Cleanup script for PathRAG and RAG Dataset Builder
# This script preserves directory structure but removes data

# Create required directories with .gitkeep files
echo "Creating directory structure with .gitkeep files..."

# PathRAG directories
PATHRAG_DIRS=(
  "pathrag/data/chroma_db"
  "pathrag/data/datasets"
  "pathrag/data/document_store"
  "pathrag/data/minimal_pathrag"
  "pathrag/logs"
)

# RAG Dataset Builder directories
RAG_DIRS=(
  "rag-dataset-builder/data/input"
  "rag-dataset-builder/data/output"
  "rag-dataset-builder/data/chunks"
  "rag-dataset-builder/data/embeddings"
  "rag-dataset-builder/data/registry"
  "rag-dataset-builder/logs"
  "rag-dataset-builder/.cache"
)

# Source documents directories
SOURCE_DIRS=(
  "source_documents/academic_papers"
  "source_documents/papers"
)

# Database directories
DB_DIRS=(
  "rag_databases/pathrag/chunks"
  "rag_databases/pathrag/embeddings"
  "rag_databases/pathrag/graph"
  "rag_databases/pathrag/metadata"
)

# Clean existing data but preserve structure
echo "Cleaning existing data..."

# Clean source documents
find source_documents/academic_papers -type f -name "*.pdf" -delete
find source_documents/papers -type f -name "*.pdf" -delete

# Clean database directories
rm -rf rag_databases/pathrag/chunks/*
rm -rf rag_databases/pathrag/embeddings/*
rm -rf rag_databases/pathrag/graph/*
rm -rf rag_databases/pathrag/metadata/*

# Create all directories and add .gitkeep files
for dir in "${PATHRAG_DIRS[@]}" "${RAG_DIRS[@]}" "${SOURCE_DIRS[@]}" "${DB_DIRS[@]}"; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
  echo "Created $dir/.gitkeep"
done

echo "Cleanup complete. Directory structure preserved with empty directories."

# Clean up large data files (safely)
echo "Removing large data files..."

# PathRAG data files (excluding .gitkeep)
find pathrag/data -type f -not -name '.gitkeep' -exec rm -f {} \;
find pathrag/database -type f -not -name '.gitkeep' -exec rm -f {} \;
find pathrag/logs -type f -not -name '.gitkeep' -exec rm -f {} \;

# RAG Dataset Builder specific files
echo "Cleaning RAG Dataset Builder files..."
rm -f rag-dataset-builder/data/registry/academic_papers.json
rm -rf rag-dataset-builder/data/input/pubmed/*
rm -rf rag-dataset-builder/data/input/semantic_scholar/*
rm -rf rag-dataset-builder/data/input/socarxiv/*
rm -rf rag-dataset-builder/data/input/arxiv/*
rm -f rag-dataset-builder/data/input/*.txt

# Clean processed database
echo "Cleaning processed PathRAG database..."
rm -rf rag_databases/current/*

# Arize Phoenix data
rm -rf arize_phoenix_data/
rm -f phoenix_client.log
rm -f *.trace.json

echo "Cleanup complete!"
