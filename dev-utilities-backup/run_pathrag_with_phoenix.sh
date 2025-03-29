#!/bin/bash

# Run the RAG dataset builder with Phoenix integration

# Set environment variables
export PHOENIX_PROJECT_NAME="pathrag-dataset-builder"
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:8084"

# Print confirmation
echo "Running RAG dataset builder with Phoenix project: ${PHOENIX_PROJECT_NAME}"

# Navigate to dataset builder directory and run it
cd "$(dirname "$0")/../rag-dataset-builder" || exit 1
python src/main.py --config config/config.yaml "$@"
