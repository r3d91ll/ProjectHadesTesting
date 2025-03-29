#!/bin/bash

# Phoenix environment variables for RAG dataset builder
export PHOENIX_PROJECT_NAME="pathrag-dataset-builder"
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:8084"

# Print the environment variables for confirmation
echo "Phoenix environment variables set:"
echo "  PHOENIX_PROJECT_NAME=${PHOENIX_PROJECT_NAME}"
echo "  PHOENIX_COLLECTOR_ENDPOINT=${PHOENIX_COLLECTOR_ENDPOINT}"
