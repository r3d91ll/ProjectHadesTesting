#!/bin/bash
# Environment Setup for PathRAG Dataset Builder
# Sets up the necessary environment for running the dataset builder

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Setting up environment for PathRAG Dataset Builder..."

# Check for required dependencies
echo "Checking dependencies..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    exit 1
fi

# Check for lsyncd (required for RAM disk operations)
if ! command -v lsyncd &> /dev/null; then
    echo "WARNING: lsyncd is not installed. RAM disk operations will not work."
    echo "To install lsyncd: sudo apt-get install lsyncd"
fi

# Check for PyTorch
if ! python3 -c "import torch" &> /dev/null; then
    echo "WARNING: PyTorch is not installed. Please install it for embedding operations."
    echo "To install PyTorch: pip install torch"
fi

# Check for sentence-transformers
if ! python3 -c "import sentence_transformers" &> /dev/null; then
    echo "WARNING: sentence-transformers is not installed. Please install it for embedding operations."
    echo "To install sentence-transformers: pip install sentence-transformers"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p "${PROJECT_ROOT}/../source_documents"
mkdir -p "${PROJECT_ROOT}/../rag_databases"
mkdir -p "${PROJECT_ROOT}/../logs"

# Set up Python environment variables
echo "Setting up Python environment variables..."
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Set up PyTorch environment variables for optimal performance
NUM_THREADS=$(nproc)
export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export NUMEXPR_NUM_THREADS=${NUM_THREADS}
export NUMEXPR_MAX_THREADS=${NUM_THREADS}

echo "Environment setup complete."
echo "You can now run the dataset builder using one of the following commands:"
echo "  ./scripts/run.sh [--cpu|--gpu] [--pathrag|--graphrag|--literag]"
echo "  ./scripts/run_with_ramdisk.sh [--cpu|--gpu] [--pathrag|--graphrag|--literag]"
echo "  ./scripts/run_benchmark.sh [--pathrag|--graphrag|--literag]"
