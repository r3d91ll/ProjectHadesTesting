#!/bin/bash
# Script to set up the development environment for HADES XnX Notation Experimental Validation

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/datasets/kilt
mkdir -p data/results
mkdir -p tests/pathrag
mkdir -p tests/graphrag
mkdir -p tests/metrics

# Set up Python virtual environment if not already exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv .venv
    
    echo "Activating virtual environment..."
    source .venv/bin/activate
    
    echo "Installing dependencies..."
    pip install -e .
else
    echo "Virtual environment already exists, activating..."
    source .venv/bin/activate
    
    echo "Updating dependencies..."
    pip install -e .
fi

# Download KILT datasets if they don't exist
if [ ! -f "data/datasets/kilt/nq-test.jsonl" ]; then
    echo "Downloading KILT Natural Questions dataset..."
    wget -O data/datasets/kilt/nq-test.jsonl https://dl.fbaipublicfiles.com/KILT/nq-test.jsonl
fi

if [ ! -f "data/datasets/kilt/hotpotqa-test.jsonl" ]; then
    echo "Downloading KILT HotpotQA dataset..."
    wget -O data/datasets/kilt/hotpotqa-test.jsonl https://dl.fbaipublicfiles.com/KILT/hotpotqa-test.jsonl
fi

# Create symbolic links to original implementations
echo "Creating symbolic links to original implementations..."
if [ -d "temp/pathrag" ]; then
    ln -sf "$(pwd)/temp/pathrag" "$(pwd)/implementations/pathrag/original/src"
    echo "Linked PathRAG source code"
else
    echo "WARNING: PathRAG source code not found in temp/pathrag"
fi

if [ -d "temp/neo4j-graphrag-python" ]; then
    ln -sf "$(pwd)/temp/neo4j-graphrag-python" "$(pwd)/implementations/graphrag/original/src"
    echo "Linked Neo4j GraphRAG source code"
else
    echo "WARNING: Neo4j GraphRAG source code not found in temp/neo4j-graphrag-python"
fi

echo "Setup complete!"
