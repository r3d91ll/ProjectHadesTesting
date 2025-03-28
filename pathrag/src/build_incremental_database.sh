#!/bin/bash
# Incremental PathRAG Database Builder
# This script builds a PathRAG database incrementally, processing data in stages
# to handle the large dataset efficiently

# Load virtual environment
source .venv/bin/activate

# Set base directories
DATA_DIR="pathrag/data/test_datasets"
OUTPUT_DIR="pathrag/data/pathrag_complete_database"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting incremental PathRAG database build..."

# Process papers by category to reduce memory pressure
echo "Processing research papers by category..."

# Array of paper categories
categories=(
    "actor_network_theory"
    "sts_digital_sociology" 
    "knowledge_graphs_retrieval"
    "computational_linguistics"
    "ethics_bias_ai"
    "graph_reasoning_ml"
    "semiotics_linguistic_anthropology"
)

# Process each category sequentially
for category in "${categories[@]}"; do
    echo "Processing category: $category"
    python pathrag/src/pathrag_db_builder.py \
        --data-dir "$DATA_DIR/papers/$category" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 5
    
    # Force garbage collection between categories
    sleep 2
done

# Process extended documentation
echo "Processing documentation..."
python pathrag/src/pathrag_db_builder.py \
    --data-dir "$DATA_DIR/extended/documentation" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 2

# Process text data
echo "Processing text files..."
python pathrag/src/pathrag_db_builder.py \
    --data-dir "$DATA_DIR/text" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 5

# Process datasets if available
if [ -d "$DATA_DIR/datasets" ] && [ "$(ls -A $DATA_DIR/datasets)" ]; then
    echo "Processing datasets..."
    python pathrag/src/pathrag_db_builder.py \
        --data-dir "$DATA_DIR/datasets" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 2
fi

echo "Database build complete!"
echo "PathRAG database is available at: $OUTPUT_DIR"

# Print database statistics
echo "Database Statistics:"
find "$OUTPUT_DIR" -type f | wc -l
du -h "$OUTPUT_DIR"
