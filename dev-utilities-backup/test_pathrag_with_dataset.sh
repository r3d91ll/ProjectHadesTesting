#!/bin/bash
# Test PathRAG with the newly built dataset
# This script configures PathRAG to use our dataset and runs a test query

set -e  # Exit on error

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PATHRAG_DIR="$PROJECT_ROOT/pathrag"
DATASET_DIR="$PROJECT_ROOT/rag_databases/current"

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Error: Dataset directory not found at $DATASET_DIR"
    echo "Please run the RAG dataset builder first"
    exit 1
fi

# Check if PathRAG exists
if [ ! -d "$PATHRAG_DIR" ]; then
    echo "❌ Error: PathRAG directory not found at $PATHRAG_DIR"
    exit 1
fi

# Copy our configuration to PathRAG
echo "📝 Configuring PathRAG to use the newly built dataset..."
cp "$SCRIPT_DIR/pathrag_env_config" "$PATHRAG_DIR/.env"

# Activate the virtual environment
echo "🔄 Activating virtual environment..."
cd "$PATHRAG_DIR"
source venv/bin/activate

# Apply the direct fix to ensure PathRAG uses the correct project name
echo "🔧 Configuring PathRAG to use a separate project in Phoenix..."
python "$SCRIPT_DIR/fix_pathrag_config.py"

# Run a test query
echo "🔍 Running test query with PathRAG..."
echo "Query: 'What is the transformer architecture?'"
python src/pathrag_runner.py --query "What is the transformer architecture?" --session-id "test_session"

# Check if Phoenix is running and show dashboard link
if curl -s http://localhost:8084/health > /dev/null; then
    echo "✅ Arize Phoenix is running"
    echo "📊 View telemetry at http://localhost:8084"
    
    # Check Phoenix projects
    echo -e "\n🔍 Checking Phoenix projects..."
    PROJECTS=$(curl -s http://localhost:8084/api/projects | jq -r '.projects[] | .name' 2>/dev/null)
    if [[ $? -eq 0 && ! -z "$PROJECTS" ]]; then
        echo "Found projects in Phoenix:"
        echo "$PROJECTS" | while read project; do
            echo "  - $project"
        done
        if echo "$PROJECTS" | grep -q "pathrag-inference"; then
            echo "✅ PathRAG project 'pathrag-inference' found in Phoenix"
        else
            echo "❌ PathRAG project 'pathrag-inference' not found in Phoenix"
        fi
    else
        echo "⚠️ Could not retrieve Phoenix projects (is jq installed?)"
    fi
else
    echo "⚠️ Arize Phoenix is not running"
    echo "To start Phoenix, run: docker-compose up -d arize-phoenix"
fi

echo -e "\n✅ Test complete"
