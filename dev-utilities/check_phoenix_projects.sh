#!/bin/bash
# Check Phoenix projects and traces
# This script queries the Phoenix API to list all projects and traces

set -e  # Exit on error

PHOENIX_HOST=${PHOENIX_HOST:-localhost}
PHOENIX_PORT=${PHOENIX_PORT:-8084}
PHOENIX_URL="http://${PHOENIX_HOST}:${PHOENIX_PORT}"

# Check if Phoenix is running
echo "🔍 Checking Phoenix connection..."
if ! curl -s "${PHOENIX_URL}/health" > /dev/null; then
    echo "❌ Phoenix is not running at ${PHOENIX_URL}"
    exit 1
fi

echo "✅ Phoenix is running at ${PHOENIX_URL}"

# Get list of projects
echo -e "\n📊 Listing Phoenix projects..."
PROJECTS=$(curl -s "${PHOENIX_URL}/api/projects" | jq -r '.projects[] | .name' 2>/dev/null)

if [[ $? -ne 0 || -z "$PROJECTS" ]]; then
    echo "❌ Failed to retrieve projects from Phoenix"
    exit 1
fi

echo "Found projects in Phoenix:"
echo "$PROJECTS" | while read project; do
    echo "  - $project"
done

# Check for pathrag-inference project
if echo "$PROJECTS" | grep -q "pathrag-inference"; then
    echo -e "\n✅ PathRAG project 'pathrag-inference' found in Phoenix"
    
    # Get traces for pathrag-inference project
    echo -e "\n🔍 Getting traces for 'pathrag-inference' project..."
    TRACES=$(curl -s "${PHOENIX_URL}/api/projects/pathrag-inference/traces?limit=5" | jq -r '.traces[] | .id' 2>/dev/null)
    
    if [[ $? -ne 0 || -z "$TRACES" ]]; then
        echo "❌ Failed to retrieve traces for 'pathrag-inference' project"
    else
        echo "Found traces in 'pathrag-inference' project:"
        echo "$TRACES" | while read trace; do
            echo "  - $trace"
        done
    fi
else
    echo -e "\n❌ PathRAG project 'pathrag-inference' not found in Phoenix"
fi

echo -e "\n✅ Check complete"
