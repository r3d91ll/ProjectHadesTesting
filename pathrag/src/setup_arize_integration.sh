#!/bin/bash
# Setup script for Arize Phoenix integration with PathRAG

echo "PathRAG - Arize Phoenix Integration Setup"
echo "========================================="
echo

# Check if Arize Phoenix is running
echo "Checking if Arize Phoenix is running..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8084/health; then
    echo "✅ Arize Phoenix is running and accessible at http://localhost:8084"
else
    echo "❌ Arize Phoenix is not accessible at http://localhost:8084"
    echo
    echo "Please start Arize Phoenix using your existing Docker configuration:"
    echo "1. Run 'docker ps -a | grep arize-phoenix' to check for existing containers"
    echo "2. If it exists but is stopped, run 'docker start <container_id>'"
    echo "3. If it doesn't exist, run your docker-compose or docker run command to start it"
    echo
    echo "Based on your configuration, Arize Phoenix should be available at http://localhost:8084"
    echo "Once Arize Phoenix is running, re-run this script to continue."
    exit 1
fi

# Activate the virtual environment if it exists
if [ -d "../venv" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check for required Python packages
echo "Checking for required packages..."
python3 -c "import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing scikit-learn for embedding visualization..."
    pip install scikit-learn
fi

# Load database into Arize Phoenix
echo
echo "Loading PathRAG database into Arize Phoenix..."
python3 load_database_to_arize.py --database-dir ../database

# Instructions for accessing the dashboard
echo
echo "🚀 Database loaded into Arize Phoenix!"
echo
echo "Access the Arize Phoenix dashboard at: http://localhost:8084"
echo
echo "In the dashboard, you can view:"
echo "1. Knowledge Graph structure and statistics"
echo "2. Embedding clusters visualization"
echo "3. Document metadata and categories analysis"
echo
echo "For detailed database statistics:"
echo "- Total chunks: $(find ../database/chunks -type f | wc -l)"
echo "- Unique documents: $(find ../database/metadata -type f | wc -l)"
echo "- Graph nodes: $(grep -o '"total_entities":[^,]*' ../database/checkpoint.json | cut -d':' -f2)"
echo "- Graph relationships: $(grep -o '"total_relationships":[^,]*' ../database/checkpoint.json | cut -d':' -f2)"
