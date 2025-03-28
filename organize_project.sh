#!/bin/bash

# Project organization script for PathRAG
# This script moves redundant files to 'archive' directories
# but doesn't delete them in case they're needed later

echo "Organizing PathRAG project structure..."

# Create archive directories if they don't exist
mkdir -p /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive
mkdir -p /home/todd/ML-Lab/New-HADES/rag-dataset-builder/config/archive

# Move redundant example scripts to archive
echo "Archiving redundant example scripts..."
mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/config_demo.py \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/

mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/direct_pathrag_test.py \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/

mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/pathrag_example.py \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/

mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/phoenix_pathrag_test.py \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/

mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/complete_pipeline.py \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/

# Move example config files to archive
echo "Archiving example config files..."
mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/env_config_example.yaml \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/

mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/user_config_example.yaml \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/

mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/pathrag_custom.yaml \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/

# Move redundant config files to archive
echo "Archiving redundant config files..."
mv /home/todd/ML-Lab/New-HADES/rag-dataset-builder/config/focused_config.yaml \
   /home/todd/ML-Lab/New-HADES/rag-dataset-builder/config/archive/

# Remove any temporary, cache, or log files
echo "Cleaning temporary files..."
find /home/todd/ML-Lab/New-HADES -name "*.pyc" -type f -delete
find /home/todd/ML-Lab/New-HADES -name "__pycache__" -type d -exec rm -rf {} +
find /home/todd/ML-Lab/New-HADES -name "*.log" -type f -delete
find /home/todd/ML-Lab/New-HADES -name ".DS_Store" -type f -delete

echo "Creating a README in archive directories..."
cat > /home/todd/ML-Lab/New-HADES/rag-dataset-builder/examples/archive/README.md << 'EOF'
# Archived Example Scripts

This directory contains example scripts and configurations that are no longer in active use but are preserved for reference.

These files may contain useful patterns or implementations that could be helpful for future development.
EOF

cat > /home/todd/ML-Lab/New-HADES/rag-dataset-builder/config/archive/README.md << 'EOF'
# Archived Configuration Files

This directory contains older configuration files that are no longer in active use but are preserved for reference.

The main active configuration file is in the parent directory (`../config.yaml`).
EOF

echo "Project organization complete!"
echo "Core files remain in their original locations."
echo "Redundant files have been moved to archive directories."
