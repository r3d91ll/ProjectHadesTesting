#!/bin/bash
# Start the GPU metrics exporter and make sure the directory exists

# Create metrics directory if it doesn't exist
mkdir -p /tmp/node_exporter_metrics
chmod 755 /tmp/node_exporter_metrics

# Run the GPU metrics exporter script in background
/home/todd/ML-Lab/New-HADES/ladon/scripts/gpu_metrics_exporter.sh &

# Set proper ownership
echo "GPU metrics exporter started."
