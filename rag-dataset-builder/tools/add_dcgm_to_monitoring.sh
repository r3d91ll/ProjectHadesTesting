#!/bin/bash
# Script to add NVIDIA DCGM exporter to existing monitoring setup
# This integrates with your existing Prometheus and Grafana

set -e

echo "Adding NVIDIA DCGM exporter to existing monitoring setup..."

# Find the existing docker-compose.yml file
COMPOSE_FILES=$(find /home/todd/ML-Lab/New-HADES -name "docker-compose.yml" -type f | grep -v "monitoring/docker-compose.yml")

if [ -z "$COMPOSE_FILES" ]; then
    echo "ERROR: Could not find existing docker-compose.yml file."
    echo "Please specify the path to your existing monitoring setup."
    exit 1
fi

echo "Found existing docker-compose files:"
echo "$COMPOSE_FILES"

# Ask user to select the correct file
echo "Please enter the number of the docker-compose file to modify:"
select COMPOSE_FILE in $COMPOSE_FILES; do
    if [ -n "$COMPOSE_FILE" ]; then
        break
    fi
done

echo "Using docker-compose file: $COMPOSE_FILE"
COMPOSE_DIR=$(dirname "$COMPOSE_FILE")

# Create a temporary file for the DCGM service definition
TEMP_FILE=$(mktemp)
cat > "$TEMP_FILE" << EOF

  # NVIDIA DCGM exporter for GPU metrics
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "9400:9400"
    command: -f /etc/dcgm-exporter/dcp-metrics-included.csv

EOF

# Add the DCGM service to the docker-compose file
echo "Adding DCGM exporter service to docker-compose.yml..."
sed -i '/^services:/r '"$TEMP_FILE"'' "$COMPOSE_FILE"

# Clean up
rm "$TEMP_FILE"

# Update Prometheus configuration to scrape DCGM exporter
PROMETHEUS_CONFIG=$(find "$COMPOSE_DIR" -name "prometheus.yml" -type f)

if [ -z "$PROMETHEUS_CONFIG" ]; then
    echo "WARNING: Could not find Prometheus configuration file."
    echo "Please manually add the following scrape config to your prometheus.yml:"
    echo ""
    echo "  - job_name: 'dcgm'"
    echo "    static_configs:"
    echo "      - targets: ['dcgm-exporter:9400']"
    echo ""
else
    echo "Found Prometheus configuration: $PROMETHEUS_CONFIG"
    
    # Check if DCGM scrape config already exists
    if grep -q "job_name: 'dcgm'" "$PROMETHEUS_CONFIG"; then
        echo "DCGM scrape config already exists in Prometheus configuration."
    else
        # Add DCGM scrape config to Prometheus configuration
        TEMP_FILE=$(mktemp)
        cat > "$TEMP_FILE" << EOF

  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']
EOF
        
        echo "Adding DCGM scrape config to Prometheus configuration..."
        sed -i '/scrape_configs:/r '"$TEMP_FILE"'' "$PROMETHEUS_CONFIG"
        
        # Clean up
        rm "$TEMP_FILE"
    fi
fi

echo "Configuration complete!"
echo ""
echo "To apply changes, run the following commands:"
echo "cd $COMPOSE_DIR"
echo "docker-compose up -d"
echo ""
echo "After restarting, import the NVIDIA DCGM dashboard in Grafana:"
echo "1. Go to your Grafana dashboard"
echo "2. Click on '+' > 'Import'"
echo "3. Enter dashboard ID: 12239 (NVIDIA DCGM Exporter Dashboard)"
echo "4. Select your Prometheus data source and click 'Import'"
