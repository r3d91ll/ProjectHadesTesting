#!/bin/bash
# GPU Monitoring Setup using NVIDIA DCGM
# This script sets up persistent GPU monitoring with Grafana and DCGM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Setting up GPU monitoring with NVIDIA DCGM..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Please install Docker first: https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA drivers are required but not installed."
    exit 1
fi

# Create persistent storage directories
MONITORING_DIR="${PROJECT_ROOT}/../monitoring"
GRAFANA_DIR="${MONITORING_DIR}/grafana"
PROMETHEUS_DIR="${MONITORING_DIR}/prometheus"

mkdir -p "${GRAFANA_DIR}"
mkdir -p "${PROMETHEUS_DIR}"

# Create Prometheus configuration
cat > "${PROMETHEUS_DIR}/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']
EOF

# Create Docker Compose file
DOCKER_COMPOSE_FILE="${MONITORING_DIR}/docker-compose.yml"

cat > "${DOCKER_COMPOSE_FILE}" << EOF
version: '3.8'

services:
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "9400:9400"
    command: -f /etc/dcgm-exporter/dcp-metrics-included.csv

  prometheus:
    image: prom/prometheus:v2.42.0
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ${PROMETHEUS_DIR}:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:9.5.1
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
EOF

echo "Configuration files created."
echo "Starting monitoring stack..."

# Start the monitoring stack
cd "${MONITORING_DIR}"
docker-compose up -d

echo "Waiting for services to start..."
sleep 10

# Configure Grafana
GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="admin"

# Add Prometheus data source
curl -s -X POST -H "Content-Type: application/json" -d '{
    "name":"Prometheus",
    "type":"prometheus",
    "url":"http://prometheus:9090",
    "access":"proxy",
    "basicAuth":false
}' -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" "${GRAFANA_URL}/api/datasources" > /dev/null

echo "Monitoring stack is running!"
echo "Grafana dashboard is available at: ${GRAFANA_URL}"
echo "Login with username: ${GRAFANA_USER}, password: ${GRAFANA_PASSWORD}"
echo ""
echo "To import NVIDIA GPU dashboard:"
echo "1. Go to ${GRAFANA_URL}"
echo "2. Login with the credentials above"
echo "3. Click on '+' > 'Import'"
echo "4. Enter dashboard ID: 12239 (NVIDIA DCGM Exporter Dashboard)"
echo "5. Select the Prometheus data source and click 'Import'"
echo ""
echo "GPU monitoring is now set up with persistent storage!"
