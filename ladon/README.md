# Ladon Monitoring Stack

## Overview

Ladon is a comprehensive monitoring stack designed for ML experimentation and system observability. This stack provides robust monitoring capabilities for both hardware resources and ML model performance, allowing for consistent benchmarking and performance analysis.

## Components

- **Prometheus**: Time-series database for metrics collection
- **Grafana**: Visualization and dashboarding
- **Node Exporter**: Hardware and OS metrics
- **cAdvisor**: Container metrics
- **Process Exporter**: Process-level metrics
- **Arize Phoenix**: ML model monitoring and evaluation

## Directory Structure

```
ladon/
├── README.md                 # This file
├── docker-compose.yml        # Main Docker Compose configuration
├── grafana/                  # Grafana configuration
│   └── provisioning/         # Grafana dashboards and datasources
├── prometheus/               # Prometheus configuration
│   ├── prometheus.yml        # Main Prometheus config
│   ├── process-exporter.yml  # Process exporter config
│   └── jmx_exporter/         # JMX exporter for Neo4j monitoring
└── scripts/                  # Helper scripts
    ├── metrics_exporter.py   # Python metrics exporter
    ├── start_monitoring.sh   # Container startup script
    └── monitoring.sh         # Monitoring stack management script
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose

### Starting the Stack

```bash
# Navigate to the Ladon directory
cd ladon

# Make the monitoring script executable
chmod +x monitoring.sh

# Start the monitoring stack
./monitoring.sh start
```

### Accessing the Interfaces

- **Grafana**: http://localhost:3000 (admin/admin_password)
- **Prometheus**: http://localhost:9090
- **Arize Phoenix**: http://localhost:8084

## Integration with ProjectHadesTesting

The Ladon monitoring stack is designed to work seamlessly with the ProjectHadesTesting experiments. It provides:

1. **System Monitoring**: Track CPU, memory, disk, and network usage during experiments
2. **Neo4j Monitoring**: Monitor Neo4j database performance for GraphRAG experiments
3. **ML Model Monitoring**: Track model performance metrics, latency, and accuracy
4. **Experiment Tracking**: Record experiment runs, parameters, and results

## Custom Metrics

The monitoring stack includes custom metrics for:

- PathRAG with NetworkX
- GraphRAG with Neo4j 
- Qwen2.5 Coder model performance
- XnX notation effectiveness
