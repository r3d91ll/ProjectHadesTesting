# GPU Monitoring for LLM Experiments

This document explains how the GPU monitoring infrastructure integrates with LLM experiments, particularly PathRAG and GraphRAG implementations.

## Overview

The HADES project includes a robust GPU monitoring system to track resource utilization during LLM inference and training. This is especially important for comparing performance characteristics of different RAG implementations (PathRAG, GraphRAG) and model variants (Qwen 2.5, XnX, etc.).

## Monitoring Infrastructure

### Components

1. **Python GPU Metrics Exporter** (`/ladon/scripts/gpu_metrics_exporter.py`)
   - Collects GPU metrics using the NVIDIA SMI tool
   - Exposes metrics via Prometheus endpoint
   - Tracks per-GPU utilization, memory usage, temperature, and more

2. **Prometheus** (`/ladon/prometheus/`)
   - Time-series database for storing metrics
   - Scrapes GPU metrics from the exporter
   - Handles alerting based on thresholds

3. **Grafana** (`/ladon/grafana/`)
   - Visualization platform for GPU metrics
   - Provides pre-configured dashboards for monitoring
   - Supports custom dashboards for specific experiment tracking

4. **Node Exporter**
   - Collects system-level metrics (CPU, memory, disk)
   - Complements GPU-specific metrics

5. **Unified Start Script** (`/ladon/scripts/start_monitoring.sh`)
   - Single entry point to launch the complete monitoring stack
   - Handles dependencies and proper shutdown

## Key Metrics for LLM Experiments

### GPU Utilization Metrics

| Metric | Description | Relevance to LLM Experiments |
|--------|-------------|------------------------------|
| `gpu_utilization_percent` | GPU compute utilization | Indicates LLM inference efficiency |
| `gpu_memory_used_bytes` | GPU memory consumption | Critical for max batch size determination |
| `gpu_memory_free_bytes` | Available GPU memory | Helps optimize context length |
| `gpu_temperature_celsius` | GPU temperature | Important for long-running experiments |
| `gpu_power_watts` | Power consumption | Energy efficiency of different models |
| `gpu_memory_bandwidth` | Memory throughput | Identifies memory bottlenecks |

### Experiment-Specific Metrics

Additional metrics specific to LLM experiments:

- **Tokens per second** - Processing speed metric
- **Batch completion time** - Time to process a batch of queries
- **Memory per token** - Memory efficiency metric
- **Context loading time** - Time spent loading context
- **Inference vs. retrieval ratio** - Balance of time spent

## Setup for LLM Experiments

### Pre-Experiment Setup

1. Start the monitoring stack:
   ```bash
   cd /home/todd/ML-Lab/New-HADES/ladon/scripts
   ./start_monitoring.sh
   ```

2. Verify all components are running:
   - Python GPU Exporter (check logs in `/tmp/gpu_exporter.log`)
   - Prometheus (accessible at http://localhost:9090)
   - Grafana (accessible at http://localhost:3000)

3. Configure experiment-specific dashboards:
   - Import the LLM Experiment dashboard template
   - Set appropriate time ranges for your experiment

### Monitoring During Experiments

When running PathRAG or GraphRAG experiments:

1. **Baseline Measurements**:
   - Record idle GPU metrics before starting
   - Note baseline temperature and power draw

2. **Experiment Tagging**:
   - Add annotations in Grafana for experiment start/end
   - Use consistent tags for different implementations

3. **Critical Thresholds**:
   - Memory usage should not exceed 90% to avoid OOM errors
   - GPU temperature should remain below 80°C for NVIDIA RTX A6000 GPUs

## Integration with PathRAG Implementations

### PathRAG Original vs. Qwen25

The monitoring system is particularly useful for comparing:

- Memory efficiency of hosted vs. local models
- Throughput differences between implementation variants
- Power consumption of different model architectures

### Resource Requirements

| Implementation | Typical Memory | Utilization | Power Draw | Notes |
|----------------|----------------|-------------|------------|-------|
| PathRAG (OpenAI) | 2-4 GB | 40-60% | 70-100W | API-dependent latency |
| PathRAG (Qwen25) | 10-12 GB | 80-95% | 150-180W | Higher local GPU usage |
| GraphRAG | 6-8 GB | 70-85% | 120-150W | Graph operations intensive |
| XnX Variant | 8-10 GB | 75-90% | 130-160W | Additional parsing overhead |

## Visualizing Results

### Grafana Dashboards

The monitoring system includes specialized dashboards:

1. **LLM Overview** - High-level metrics across all GPUs
2. **PathRAG Performance** - Detailed metrics for PathRAG variants
3. **Model Comparison** - Side-by-side view of different implementations

### Exporting Results

To capture GPU metrics for publication or analysis:

1. Use Grafana's export feature to save charts as images
2. Export raw metrics as CSV for statistical analysis
3. Save dashboard JSON configuration for reproducibility

## Troubleshooting

Common issues and solutions:

1. **Missing GPU Metrics**
   - Verify nvidia-smi is working: `nvidia-smi --query`
   - Check GPU exporter logs: `cat /tmp/gpu_exporter.log`
   - Restart the Python exporter: `pkill -f gpu_metrics_exporter.py && python3 /ladon/scripts/gpu_metrics_exporter.py`

2. **Out of Memory Errors**
   - Reduce batch size or context length
   - Monitor `gpu_memory_used_bytes` to catch approaching limits
   - Consider model quantization (8-bit or 4-bit precision)

3. **Performance Degradation**
   - Check for thermal throttling when `gpu_temperature_celsius` is high
   - Monitor system CPU/memory alongside GPU metrics
   - Check for competing workloads on shared GPUs

## Best Practices

1. **Regular Baseline Collection**
   - Periodically measure baseline performance with standard queries
   - Track changes over time to detect drift

2. **Experiment Isolation**
   - Run one model variant at a time for clean comparisons
   - Allow cool-down periods between experiments

3. **Configuration Tracking**
   - Record model parameters alongside GPU metrics
   - Document any system changes that might affect performance

## References

- NVIDIA SMI Documentation
- Prometheus and Grafana Documentation
- HADES GPU Monitoring Implementation Details
