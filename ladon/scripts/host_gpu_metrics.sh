#!/bin/bash
# Host-side GPU metrics script for Prometheus Node Exporter

METRICS_DIR="/tmp/node_exporter_metrics"
METRICS_FILE="${METRICS_DIR}/gpu_metrics.prom"

# Create metrics directory if it doesn't exist
mkdir -p ${METRICS_DIR}

# Get timestamp for metrics
TIMESTAMP=$(date +%s)

# Clear existing metrics file
> ${METRICS_FILE}

# Function to sanitize metric names for Prometheus
sanitize() {
    echo "$1" | tr -cd '[:alnum:]_' | tr '[:upper:]' '[:lower:]'
}

# Add GPU count metric directly from device query
echo "# HELP gpu_count Number of GPUs available" >> ${METRICS_FILE}
echo "# TYPE gpu_count gauge" >> ${METRICS_FILE}
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "gpu_count ${GPU_COUNT}" >> ${METRICS_FILE}

# Get metrics for each GPU
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits | while IFS=, read -r gpu_id gpu_name util mem_used mem_total temp power_draw power_limit; do
    # Clean up values
    gpu_id=$(echo $gpu_id | tr -d ' ')
    gpu_name=$(sanitize "$(echo $gpu_name | tr -d ' ')")
    util=$(echo $util | tr -d ' ')
    mem_used=$(echo $mem_used | tr -d ' ')
    mem_total=$(echo $mem_total | tr -d ' ')
    temp=$(echo $temp | tr -d ' ')
    
    # Handle possible N/A values in power metrics
    if [[ "$power_draw" == *"N/A"* ]]; then
        power_draw="0"
    else
        power_draw=$(echo $power_draw | tr -d ' ')
    fi
    
    if [[ "$power_limit" == *"N/A"* ]]; then
        power_limit="0"
    else
        power_limit=$(echo $power_limit | tr -d ' ')
    fi

    # Add GPU metrics
    echo "# HELP gpu_utilization_percent GPU utilization percentage" >> ${METRICS_FILE}
    echo "# TYPE gpu_utilization_percent gauge" >> ${METRICS_FILE}
    echo "gpu_utilization_percent{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${util}" >> ${METRICS_FILE}
    
    echo "# HELP gpu_memory_used_mb GPU memory used in MB" >> ${METRICS_FILE}
    echo "# TYPE gpu_memory_used_mb gauge" >> ${METRICS_FILE}
    echo "gpu_memory_used_mb{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${mem_used}" >> ${METRICS_FILE}
    
    echo "# HELP gpu_memory_total_mb GPU total memory in MB" >> ${METRICS_FILE}
    echo "# TYPE gpu_memory_total_mb gauge" >> ${METRICS_FILE}
    echo "gpu_memory_total_mb{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${mem_total}" >> ${METRICS_FILE}
    
    echo "# HELP gpu_temperature_celsius GPU temperature in Celsius" >> ${METRICS_FILE}
    echo "# TYPE gpu_temperature_celsius gauge" >> ${METRICS_FILE}
    echo "gpu_temperature_celsius{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${temp}" >> ${METRICS_FILE}
    
    echo "# HELP gpu_power_usage_watts GPU power usage in watts" >> ${METRICS_FILE}
    echo "# TYPE gpu_power_usage_watts gauge" >> ${METRICS_FILE}
    echo "gpu_power_usage_watts{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${power_draw}" >> ${METRICS_FILE}
    
    echo "# HELP gpu_power_limit_watts GPU power limit in watts" >> ${METRICS_FILE}
    echo "# TYPE gpu_power_limit_watts gauge" >> ${METRICS_FILE}
    echo "gpu_power_limit_watts{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${power_limit}" >> ${METRICS_FILE}
done

# Set proper permissions so node exporter can read it
chmod 644 ${METRICS_FILE}

echo "GPU metrics written to ${METRICS_FILE}"
