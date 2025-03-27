#!/bin/bash
# GPU metrics exporter script for Prometheus

METRICS_DIR="/tmp/node_exporter_metrics"
METRICS_FILE="${METRICS_DIR}/gpu_metrics.prom"
INTERVAL=5  # Collection interval in seconds

# Create metrics directory if it doesn't exist
mkdir -p ${METRICS_DIR}

# Main collection loop
while true; do
    # Clear existing metrics file
    > ${METRICS_FILE}

    # Get GPU count and write metric
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | tr -d ' ' | head -1)
    echo "# HELP gpu_count Number of GPUs available" >> ${METRICS_FILE}
    echo "# TYPE gpu_count gauge" >> ${METRICS_FILE}
    echo "gpu_count ${GPU_COUNT}" >> ${METRICS_FILE}
    
    # Define all metric types once
    echo "# HELP gpu_utilization_percent GPU utilization percentage" >> ${METRICS_FILE}
    echo "# TYPE gpu_utilization_percent gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_memory_utilization_percent GPU memory utilization percentage" >> ${METRICS_FILE}
    echo "# TYPE gpu_memory_utilization_percent gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_memory_used_mb GPU memory used in MB" >> ${METRICS_FILE}
    echo "# TYPE gpu_memory_used_mb gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_memory_total_mb GPU total memory in MB" >> ${METRICS_FILE}
    echo "# TYPE gpu_memory_total_mb gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_temperature_celsius GPU temperature in Celsius" >> ${METRICS_FILE}
    echo "# TYPE gpu_temperature_celsius gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_power_usage_watts GPU power usage in watts" >> ${METRICS_FILE}
    echo "# TYPE gpu_power_usage_watts gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_power_limit_watts GPU power limit in watts" >> ${METRICS_FILE}
    echo "# TYPE gpu_power_limit_watts gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_clock_mhz GPU clock frequency in MHz" >> ${METRICS_FILE}
    echo "# TYPE gpu_clock_mhz gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_memory_clock_mhz GPU memory clock frequency in MHz" >> ${METRICS_FILE}
    echo "# TYPE gpu_memory_clock_mhz gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_max_clock_mhz GPU maximum clock frequency in MHz" >> ${METRICS_FILE}
    echo "# TYPE gpu_max_clock_mhz gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_max_memory_clock_mhz GPU maximum memory clock frequency in MHz" >> ${METRICS_FILE}
    echo "# TYPE gpu_max_memory_clock_mhz gauge" >> ${METRICS_FILE}
    echo "# HELP gpu_fan_speed_percent GPU fan speed percentage" >> ${METRICS_FILE}
    echo "# TYPE gpu_fan_speed_percent gauge" >> ${METRICS_FILE}

    # Get detailed GPU metrics
    nvidia-smi --query-gpu=index,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,fan.speed --format=csv,noheader,nounits | while IFS=, read -r gpu_id gpu_name gpu_util mem_util mem_used mem_total temp power_draw power_limit gpu_clock mem_clock gpu_clock_max mem_clock_max fan_speed; do
        # Clean up values
        gpu_id=$(echo $gpu_id | tr -d ' ')
        gpu_name=$(echo $gpu_name | tr -d ' ' | tr -d '[:punct:]' | tr '[:upper:]' '[:lower:]')
        gpu_util=$(echo $gpu_util | tr -d ' ')
        mem_util=$(echo $mem_util | tr -d ' ')
        mem_used=$(echo $mem_used | tr -d ' ')
        mem_total=$(echo $mem_total | tr -d ' ')
        temp=$(echo $temp | tr -d ' ')
        
        # Handle N/A values
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
        
        if [[ "$gpu_clock" == *"N/A"* ]]; then
            gpu_clock="0"
        else
            gpu_clock=$(echo $gpu_clock | tr -d ' ')
        fi
        
        if [[ "$mem_clock" == *"N/A"* ]]; then
            mem_clock="0"
        else
            mem_clock=$(echo $mem_clock | tr -d ' ')
        fi
        
        if [[ "$gpu_clock_max" == *"N/A"* ]]; then
            gpu_clock_max="0"
        else
            gpu_clock_max=$(echo $gpu_clock_max | tr -d ' ')
        fi
        
        if [[ "$mem_clock_max" == *"N/A"* ]]; then
            mem_clock_max="0"
        else
            mem_clock_max=$(echo $mem_clock_max | tr -d ' ')
        fi
        
        if [[ "$fan_speed" == *"N/A"* ]]; then
            fan_speed="0"
        else
            fan_speed=$(echo $fan_speed | tr -d ' ')
        fi

        # Write metrics to file
        echo "gpu_utilization_percent{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${gpu_util}" >> ${METRICS_FILE}
        
        echo "gpu_memory_utilization_percent{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${mem_util}" >> ${METRICS_FILE}
        
        echo "gpu_memory_used_mb{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${mem_used}" >> ${METRICS_FILE}
        
        echo "gpu_memory_total_mb{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${mem_total}" >> ${METRICS_FILE}
        
        echo "gpu_temperature_celsius{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${temp}" >> ${METRICS_FILE}
        
        echo "gpu_power_usage_watts{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${power_draw}" >> ${METRICS_FILE}
        
        echo "gpu_power_limit_watts{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${power_limit}" >> ${METRICS_FILE}
        
        echo "gpu_clock_mhz{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${gpu_clock}" >> ${METRICS_FILE}
        
        echo "gpu_memory_clock_mhz{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${mem_clock}" >> ${METRICS_FILE}
        
        echo "gpu_max_clock_mhz{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${gpu_clock_max}" >> ${METRICS_FILE}
        
        echo "gpu_max_memory_clock_mhz{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${mem_clock_max}" >> ${METRICS_FILE}
        
        echo "gpu_fan_speed_percent{gpu_id=\"${gpu_id}\", gpu_name=\"${gpu_name}\"} ${fan_speed}" >> ${METRICS_FILE}
    done

    # Set proper permissions so node exporter can read it
    chmod 644 ${METRICS_FILE}
    # Change ownership to nobody:nogroup for node-exporter compatibility
    chown nobody:nogroup ${METRICS_FILE}
    
    # Sleep for the collection interval
    sleep ${INTERVAL}
done
