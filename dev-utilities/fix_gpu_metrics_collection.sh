#!/bin/bash
# Script to fix GPU metrics collection for Grafana
# This script ensures that real-time GPU metrics are properly collected and sent to Prometheus

set -e

echo "🔧 Fixing GPU metrics collection for Grafana..."

# Stop any existing GPU metrics exporters
echo "Stopping existing GPU metrics exporters..."
sudo pkill -f "gpu_metrics_exporter" || true

# Create metrics directory if it doesn't exist
echo "Creating metrics directory..."
sudo mkdir -p $HOME/gpu_metrics
sudo chmod 777 $HOME/gpu_metrics

# Create a wrapper script to run the Python GPU metrics exporter
echo "Creating GPU metrics collection script..."
cat > /tmp/collect_gpu_metrics.sh << 'EOF'
#!/bin/bash
# Collect GPU metrics and write to node_exporter textfile directory

METRICS_FILE="$HOME/gpu_metrics/gpu_metrics.prom"

while true; do
  # Get GPU metrics using nvidia-smi
  echo "# HELP gpu_utilization_percent GPU utilization percentage" > $METRICS_FILE
  echo "# TYPE gpu_utilization_percent gauge" >> $METRICS_FILE
  
  echo "# HELP gpu_memory_used_mb GPU memory used in MB" >> $METRICS_FILE
  echo "# TYPE gpu_memory_used_mb gauge" >> $METRICS_FILE
  
  echo "# HELP gpu_memory_total_mb GPU total memory in MB" >> $METRICS_FILE
  echo "# TYPE gpu_memory_total_mb gauge" >> $METRICS_FILE
  
  echo "# HELP gpu_temperature_celsius GPU temperature in Celsius" >> $METRICS_FILE
  echo "# TYPE gpu_temperature_celsius gauge" >> $METRICS_FILE
  
  # Get GPU metrics
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS="," read -r gpu_id gpu_name gpu_util mem_used mem_total temp; do
    # Clean up values
    gpu_id=$(echo $gpu_id | xargs)
    gpu_name=$(echo $gpu_name | xargs | sed 's/ /_/g')
    gpu_util=$(echo $gpu_util | xargs)
    mem_used=$(echo $mem_used | xargs)
    mem_total=$(echo $mem_total | xargs)
    temp=$(echo $temp | xargs)
    
    # Write metrics
    echo "gpu_utilization_percent{gpu_id=\"$gpu_id\",gpu_name=\"$gpu_name\"} $gpu_util" >> $METRICS_FILE
    echo "gpu_memory_used_mb{gpu_id=\"$gpu_id\",gpu_name=\"$gpu_name\"} $mem_used" >> $METRICS_FILE
    echo "gpu_memory_total_mb{gpu_id=\"$gpu_id\",gpu_name=\"$gpu_name\"} $mem_total" >> $METRICS_FILE
    echo "gpu_temperature_celsius{gpu_id=\"$gpu_id\",gpu_name=\"$gpu_name\"} $temp" >> $METRICS_FILE
    
    echo "Updated GPU $gpu_id metrics: Util: ${gpu_util}%, Mem: ${mem_used}/${mem_total}MB, Temp: ${temp}°C"
  done
  
  # Update every 5 seconds
  sleep 5
done
EOF

chmod +x /tmp/collect_gpu_metrics.sh

# Start the metrics collector in the background
echo "Starting GPU metrics collector..."
sudo nohup /tmp/collect_gpu_metrics.sh > /tmp/gpu_metrics.log 2>&1 &

# Restart node-exporter to pick up the new metrics
echo "Restarting node-exporter container..."
cd /home/todd/ML-Lab/New-HADES/ladon
docker-compose restart node-exporter

echo "✅ GPU metrics collection fixed! Metrics should now appear in Grafana."
echo "You can check the logs at /tmp/gpu_metrics.log"
