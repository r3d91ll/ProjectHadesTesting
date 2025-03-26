#!/bin/bash
# Script to start monitoring and then launch the application

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if metrics are enabled
if [ "${ENABLE_METRICS}" = "true" ]; then
    echo -e "${GREEN}Starting metrics exporter on port ${METRICS_PORT}...${NC}"
    
    # Start the metrics exporter in the background
    python /app/monitoring/scripts/metrics_exporter.py &
    METRICS_PID=$!
    
    # Register a trap to ensure the metrics exporter is terminated when the container stops
    trap "echo 'Stopping metrics exporter...'; kill $METRICS_PID; exit" SIGTERM SIGINT
    
    echo -e "${GREEN}Metrics exporter started with PID ${METRICS_PID}${NC}"
else
    echo -e "${YELLOW}Metrics collection is disabled${NC}"
fi

# If a command was passed to the container, execute it
if [ $# -gt 0 ]; then
    echo -e "${GREEN}Executing command: $@${NC}"
    exec "$@"
else
    # Otherwise, just keep the container running
    echo -e "${GREEN}No command specified, container will remain running${NC}"
    tail -f /dev/null
fi
