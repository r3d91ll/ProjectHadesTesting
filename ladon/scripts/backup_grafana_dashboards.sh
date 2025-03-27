#!/bin/bash
# Script to backup Grafana dashboards to Git repository
# Run this before committing changes to Git

GRAFANA_URL="http://localhost:3000"
API_KEY="admin:admin_password"  # Using admin credentials from docker-compose
OUTPUT_DIR="../grafana/provisioning/dashboards/json"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Get list of dashboards
DASHBOARDS=$(curl -s -u "$API_KEY" "$GRAFANA_URL/api/search?type=dash-db" | jq -r '.[] | .uid')

# Export each dashboard and save to JSON file
for UID in $DASHBOARDS; do
  echo "Exporting dashboard with UID: $UID"
  DASHBOARD_JSON=$(curl -s -u "$API_KEY" "$GRAFANA_URL/api/dashboards/uid/$UID" | jq '.dashboard')
  DASHBOARD_TITLE=$(echo $DASHBOARD_JSON | jq -r '.title' | sed 's/ /-/g' | tr '[:upper:]' '[:lower:]')
  
  # Save dashboard JSON to file
  echo $DASHBOARD_JSON | jq '.' > "$OUTPUT_DIR/$DASHBOARD_TITLE.json"
  echo "Saved dashboard to $OUTPUT_DIR/$DASHBOARD_TITLE.json"
done

echo "Grafana dashboard backup complete!"
