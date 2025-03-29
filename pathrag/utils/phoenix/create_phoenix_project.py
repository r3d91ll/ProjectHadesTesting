#!/usr/bin/env python3
"""
Phoenix Project Creator

This utility script creates a new project in Phoenix directly using the raw API.
"""

import os
import sys
import logging
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phoenix_project_creator")

# Phoenix settings
PHOENIX_HOST = os.environ.get("PHOENIX_HOST", "localhost")
PHOENIX_PORT = os.environ.get("PHOENIX_PORT", "8084")
PROJECT_NAME = "pathrag-dataset-builder"

def create_phoenix_project():
    """Create a project in Phoenix directly using the API"""
    
    # First, check if Phoenix is running
    health_url = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}/health"
    try:
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ Phoenix is running at {health_url}")
        else:
            logger.error(f"❌ Phoenix returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Could not connect to Phoenix: {e}")
        return False
    
    # Try creating a project directly
    logger.info(f"Creating project '{PROJECT_NAME}'...")
    
    # 1. Try the management API endpoint
    try:
        project_url = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}/api/projects"
        payload = {
            "name": PROJECT_NAME,
            "display_name": "PathRAG Dataset Builder",
            "description": "Telemetry data from the PathRAG dataset builder"
        }
        
        response = requests.post(
            project_url, 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 201]:
            logger.info(f"✅ Successfully created project: {PROJECT_NAME}")
            return True
        else:
            logger.warning(f"⚠️ Failed to create project via management API: {response.status_code}")
            logger.warning(f"Response: {response.text}")
    except Exception as e:
        logger.warning(f"⚠️ Error creating project via management API: {e}")
    
    # 2. Try sending a record to the project to create it implicitly
    try:
        logger.info("Trying to create project by sending a record...")
        
        trace_url = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}/v1/traces"
        
        # Create a minimal OTLP trace
        trace_id = "01020304050607080102030405060708"
        span_id = "0102030405060708"
        timestamp = int(datetime.now().timestamp() * 1_000_000_000)  # nanoseconds
        
        payload = {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "pathrag-dataset-builder"}},
                        {"key": "telemetry.sdk.language", "value": {"stringValue": "python"}},
                        {"key": "telemetry.sdk.name", "value": {"stringValue": "opentelemetry"}}
                    ]
                },
                "scopeSpans": [{
                    "scope": {
                        "name": "pathrag-dataset-builder"
                    },
                    "spans": [{
                        "traceId": trace_id,
                        "spanId": span_id,
                        "name": "test-span",
                        "kind": 1,  # INTERNAL
                        "startTimeUnixNano": timestamp,
                        "endTimeUnixNano": timestamp + 1_000_000_000,  # 1 second later
                        "attributes": [
                            {"key": "project", "value": {"stringValue": PROJECT_NAME}}
                        ]
                    }]
                }]
            }]
        }
        
        response = requests.post(
            trace_url, 
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Phoenix-Project": PROJECT_NAME
            }
        )
        
        if response.status_code == 200:
            logger.info(f"✅ Successfully sent trace to project: {PROJECT_NAME}")
            logger.info("Check your Phoenix UI for the new project!")
            return True
        else:
            logger.warning(f"⚠️ Failed to send trace: {response.status_code}")
            logger.warning(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error sending test trace: {e}")
        return False
        
    return False

if __name__ == "__main__":
    logger.info("Starting Phoenix project creator...")
    create_phoenix_project()
