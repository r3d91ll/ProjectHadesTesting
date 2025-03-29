#!/usr/bin/env python3
"""
Direct Phoenix test script to verify connectivity and project setup.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_phoenix")

# Set the Phoenix environment variables directly in code
os.environ["PHOENIX_PROJECT_NAME"] = "pathrag-dataset-builder"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:8084"

try:
    from arize.phoenix.client import Client
    from arize.phoenix.types import Record, RecordType, Metric
    PHOENIX_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Phoenix modules: {e}")
    logger.error("Make sure arize-phoenix is installed: pip install arize-phoenix")
    PHOENIX_AVAILABLE = False
    sys.exit(1)

def test_phoenix_direct():
    """Test Phoenix connectivity by sending a direct trace"""
    
    # Create a Phoenix client
    logger.info("Creating Phoenix client...")
    client = Client(url="http://localhost:8084")
    
    # Create a test record
    record_id = f"test-record-{int(time.time())}"
    logger.info(f"Creating test record with ID: {record_id}")
    
    record = Record(
        id=record_id,
        record_type=RecordType.FEATURE_STORE,
        metadata={
            "test": True,
            "timestamp": datetime.now().isoformat(),
            "source": "direct_test_script"
        },
        metrics={
            "test_value": Metric(1.0),
            "test_timestamp": Metric(time.time())
        }
    )
    
    # Log the record directly
    logger.info(f"Sending record to Phoenix project: 'pathrag-dataset-builder'")
    try:
        # Try different project name formats to see which one works
        project_names = [
            "pathrag-dataset-builder",  # As specified in config
            "pathrag_dataset_builder",  # Underscores instead of hyphens
            "dataset-builder",          # Simplified name
            "default"                   # Default project
        ]
        
        for project_name in project_names:
            logger.info(f"Trying project name: '{project_name}'")
            try:
                client.log_records(
                    project_name=project_name,
                    records=[record]
                )
                logger.info(f"✅ Successfully sent record to project: '{project_name}'")
            except Exception as e:
                logger.error(f"❌ Failed to send to project '{project_name}': {e}")
        
    except Exception as e:
        logger.error(f"❌ Error sending records to Phoenix: {e}")
        return False
    
    logger.info("✅ Test complete. Check Phoenix UI for traces.")
    return True

if __name__ == "__main__":
    logger.info("Starting direct Phoenix test...")
    test_phoenix_direct()
