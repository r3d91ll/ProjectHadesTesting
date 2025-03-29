#!/usr/bin/env python3
"""
Test script to generate a distinctive trace in Phoenix for the pathrag-inference project.
This will help us verify that traces are being properly sent to Phoenix.
"""

import os
import sys
import time
import uuid
from datetime import datetime

# Add the necessary directories to the Python path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pathrag_path = os.path.join(root_path, "pathrag")
sys.path.insert(0, pathrag_path)
sys.path.insert(0, root_path)  # Add the root path for implementations

# Import PathRAG configuration and adapter
from config.pathrag_config import get_config
from implementations.pathrag.arize_integration.adapter import PathRAGArizeAdapter

def test_phoenix_trace():
    """Generate a test trace with a distinctive name."""
    print("\n🔍 Testing Phoenix trace generation...\n")
    
    # Get PathRAG configuration
    config = get_config()
    
    # Print Phoenix configuration
    print(f"📊 Phoenix Configuration:")
    print(f"  Host: {config.get('phoenix_host', 'Not set')}")
    print(f"  Port: {config.get('phoenix_port', 'Not set')}")
    print(f"  Project: {config.get('project_name', 'Not set')}")
    
    # Create a distinctive trace ID and name
    trace_id = str(uuid.uuid4())
    trace_name = f"test-trace-{int(time.time())}"
    
    # Initialize PathRAG adapter
    adapter = PathRAGArizeAdapter(config)
    adapter.initialize()
    
    # Generate a test query and response
    query = f"This is a test query with ID: {trace_id}"
    response = f"This is a test response for query: {trace_id}"
    
    # Create path information for telemetry
    path_info = [
        {"text": "Test document 1", "score": 0.95, "metadata": {"source": "test1.pdf"}},
        {"text": "Test document 2", "score": 0.85, "metadata": {"source": "test2.pdf"}}
    ]
    
    # Log telemetry to Phoenix
    metadata = {
        "session_id": trace_name,
        "user_id": "test-user",
        "timestamp": datetime.now().isoformat(),
        "test": True
    }
    
    # Calculate latency
    latency_ms = 10.5
    
    # Simulate token usage
    token_usage = {
        "prompt_tokens": 50,
        "completion_tokens": 30,
        "total_tokens": 80
    }
    
    # Log the trace
    print(f"\n📝 Logging trace to Phoenix...")
    print(f"  Trace ID: {trace_id}")
    print(f"  Trace Name: {trace_name}")
    
    logged_trace_id = adapter.log_telemetry(
        trace_id=trace_id,
        query=query,
        response=response,
        path=path_info,
        latency_ms=latency_ms,
        token_usage=token_usage,
        metadata=metadata
    )
    
    if logged_trace_id:
        print(f"\n✅ Successfully logged trace to Phoenix")
        print(f"  Trace ID: {logged_trace_id}")
        print(f"  Project: {config.get('project_name', 'Not set')}")
        print(f"  Phoenix URL: http://{config.get('phoenix_host', 'localhost')}:{config.get('phoenix_port', '8084')}")
        
        # Instructions for checking the trace in Phoenix
        print("\n📋 To check the trace in Phoenix:")
        print(f"  1. Open http://{config.get('phoenix_host', 'localhost')}:{config.get('phoenix_port', '8084')} in your browser")
        print(f"  2. Select the project '{config.get('project_name', 'pathrag-inference')}'")
        print(f"  3. Look for a trace with name containing '{trace_name}'")
    else:
        print(f"\n❌ Failed to log trace to Phoenix")

if __name__ == "__main__":
    test_phoenix_trace()
