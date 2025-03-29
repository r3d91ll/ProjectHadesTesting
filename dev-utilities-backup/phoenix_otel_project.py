#!/usr/bin/env python3
"""
Phoenix Project Creator using OTEL

This script creates a project in Phoenix using the OpenTelemetry approach.
"""

import os
import sys
import time
from pathlib import Path

# Project name for PathRAG
PROJECT_NAME = "pathrag-inference"

# Phoenix endpoint
PHOENIX_HOST = os.environ.get("PHOENIX_HOST", "localhost")
PHOENIX_PORT = os.environ.get("PHOENIX_PORT", "8084")
PHOENIX_ENDPOINT = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}/v1/traces"

print(f"🔍 Creating project '{PROJECT_NAME}' in Phoenix at {PHOENIX_ENDPOINT}...")

try:
    # Import Phoenix OTEL
    from phoenix.otel import register
    
    # Register the project
    tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint=PHOENIX_ENDPOINT
    )
    
    print(f"✅ Created project '{PROJECT_NAME}' in Phoenix")
    
    # Send a test span
    from opentelemetry import trace
    tracer = trace.get_tracer("pathrag-test")
    
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("test.attribute", "test-value")
        span.add_event("test-event")
        time.sleep(1)  # Give time for the span to be processed
    
    print(f"✅ Sent test span to project '{PROJECT_NAME}'")
    
except ImportError:
    print("❌ Phoenix OTEL not installed. Installing...")
    os.system("pip install phoenix-otel")
    
    # Try again after installation
    try:
        from phoenix.otel import register
        
        # Register the project
        tracer_provider = register(
            project_name=PROJECT_NAME,
            endpoint=PHOENIX_ENDPOINT
        )
        
        print(f"✅ Created project '{PROJECT_NAME}' in Phoenix")
    except Exception as e:
        print(f"❌ Failed to create project: {e}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n✅ Done! Project should now be visible in Phoenix.")
print(f"   Visit: http://{PHOENIX_HOST}:{PHOENIX_PORT}")

# Create a .env file with the project name for PathRAG
ENV_FILE = Path("/home/todd/ML-Lab/New-HADES/pathrag/.env")
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        env_content = f.read()
    
    if "PHOENIX_PROJECT_NAME" not in env_content:
        with open(ENV_FILE, 'a') as f:
            f.write(f"\n# Phoenix Project Name\nPHOENIX_PROJECT_NAME={PROJECT_NAME}\n")
        print(f"✅ Added PHOENIX_PROJECT_NAME to PathRAG .env file")
    else:
        print("✅ PHOENIX_PROJECT_NAME already in PathRAG .env file")
else:
    with open(ENV_FILE, 'w') as f:
        f.write(f"# Phoenix Project Name\nPHOENIX_PROJECT_NAME={PROJECT_NAME}\n")
    print(f"✅ Created PathRAG .env file with PHOENIX_PROJECT_NAME")

print("\nNow run a PathRAG query to see if traces are logged to the project:")
print("   cd /home/todd/ML-Lab/New-HADES/pathrag")
print("   source venv/bin/activate")
print('   python src/pathrag_runner.py --query "What is the transformer architecture?" --session-id "test_project"')
