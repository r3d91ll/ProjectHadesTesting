#!/usr/bin/env python3
"""
Phoenix Project Creator

This script creates a project in Phoenix and sends a test trace to it.
It uses the Arize Phoenix API directly to ensure the project is created.
"""

import os
import sys
import uuid
import json
import time
from datetime import datetime
from pathlib import Path

# Try to import Arize Phoenix
try:
    from arize.phoenix.session import Session
    from arize.phoenix.trace.trace import LLMTrace
except ImportError:
    print("❌ Arize Phoenix not installed. Installing...")
    os.system("pip install arize-phoenix")
    from arize.phoenix.session import Session
    from arize.phoenix.trace.trace import LLMTrace

# Phoenix settings
PHOENIX_HOST = os.environ.get("PHOENIX_HOST", "localhost")
PHOENIX_PORT = os.environ.get("PHOENIX_PORT", "8084")
PHOENIX_URL = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}"
PROJECT_NAME = "pathrag-inference"

print(f"🔍 Creating project '{PROJECT_NAME}' in Phoenix at {PHOENIX_URL}...")

# Create a Phoenix session with the project name
session = Session(url=PHOENIX_URL, project_name=PROJECT_NAME)
print(f"✅ Created Phoenix session for project '{PROJECT_NAME}'")

# Create a test trace
trace_id = str(uuid.uuid4())
trace = LLMTrace(
    id=trace_id,
    name="PathRAG Test",
    model="test-model",
    input="What is PathRAG?",
    output="PathRAG is a path-based retrieval augmented generation system.",
    prompt_tokens=10,
    completion_tokens=15,
    latency_ms=100,
    metadata={
        "source": "test-script",
        "timestamp": datetime.now().isoformat()
    }
)

# Log the trace to Phoenix
print(f"🔄 Logging test trace {trace_id} to project '{PROJECT_NAME}'...")
session.log_trace(trace)
print(f"✅ Logged test trace {trace_id} to project '{PROJECT_NAME}'")

print("\n✅ Done! Project should now be visible in Phoenix at:")
print(f"   {PHOENIX_URL}")
print("\nNow run a PathRAG query to see if traces are logged to the project:")
print("   cd /home/todd/ML-Lab/New-HADES/pathrag")
print("   source venv/bin/activate")
print('   python src/pathrag_runner.py --query "What is the transformer architecture?" --session-id "test_project"')
