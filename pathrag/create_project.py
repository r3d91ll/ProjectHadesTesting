#!/usr/bin/env python3
"""
Create Phoenix Project

This script creates a project in Phoenix to ensure it exists before running PathRAG.
"""

import os
import sys
import requests
import json

# Phoenix settings
PHOENIX_HOST = os.environ.get("PHOENIX_HOST", "localhost")
PHOENIX_PORT = os.environ.get("PHOENIX_PORT", "8084")
PHOENIX_URL = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}"
PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "pathrag-inference")

print(f"🔍 Creating project '{PROJECT_NAME}' in Phoenix...")

try:
    # Check if project exists
    response = requests.get(f"{PHOENIX_URL}/api/projects")
    if response.status_code == 200:
        projects = response.json().get("projects", [])
        project_exists = any(p.get("name") == PROJECT_NAME for p in projects)
        
        if project_exists:
            print(f"✅ Project '{PROJECT_NAME}' already exists in Phoenix")
        else:
            # Create the project
            create_response = requests.post(
                f"{PHOENIX_URL}/api/projects",
                json={"name": PROJECT_NAME}
            )
            
            if create_response.status_code in (200, 201):
                print(f"✅ Created project '{PROJECT_NAME}' in Phoenix")
            else:
                print(f"❌ Failed to create project: {create_response.status_code}")
                print(create_response.text)
    else:
        print(f"❌ Failed to get projects: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Error: {e}")
