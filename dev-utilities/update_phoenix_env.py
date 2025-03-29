#!/usr/bin/env python3
"""
Temporary utility script to update the PathRAG .env file with the new PHOENIX_INFERENCE_PROJECT_NAME variable.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the pathrag directory to the Python path
pathrag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "pathrag")

# Path to the .env file
env_file_path = os.path.join(pathrag_path, ".env")

# Create a backup of the .env file
if os.path.exists(env_file_path):
    backup_path = env_file_path + ".bak"
    shutil.copy2(env_file_path, backup_path)
    print(f"✅ Created backup of .env file at {backup_path}")

# Read the current .env file
env_content = ""
if os.path.exists(env_file_path):
    with open(env_file_path, "r") as f:
        env_content = f.read()

# Check if PHOENIX_PROJECT_NAME exists
if "PHOENIX_PROJECT_NAME" in env_content:
    print(f"✅ Found PHOENIX_PROJECT_NAME in .env file")
    
    # Check if PHOENIX_INFERENCE_PROJECT_NAME already exists
    if "PHOENIX_INFERENCE_PROJECT_NAME" in env_content:
        print(f"✅ PHOENIX_INFERENCE_PROJECT_NAME already exists in .env file")
    else:
        # Add PHOENIX_INFERENCE_PROJECT_NAME
        lines = env_content.split("\n")
        updated_lines = []
        for line in lines:
            updated_lines.append(line)
            if line.startswith("PHOENIX_PROJECT_NAME="):
                project_name = line.split("=")[1]
                updated_lines.append(f"PHOENIX_INFERENCE_PROJECT_NAME=pathrag-inference")
        
        env_content = "\n".join(updated_lines)
        
        # Write the updated .env file
        with open(env_file_path, "w") as f:
            f.write(env_content)
        
        print(f"✅ Added PHOENIX_INFERENCE_PROJECT_NAME=pathrag-inference to .env file")
else:
    print(f"❌ PHOENIX_PROJECT_NAME not found in .env file")

print("\n✅ Environment file update complete")
print("\nTo test the updated configuration, run:")
print(f"cd {pathrag_path} && source venv/bin/activate && python src/pathrag_runner.py --query \"What are the key principles of RAG systems?\" --session-id \"test_inference_project\"")
