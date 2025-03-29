#!/usr/bin/env python3
"""
Temporary utility script to update the PathRAG .env file to use the alternative Qwen model.
This script is placed in the dev-utilities directory as per the project's organization standards.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the pathrag directory to the Python path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pathrag_path = os.path.join(root_path, "pathrag")

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

# Update the OLLAMA_MODEL in the .env file
if "OLLAMA_MODEL=" in env_content:
    # Replace the current model with the alternative Qwen model
    env_content = env_content.replace(
        "OLLAMA_MODEL=qwen2.5-128k", 
        "OLLAMA_MODEL=mbenhamd/qwen2.5-14b-instruct-cline-128k-q8_0"
    )
    
    # Write the updated .env file
    with open(env_file_path, "w") as f:
        f.write(env_content)
    
    print(f"✅ Updated .env file to use mbenhamd/qwen2.5-14b-instruct-cline-128k-q8_0 model")
else:
    print(f"❌ OLLAMA_MODEL not found in .env file")

print("\n✅ Environment file update complete")
print("\nTo test the updated configuration, run:")
print(f"cd {pathrag_path} && source venv/bin/activate && python src/pathrag_runner.py --query \"What are the key principles of RAG systems?\" --session-id \"test_qwen_alt\"")
