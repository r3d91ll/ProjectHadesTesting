#!/usr/bin/env python3
"""
Temporary utility script to set up the Qwen2.5-72B-Instruct model in Ollama with 32768K context length.
This script will:
1. Create a Modelfile for Qwen2.5-72B-Instruct with 32768K context length
2. Create the model in Ollama
3. Update the PathRAG .env file to use this model
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Paths
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pathrag_path = os.path.join(root_path, "pathrag")
modelfile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen2.5-72B-Instruct.modelfile")
env_file_path = os.path.join(pathrag_path, ".env")

# Model name
model_name = "qwen2.5-72b-instruct-32k"

# Check if the Modelfile exists
if not os.path.exists(modelfile_path):
    print(f"❌ Modelfile not found at {modelfile_path}")
    sys.exit(1)

# Create the model in Ollama
print(f"\n🔍 Creating {model_name} in Ollama...")
try:
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"✅ Successfully created {model_name} in Ollama")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to create {model_name} in Ollama")
    print(f"Error: {e.stderr}")
    sys.exit(1)

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
    env_content = env_content.replace(
        f"OLLAMA_MODEL=tinyllama", 
        f"OLLAMA_MODEL={model_name}"
    )
    
    # Write the updated .env file
    with open(env_file_path, "w") as f:
        f.write(env_content)
    
    print(f"✅ Updated .env file to use {model_name}")
else:
    print(f"❌ OLLAMA_MODEL not found in .env file")

print("\n✅ Setup complete")
print("\nTo test the updated configuration, run:")
print(f"cd {pathrag_path} && source venv/bin/activate && python src/pathrag_runner.py --query \"What are the key principles of RAG systems?\" --session-id \"test_qwen\"")
