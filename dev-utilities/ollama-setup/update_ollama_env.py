#!/usr/bin/env python3
"""
Temporary utility script to update the PathRAG .env file with Ollama configuration.
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

# Ollama configuration to add
ollama_config = """
# LLM Provider (openai or ollama)
LLM_PROVIDER=ollama

# Ollama API settings
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=tinyllama
"""

# Check if LLM_PROVIDER already exists
if "LLM_PROVIDER" in env_content:
    print(f"✅ LLM_PROVIDER already exists in .env file")
else:
    # Add Ollama configuration
    env_content += ollama_config
    
    # Write the updated .env file
    with open(env_file_path, "w") as f:
        f.write(env_content)
    
    print(f"✅ Added Ollama configuration to .env file")

print("\n✅ Environment file update complete")
print("\nTo test the updated configuration, run:")
print(f"cd {pathrag_path} && source venv/bin/activate && python src/pathrag_runner.py --query \"What are the key principles of RAG systems?\" --session-id \"test_ollama\"")
