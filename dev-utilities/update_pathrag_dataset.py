#!/usr/bin/env python3
"""
Temporary utility script to update PathRAG configuration to use the RAG dataset builder's output.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the pathrag directory to the Python path
pathrag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "pathrag")
sys.path.insert(0, pathrag_path)

# Get the path to the RAG dataset
rag_dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "rag_databases/current")

# Check if the RAG dataset exists
if not os.path.exists(rag_dataset_path):
    print(f"❌ RAG dataset not found at {rag_dataset_path}")
    sys.exit(1)

print(f"✅ Found RAG dataset at {rag_dataset_path}")

# Update the .env file to use the RAG dataset
env_file_path = os.path.join(pathrag_path, ".env")

# Create a backup of the .env file
if os.path.exists(env_file_path):
    backup_path = env_file_path + ".bak"
    shutil.copy2(env_file_path, backup_path)
    print(f"✅ Created backup of .env file at {backup_path}")

# Read the current .env file or create a new one
env_content = ""
if os.path.exists(env_file_path):
    with open(env_file_path, "r") as f:
        env_content = f.read()

# Update or add the DOCUMENT_STORE_PATH variable
if "DOCUMENT_STORE_PATH" in env_content:
    # Replace the existing value
    lines = env_content.split("\n")
    updated_lines = []
    for line in lines:
        if line.startswith("DOCUMENT_STORE_PATH="):
            updated_lines.append(f"DOCUMENT_STORE_PATH={rag_dataset_path}")
        else:
            updated_lines.append(line)
    env_content = "\n".join(updated_lines)
else:
    # Add the variable
    env_content += f"\nDOCUMENT_STORE_PATH={rag_dataset_path}"

# Write the updated .env file
with open(env_file_path, "w") as f:
    f.write(env_content)

print(f"✅ Updated .env file to use RAG dataset at {rag_dataset_path}")

# Now run a test query to verify the configuration
print("\nRunning test query to verify configuration...")
print("Command: python src/pathrag_runner.py --query \"What is the transformer architecture?\" --session-id \"test_dataset\"")
print("\nTo run this query, execute the following command:")
print(f"cd {pathrag_path} && source venv/bin/activate && python src/pathrag_runner.py --query \"What is the transformer architecture?\" --session-id \"test_dataset\"")
