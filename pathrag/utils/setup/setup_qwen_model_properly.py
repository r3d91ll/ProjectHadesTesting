#!/usr/bin/env python3
"""
Utility script to properly set up the Qwen model for PathRAG.

This script will:
1. Pull the Qwen model from Ollama
2. Create a Modelfile with extended context length
3. Create the model in Ollama
4. Update the PathRAG .env file to use this model
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# Paths
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHRAG_PATH = os.path.join(ROOT_PATH, "pathrag")
ENV_FILE_PATH = os.path.join(PATHRAG_PATH, ".env")
MODELFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen2.5-128k-64k-ctx.modelfile")

# Model configuration
MODEL_NAME = "qwen2.5-128k-64k-ctx"
BASE_MODEL = "qwen2.5-128k"  # Using the model we already have

def run_command(cmd, desc=None):
    """Run a command and print its output"""
    if desc:
        print(f"\n🔍 {desc}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Command succeeded")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def create_modelfile():
    """Create a Modelfile for the Qwen model with extended context length"""
    print(f"\n🔍 Creating Modelfile at {MODELFILE_PATH}...")
    
    # Using raw string to avoid escape sequence issues
    modelfile_content = f'''FROM {BASE_MODEL}

# Set parameters for the model
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER num_ctx 65536

# System prompt that defines the model's behavior
SYSTEM """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question is asked in a language other than English, please respond in the language used to pose the question.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
'''
    
    try:
        with open(MODELFILE_PATH, "w") as f:
            f.write(modelfile_content)
        print(f"✅ Successfully created Modelfile")
        return True
    except Exception as e:
        print(f"❌ Failed to create Modelfile: {e}")
        return False

def update_env_file(model_name):
    """Update the .env file to use the specified model"""
    print(f"\n🔍 Updating .env file to use {model_name}...")
    
    # Create a backup of the .env file
    if os.path.exists(ENV_FILE_PATH):
        backup_path = ENV_FILE_PATH + ".bak"
        shutil.copy2(ENV_FILE_PATH, backup_path)
        print(f"✅ Created backup of .env file at {backup_path}")
    
    # Read the current .env file
    env_content = ""
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, "r") as f:
            env_content = f.read()
    
    # Update the OLLAMA_MODEL in the .env file
    if "OLLAMA_MODEL=" in env_content:
        # Find the current model name
        import re
        current_model = re.search(r'OLLAMA_MODEL=([^\n]+)', env_content)
        if current_model:
            current_model = current_model.group(1)
            env_content = env_content.replace(
                f"OLLAMA_MODEL={current_model}", 
                f"OLLAMA_MODEL={model_name}"
            )
            
            # Write the updated .env file
            with open(ENV_FILE_PATH, "w") as f:
                f.write(env_content)
            
            print(f"✅ Updated .env file to use {model_name}")
            return True
        else:
            print(f"❌ Could not find OLLAMA_MODEL in .env file")
            return False
    else:
        print(f"❌ OLLAMA_MODEL not found in .env file")
        return False

def main():
    """Main function to set up the Qwen model"""
    print("=" * 80)
    print(f"Setting up Qwen model for PathRAG")
    print("=" * 80)
    
    # Step 1: Pull the base model if needed
    print(f"\n🔍 Checking if {BASE_MODEL} is already available...")
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    
    if BASE_MODEL in result.stdout:
        print(f"✅ {BASE_MODEL} is already available")
    else:
        print(f"⚠️ {BASE_MODEL} not found, pulling it now...")
        if not run_command(["ollama", "pull", BASE_MODEL], f"Pulling {BASE_MODEL}"):
            print(f"❌ Failed to pull {BASE_MODEL}, exiting")
            sys.exit(1)
    
    # Step 2: Create the Modelfile
    if not create_modelfile():
        print(f"❌ Failed to create Modelfile, exiting")
        sys.exit(1)
    
    # Step 3: Create the model in Ollama
    print(f"\n🔍 Checking if {MODEL_NAME} already exists...")
    if MODEL_NAME in result.stdout:
        print(f"⚠️ {MODEL_NAME} already exists, removing it first...")
        if not run_command(["ollama", "rm", MODEL_NAME], f"Removing existing {MODEL_NAME}"):
            print(f"⚠️ Failed to remove existing model, will try to continue anyway")
    
    if not run_command(["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH], f"Creating {MODEL_NAME}"):
        print(f"❌ Failed to create {MODEL_NAME}, exiting")
        sys.exit(1)
    
    # Step 4: Update the .env file
    if not update_env_file(MODEL_NAME):
        print(f"❌ Failed to update .env file, exiting")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print(f"✅ Successfully set up {MODEL_NAME} for PathRAG")
    print("=" * 80)
    print("\nTo test the updated configuration, run:")
    print(f"cd {PATHRAG_PATH} && source venv/bin/activate && python src/pathrag_runner.py --query \"What are the key principles of RAG systems?\" --session-id \"test_qwen_32k\"")

if __name__ == "__main__":
    main()
