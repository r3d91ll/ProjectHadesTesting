#!/usr/bin/env python3
"""
Fix PathRAG Configuration

This script adds the PHOENIX_PROJECT_NAME to the PathRAG configuration file
to ensure traces are logged to the correct project in Phoenix.
"""

import os
import sys
import re
from pathlib import Path

# Find the PathRAG directory
PATHRAG_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "pathrag"
CONFIG_FILE = PATHRAG_DIR / "config" / "pathrag_config.py"

if not CONFIG_FILE.exists():
    print(f"Error: PathRAG config file not found at {CONFIG_FILE}")
    sys.exit(1)

# Read the config file
with open(CONFIG_FILE, 'r') as f:
    content = f.read()

# Check if the project name is already in the config
if "PHOENIX_PROJECT_NAME" in content:
    print("Project name already configured in PathRAG config")
else:
    # Add the project name to the environment variables section
    env_vars_pattern = r"# Arize Phoenix settings\nPHOENIX_HOST = os\.getenv\('PHOENIX_HOST', 'localhost'\)\nPHOENIX_PORT = int\(os\.getenv\('PHOENIX_PORT', '8084'\)\)"
    env_vars_replacement = "# Arize Phoenix settings\nPHOENIX_HOST = os.getenv('PHOENIX_HOST', 'localhost')\nPHOENIX_PORT = int(os.getenv('PHOENIX_PORT', '8084'))\nPHOENIX_PROJECT_NAME = os.getenv('PHOENIX_PROJECT_NAME', 'pathrag-inference')"
    
    # Add the project name to the config dictionary
    config_dict_pattern = r"# Arize Phoenix settings for telemetry\n        \"phoenix_host\": PHOENIX_HOST,\n        \"phoenix_port\": PHOENIX_PORT,"
    config_dict_replacement = "# Arize Phoenix settings for telemetry\n        \"phoenix_host\": PHOENIX_HOST,\n        \"phoenix_port\": PHOENIX_PORT,\n        \"project_name\": PHOENIX_PROJECT_NAME,"
    
    # Apply the patches
    modified_content = re.sub(env_vars_pattern, env_vars_replacement, content)
    modified_content = re.sub(config_dict_pattern, config_dict_replacement, modified_content)
    
    # Write the modified file
    with open(CONFIG_FILE, 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Added PHOENIX_PROJECT_NAME to PathRAG config at {CONFIG_FILE}")

# Ensure the .env file has the project name
ENV_FILE = PATHRAG_DIR / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        env_content = f.read()
    
    if "PHOENIX_PROJECT_NAME" not in env_content:
        with open(ENV_FILE, 'a') as f:
            f.write("\n# Phoenix Project Name\nPHOENIX_PROJECT_NAME=pathrag-inference\n")
        print(f"✅ Added PHOENIX_PROJECT_NAME to .env file at {ENV_FILE}")
    else:
        print("Project name already configured in .env file")
else:
    print(f"Warning: .env file not found at {ENV_FILE}")

print("\nDone! PathRAG should now log traces to the 'pathrag-inference' project in Phoenix.")
