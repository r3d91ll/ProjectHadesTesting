#!/usr/bin/env python3
"""
Fix PathRAG Project Name

This script directly modifies the PathRAG adapter to use the PHOENIX_PROJECT_NAME
environment variable for creating a separate project in Phoenix.
"""

import os
import sys
from pathlib import Path

# Find the PathRAG directory
PATHRAG_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "pathrag"
ADAPTER_FILE = PATHRAG_DIR / "src" / "pathrag_runner.py"

if not ADAPTER_FILE.exists():
    print(f"Error: PathRAG runner file not found at {ADAPTER_FILE}")
    sys.exit(1)

# Read the file
with open(ADAPTER_FILE, 'r') as f:
    content = f.read()

# Check if we need to modify the file
if "PHOENIX_PROJECT_NAME" in content:
    print("Project name already configured in PathRAG runner")
else:
    # Find where the Phoenix configuration is loaded
    if "phoenix_host" in content and "phoenix_port" in content:
        # Add project name to the Phoenix configuration
        modified_content = content.replace(
            'phoenix_host = os.environ.get("PHOENIX_HOST", "localhost")',
            'phoenix_host = os.environ.get("PHOENIX_HOST", "localhost")\n    phoenix_project = os.environ.get("PHOENIX_PROJECT_NAME", "pathrag-inference")\n    print(f"Using Phoenix project: {phoenix_project}")'
        )
        
        # Add project name to the Phoenix session initialization
        modified_content = modified_content.replace(
            'phoenix_host=phoenix_host, phoenix_port=phoenix_port',
            'phoenix_host=phoenix_host, phoenix_port=phoenix_port, project_name=phoenix_project'
        )
        
        # Write the modified file
        with open(ADAPTER_FILE, 'w') as f:
            f.write(modified_content)
        
        print(f"✅ Added project name to PathRAG runner at {ADAPTER_FILE}")
    else:
        print("Could not find Phoenix configuration in PathRAG runner")

# Now check the adapter file
ADAPTER_FILE = PATHRAG_DIR / "src" / "pathrag_arize_adapter.py"
if ADAPTER_FILE.exists():
    with open(ADAPTER_FILE, 'r') as f:
        content = f.read()
    
    if "PHOENIX_PROJECT_NAME" in content or "project_name" in content:
        print("Project name already configured in PathRAG Arize adapter")
    else:
        # Add project name to the adapter
        modified_content = content.replace(
            'def __init__(self, config):',
            'def __init__(self, config):\n        self.project_name = config.get("project_name", os.environ.get("PHOENIX_PROJECT_NAME", "pathrag-inference"))\n        print(f"Using Phoenix project: {self.project_name}")'
        )
        
        # Add project name to the Phoenix session initialization
        if 'Session(' in modified_content:
            modified_content = modified_content.replace(
                'Session(',
                'Session(project_name=self.project_name, '
            )
            
            # Write the modified file
            with open(ADAPTER_FILE, 'w') as f:
                f.write(modified_content)
            
            print(f"✅ Added project name to PathRAG Arize adapter at {ADAPTER_FILE}")
        else:
            print("Could not find Phoenix Session initialization in PathRAG Arize adapter")
else:
    print(f"PathRAG Arize adapter not found at {ADAPTER_FILE}")

print("\nVerifying .env file...")
ENV_FILE = PATHRAG_DIR / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        content = f.read()
    
    if "PHOENIX_PROJECT_NAME" in content:
        print("Project name already configured in .env file")
    else:
        # Add project name to .env file
        with open(ENV_FILE, 'a') as f:
            f.write("\n# Phoenix Project Name\nPHOENIX_PROJECT_NAME=pathrag-inference\n")
        
        print(f"✅ Added PHOENIX_PROJECT_NAME to .env file at {ENV_FILE}")
else:
    print(f".env file not found at {ENV_FILE}")

print("\nDone! PathRAG should now use the 'pathrag-inference' project in Phoenix.")
