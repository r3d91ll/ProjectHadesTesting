#!/usr/bin/env python3
"""
Direct Phoenix Project Fix

This script directly modifies the PathRAG configuration and adapter to ensure
traces are logged to a separate project in Phoenix.
"""

import os
import sys
import re
from pathlib import Path
import json
import requests

# Find the PathRAG directory
PATHRAG_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "pathrag"
CONFIG_FILE = PATHRAG_DIR / "config" / "pathrag_config.py"

# Phoenix settings
PHOENIX_HOST = os.environ.get("PHOENIX_HOST", "localhost")
PHOENIX_PORT = os.environ.get("PHOENIX_PORT", "8084")
PHOENIX_URL = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}"
PROJECT_NAME = "pathrag-inference"

print(f"🔍 Checking Phoenix connection at {PHOENIX_URL}...")
try:
    response = requests.get(f"{PHOENIX_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✅ Phoenix is running")
    else:
        print(f"⚠️ Phoenix health check failed: {response.status_code}")
except Exception as e:
    print(f"❌ Failed to connect to Phoenix: {e}")
    sys.exit(1)

# Step 1: Add project name to PathRAG config
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    
    if "PHOENIX_PROJECT_NAME" in content:
        print("✅ Project name already in PathRAG config")
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
        
        print(f"✅ Added PHOENIX_PROJECT_NAME to PathRAG config")
else:
    print(f"❌ PathRAG config file not found at {CONFIG_FILE}")
    sys.exit(1)

# Step 2: Ensure .env file has the project name
ENV_FILE = PATHRAG_DIR / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        env_content = f.read()
    
    if "PHOENIX_PROJECT_NAME" in env_content:
        print("✅ Project name already in .env file")
    else:
        with open(ENV_FILE, 'a') as f:
            f.write("\n# Phoenix Project Name\nPHOENIX_PROJECT_NAME=pathrag-inference\n")
        print(f"✅ Added PHOENIX_PROJECT_NAME to .env file")
else:
    print(f"⚠️ .env file not found, creating it...")
    with open(ENV_FILE, 'w') as f:
        f.write("# Phoenix Project Name\nPHOENIX_PROJECT_NAME=pathrag-inference\n")
    print(f"✅ Created .env file with PHOENIX_PROJECT_NAME")

# Step 3: Check for the adapter files
adapter_files = list(PATHRAG_DIR.glob("**/adapter.py")) + list(PATHRAG_DIR.glob("**/pathrag_arize_adapter.py"))
if adapter_files:
    for adapter_file in adapter_files:
        print(f"🔍 Checking adapter file: {adapter_file}")
        with open(adapter_file, 'r') as f:
            adapter_content = f.read()
        
        # Check if the adapter already handles project name
        if "project_name" in adapter_content and "PHOENIX_PROJECT_NAME" in adapter_content:
            print(f"✅ Adapter already handles project name")
        else:
            # Look for Session initialization
            if "Session(" in adapter_content:
                # Add project_name parameter to Session initialization
                modified_content = re.sub(
                    r"Session\(([^)]*)\)",
                    r"Session(\1, project_name=self.project_name)",
                    adapter_content
                )
                
                # Add project_name attribute to __init__ method
                if "__init__" in modified_content:
                    modified_content = re.sub(
                        r"def __init__\(self, ([^)]*)\):(.*?)self\.track_performance",
                        r"def __init__(self, \1):\2self.track_performance\n        self.project_name = config.get('project_name', os.environ.get('PHOENIX_PROJECT_NAME', 'pathrag-inference'))\n        print(f'Using Phoenix project: {self.project_name}')",
                        modified_content,
                        flags=re.DOTALL
                    )
                
                # Write the modified file
                with open(adapter_file, 'w') as f:
                    f.write(modified_content)
                
                print(f"✅ Updated adapter to use project name")
            else:
                print(f"⚠️ Could not find Session initialization in adapter")
else:
    print("⚠️ No adapter files found")

# Step 4: Create a test script that will force the project creation
TEST_SCRIPT = PATHRAG_DIR / "create_project.py"
with open(TEST_SCRIPT, 'w') as f:
    f.write("""#!/usr/bin/env python3
\"\"\"
Create Phoenix Project

This script creates a project in Phoenix to ensure it exists before running PathRAG.
\"\"\"

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
""")

print(f"✅ Created project creation script at {TEST_SCRIPT}")

# Step 5: Make the script executable and run it
os.chmod(TEST_SCRIPT, 0o755)
print("🔄 Running project creation script...")
os.chdir(PATHRAG_DIR)
os.system(f"python {TEST_SCRIPT}")

print("\n✅ Done! PathRAG should now log traces to the 'pathrag-inference' project in Phoenix.")
