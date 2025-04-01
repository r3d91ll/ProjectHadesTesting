#!/usr/bin/env python3
"""
PathRAG Phoenix Adapter Patch

This script patches the PathRAG Arize Phoenix adapter to use the PHOENIX_PROJECT_NAME
environment variable for creating a separate project in Phoenix.
"""

import os
import sys
import re
from pathlib import Path

# Find the PathRAG adapter file
PATHRAG_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "pathrag"
ADAPTER_FILE = PATHRAG_DIR / "src" / "pathrag_db_arize_adapter.py"

if not ADAPTER_FILE.exists():
    print(f"Error: Adapter file not found at {ADAPTER_FILE}")
    sys.exit(1)

# Read the adapter file
with open(ADAPTER_FILE, 'r') as f:
    content = f.read()

# Check if the file already has project name support
if "project_name" in content and "PHOENIX_PROJECT_NAME" in content:
    print("Adapter already has project name support")
    sys.exit(0)

# Patch 1: Add project_name to the __init__ method
init_pattern = r"def __init__\(self, config: Dict\[str, Any\]\):(.*?)self\.track_performance = config\.get\(\"track_performance\", True\)"
init_replacement = r"def __init__(self, config: Dict[str, Any]):\1self.track_performance = config.get(\"track_performance\", True)\n        self.project_name = config.get(\"project_name\", os.environ.get(\"PHOENIX_PROJECT_NAME\", \"pathrag\"))\n        logger.info(f\"Using Phoenix project name: {self.project_name}\")"

# Patch 2: Add project_name to the Session initialization
session_pattern = r"self\.phoenix_session = Session\(url=self\.phoenix_url\)"
session_replacement = r"self.phoenix_session = Session(url=self.phoenix_url, project_name=self.project_name)\n        logger.info(f\"🔍 Using Phoenix project: {self.project_name}\")"

# Apply the patches
patched_content = re.sub(init_pattern, init_replacement, content, flags=re.DOTALL)
patched_content = re.sub(session_pattern, session_replacement, patched_content)

# Write the patched file
with open(ADAPTER_FILE, 'w') as f:
    f.write(patched_content)

print(f"✅ Successfully patched {ADAPTER_FILE} to use PHOENIX_PROJECT_NAME")
