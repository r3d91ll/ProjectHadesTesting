#!/usr/bin/env python3
"""
Temporary utility script to check the Phoenix project name configuration.
"""

import os
import sys
import json
from pathlib import Path

# Add the pathrag directory to the Python path
pathrag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "pathrag")
sys.path.insert(0, pathrag_path)

# Import PathRAG configuration
from config.pathrag_config import get_config, PHOENIX_PROJECT_NAME

def check_phoenix_project():
    """Check the Phoenix project name configuration."""
    print("\n🔍 Checking Phoenix project name configuration...\n")
    
    # Check environment variables
    print("📊 Environment Variables:")
    print(f"  PHOENIX_PROJECT_NAME = {os.environ.get('PHOENIX_PROJECT_NAME', 'Not set')}")
    
    # Check configuration
    config = get_config()
    print("\n📊 PathRAG Configuration:")
    print(f"  PHOENIX_PROJECT_NAME (module) = {PHOENIX_PROJECT_NAME}")
    print(f"  project_name (config) = {config.get('project_name', 'Not set')}")
    
    # Check if there are any hardcoded values in the adapter
    print("\n📊 Checking for hardcoded values in the adapter...")
    adapter_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "implementations/pathrag/arize_integration/adapter.py")
    
    if os.path.exists(adapter_path):
        with open(adapter_path, 'r') as f:
            content = f.read()
            
        # Look for project_name assignments
        import re
        project_assignments = re.findall(r'self\.project_name\s*=\s*[^\\n]*', content)
        
        if project_assignments:
            print("  Found project name assignments in adapter:")
            for assignment in project_assignments:
                print(f"    {assignment}")
        else:
            print("  No explicit project name assignments found in adapter.")
    else:
        print(f"  Adapter file not found at {adapter_path}")
    
    print("\n✅ Phoenix project name check complete")

if __name__ == "__main__":
    check_phoenix_project()
