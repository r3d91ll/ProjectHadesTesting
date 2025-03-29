#!/usr/bin/env python3
"""
Script to fix Phoenix configuration settings across the codebase.
This script updates PathRAG configuration to use the correct Phoenix host and port.
"""

import os
import sys
import argparse
import dotenv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Fix Phoenix configuration settings")
    parser.add_argument("--host", default="localhost", help="Phoenix host (default: localhost)")
    parser.add_argument("--port", default="8084", help="Phoenix port (default: 8084)")
    args = parser.parse_args()

    # Define the project root
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Update PathRAG .env file if it exists
    pathrag_env_file = project_root / "pathrag" / ".env"
    if pathrag_env_file.exists():
        # Use python-dotenv to load and modify the .env file
        dotenv.load_dotenv(pathrag_env_file)
        os.environ["PHOENIX_HOST"] = args.host
        os.environ["PHOENIX_PORT"] = args.port
        
        # Write the updated environment variables back to the .env file
        with open(pathrag_env_file, "w") as f:
            f.write(f"PHOENIX_HOST={args.host}\n")
            f.write(f"PHOENIX_PORT={args.port}\n")
            
            # Preserve any other variables that might be in the file
            for key, value in os.environ.items():
                if key not in ["PHOENIX_HOST", "PHOENIX_PORT"] and key.startswith("PHOENIX_") or key.startswith("OPENAI_") or key.startswith("PATHRAG_"):
                    f.write(f"{key}={value}\n")
        
        print(f"✅ Updated PathRAG .env file with Phoenix host={args.host} and port={args.port}")
    else:
        print(f"⚠️ PathRAG .env file not found. Creating a new one.")
        # Create a new .env file with the correct settings
        with open(pathrag_env_file, "w") as f:
            f.write(f"PHOENIX_HOST={args.host}\n")
            f.write(f"PHOENIX_PORT={args.port}\n")
        print(f"✅ Created new PathRAG .env file with Phoenix host={args.host} and port={args.port}")
    
    # Also create/update a .env file in the project root if it doesn't exist
    root_env_file = project_root / ".env"
    if not root_env_file.exists():
        with open(root_env_file, "w") as f:
            f.write(f"PHOENIX_HOST={args.host}\n")
            f.write(f"PHOENIX_PORT={args.port}\n")
        print(f"✅ Created new root .env file with Phoenix host={args.host} and port={args.port}")
    
    print("\nPhoenix configuration has been updated to use:")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print("\nYou can now run PathRAG directly without using the temporary script.")
    print("Example: python pathrag/src/pathrag_runner.py --query \"What is PathRAG?\" --session-id \"test-session\" --user-id \"test-user\"")

if __name__ == "__main__":
    main()
