#!/usr/bin/env python3
"""
Fix RAG Dataset Collector Configuration

This script directly calls the academic collector with the correct configuration,
bypassing the issue with search term detection in the main application.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add the rag-dataset-builder src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "rag-dataset-builder" / "src"))

# Import the academic collector
from collectors.academic_collector import AcademicCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_rag_dataset_collector")

def main():
    """Main function to fix the RAG dataset collector."""
    # Path to the configuration file
    config_file = Path(__file__).parent.parent / "rag-dataset-builder" / "config" / "config.yaml"
    
    # Path to the source documents directory
    source_docs_dir = Path(__file__).parent.parent / "source_documents"
    
    # Load the configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract the computer science search terms
    computer_science_terms = config.get("collection", {}).get("domains", {}).get("computer_science", {}).get("search_terms", [])
    
    # If not found in collection.domains, try direct domains path
    if not computer_science_terms:
        computer_science_terms = config.get("domains", {}).get("computer_science", {}).get("search_terms", [])
    
    if not computer_science_terms:
        logger.error("Could not find computer science search terms in the configuration file")
        sys.exit(1)
    
    logger.info(f"Found {len(computer_science_terms)} search terms for computer science")
    
    # Create a simplified configuration with just the computer science terms
    simplified_config = {
        "domains": {
            "computer_science": {
                "enabled": True,
                "search_terms": computer_science_terms
            }
        }
    }
    
    # Write the simplified configuration to a temporary file
    temp_config_file = Path(__file__).parent / "temp_academic_config.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(simplified_config, f)
    
    logger.info(f"Created temporary configuration file at {temp_config_file}")
    
    # Create an academic collector with the simplified configuration
    collector = AcademicCollector(str(temp_config_file), str(source_docs_dir))
    
    # Get the papers per term from the original config
    max_papers_per_term = config.get("collection", {}).get("max_papers_per_term", 1000)
    
    # Collect papers from arXiv
    logger.info(f"Collecting papers with max_papers_per_term={max_papers_per_term}")
    collector.collect_arxiv_papers(papers_per_term=max_papers_per_term)
    
    logger.info("Paper collection complete")

if __name__ == "__main__":
    main()
