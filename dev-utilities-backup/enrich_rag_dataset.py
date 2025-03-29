#!/usr/bin/env python3
"""
RAG Dataset Enrichment Utility

This utility script extends the RAG dataset by downloading additional papers from ArXiv
using the configured search terms, avoiding duplicates by paper ID rather than filename.

As a one-time utility, this script is placed in dev-utilities/ as per project conventions.
"""

import os
import sys
import json
import time
import logging
import yaml
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
import arxiv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("rag_dataset_enricher")

def sanitize_filename(title: str) -> str:
    """
    Create a clean, filesystem-friendly filename from a paper title.
    
    Args:
        title: Paper title
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove special characters, replace spaces with underscores
    filename = title.replace('/', '_').replace('\\', '_')[:100]
    filename = re.sub(r'[^\w\s-]', '', filename).strip().replace(' ', '_')
    return filename

def collect_arxiv_papers(config_file: str, output_dir: str, 
                        papers_per_search_term: int = 3,
                        max_papers_total: int = 50):
    """
    Collect papers from ArXiv avoiding duplicates by paper ID.
    
    Args:
        config_file: Path to configuration file
        output_dir: Directory to save papers
        papers_per_search_term: Number of papers to collect for each search term
        max_papers_total: Maximum total papers to download
    """
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_file}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Get search terms from config
    search_terms = config.get("search_terms", {})
    if not search_terms:
        search_terms = config.get("academic", {}).get("search_terms", {})
    
    if not search_terms:
        logger.error("No search terms found in config file")
        sys.exit(1)
    
    # Create output directory structure
    papers_dir = os.path.join(output_dir, "papers")
    os.makedirs(papers_dir, exist_ok=True)
    
    # Track downloaded paper IDs to avoid duplicates
    downloaded_paper_ids = set()
    
    # Find existing paper IDs
    logger.info("Scanning for existing papers...")
    for root, _, files in os.walk(papers_dir):
        for file in files:
            if file.endswith('.json'):  # Check metadata files
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        metadata = json.load(f)
                        if 'paper_id' in metadata:
                            downloaded_paper_ids.add(metadata['paper_id'])
                except Exception as e:
                    logger.warning(f"Error reading metadata file {file}: {e}")
    
    logger.info(f"Found {len(downloaded_paper_ids)} existing papers")
    
    # Track new downloads
    new_papers_downloaded = 0
    
    # Download papers for each category and search term
    for category, terms in search_terms.items():
        if not terms or not isinstance(terms, list):
            continue
            
        logger.info(f"Collecting papers for category: {category}")
        
        # Create category directory
        category_dir = os.path.join(papers_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Collect papers for each search term
        for term in terms:
            # Clean up search term for logging
            term_clean = term.replace('"', '').replace(':', '')
            logger.info(f"Searching for: {term_clean}")
            
            try:
                # Search ArXiv with more results to increase chance of finding new papers
                search = arxiv.Search(
                    query=term,
                    max_results=papers_per_search_term * 5,  # Get 5x to account for duplicates
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                papers_for_term = 0
                for paper in tqdm(search.results(), desc=f"Checking {term_clean}"):
                    # Stop if we've reached the limit for this term
                    if papers_for_term >= papers_per_search_term:
                        break
                        
                    # Stop if we've reached the overall limit
                    if new_papers_downloaded >= max_papers_total:
                        logger.info(f"Reached maximum papers limit ({max_papers_total})")
                        return
                    
                    # Get paper ID and check for duplicates
                    paper_id = paper.entry_id.split('/')[-1]
                    if paper_id in downloaded_paper_ids:
                        logger.debug(f"Skipping duplicate paper: {paper.title}")
                        continue
                    
                    # Create sanitized filename
                    filename = sanitize_filename(paper.title)
                    if paper_id:
                        filename = f"{filename}_{paper_id}"
                    
                    # Set output paths
                    pdf_path = os.path.join(category_dir, f"{filename}.pdf")
                    metadata_path = os.path.join(category_dir, f"{filename}.json")
                    
                    # Download paper
                    logger.info(f"Downloading: {paper.title}")
                    try:
                        # Create directory if needed
                        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                        
                        # Download PDF directly
                        response = requests.get(paper.pdf_url)
                        if response.status_code == 200:
                            with open(pdf_path, 'wb') as f:
                                f.write(response.content)
                            logger.info(f"Downloaded paper to {pdf_path}")
                        else:
                            logger.error(f"Failed to download PDF: {response.status_code}")
                            continue
                        
                        # Create metadata file
                        metadata = {
                            "title": paper.title,
                            "authors": [author.name for author in paper.authors],
                            "abstract": paper.summary,
                            "categories": paper.categories,
                            "url": paper.pdf_url,
                            "published": paper.published.isoformat(),
                            "search_term": term,
                            "license": paper.license if hasattr(paper, 'license') else None,
                            "paper_id": paper_id
                        }
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        # Track downloaded paper
                        downloaded_paper_ids.add(paper_id)
                        papers_for_term += 1
                        new_papers_downloaded += 1
                        
                        # Add a small delay to avoid hitting rate limits
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error downloading paper {paper.title}: {e}")
                
            except Exception as e:
                logger.error(f"Error searching ArXiv for {term}: {e}")
    
    logger.info(f"Successfully downloaded {new_papers_downloaded} new papers")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='RAG Dataset Enrichment Utility')
    parser.add_argument('--config', 
                      default='rag-dataset-builder/config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--output-dir', 
                      default='source_documents',
                      help='Directory to save papers')
    parser.add_argument('--papers-per-term', type=int, default=3,
                      help='Number of papers to download per search term')
    parser.add_argument('--max-papers', type=int, default=50,
                      help='Maximum total number of papers to download')
    
    args = parser.parse_args()
    
    # Collect papers from ArXiv
    collect_arxiv_papers(
        config_file=args.config,
        output_dir=args.output_dir,
        papers_per_search_term=args.papers_per_term,
        max_papers_total=args.max_papers
    )
    
    logger.info("Enrichment completed successfully")
    
    # Run the RAG dataset builder to process the new papers
    logger.info("You should now run the RAG dataset builder to process the new papers:")
    logger.info("python rag-dataset-builder/src/main.py --config rag-dataset-builder/config/config.yaml")

if __name__ == "__main__":
    main()
