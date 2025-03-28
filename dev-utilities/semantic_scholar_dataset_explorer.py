#!/usr/bin/env python3
"""
Semantic Scholar Dataset Explorer

This utility downloads and processes the Semantic Scholar Papers dataset
to find papers matching specific search terms and domains.

Usage:
    python semantic_scholar_dataset_explorer.py --search-terms "anthropology of value" "science and technology studies"
"""

import os
import sys
import gzip
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime
import urllib.request
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("s2_dataset_explorer")

# Constants
DEFAULT_DATASET_DIR = "semantic_scholar_data"
DEFAULT_RELEASE_DATE = "2023-05-03"  # Use the latest available release date
S3_BASE_URL = f"https://ai2-s2-research-public.s3.amazonaws.com/open-corpus"

def download_file(url: str, output_path: str, desc: str = None) -> bool:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: The URL to download from
        output_path: Path to save the file
        desc: Description for the progress bar
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if desc is None:
            desc = f"Downloading {os.path.basename(output_path)}"
        
        # Check if file already exists
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                logger.info(f"File already exists: {output_path} ({file_size} bytes)")
                return True
                
        # Use wget for better handling of large files and resume capability
        command = [
            "wget", 
            "-O", output_path,
            "--continue",  # Resume partial downloads
            "--show-progress",
            url
        ]
        
        logger.info(f"Downloading {url} to {output_path}")
        result = subprocess.run(command, check=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully downloaded to {output_path}")
            return True
        else:
            logger.error(f"Failed to download {url}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading file {url}: {e}")
        return False

def download_dataset(dataset_type: str, output_dir: str, release_date: str = DEFAULT_RELEASE_DATE) -> str:
    """
    Download a Semantic Scholar dataset.
    
    Args:
        dataset_type: Type of dataset ('papers', 'citations', 'abstracts')
        output_dir: Directory to save the dataset
        release_date: Release date of the dataset
        
    Returns:
        Path to the downloaded file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the URL
    url = f"{S3_BASE_URL}/{release_date}/{dataset_type}.json.gz"
    output_path = os.path.join(output_dir, f"{dataset_type}_{release_date}.json.gz")
    
    success = download_file(url, output_path)
    if not success:
        logger.error(f"Failed to download {dataset_type} dataset")
        return None
        
    return output_path

def sample_dataset(dataset_path: str, sample_size: int = 10) -> List[Dict]:
    """
    Read a sample of records from a gzipped JSON Lines file.
    
    Args:
        dataset_path: Path to the gzipped dataset file
        sample_size: Number of records to sample
        
    Returns:
        List of sampled records
    """
    samples = []
    try:
        with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                record = json.loads(line.strip())
                samples.append(record)
    except Exception as e:
        logger.error(f"Error sampling dataset {dataset_path}: {e}")
    
    return samples

def search_dataset(dataset_path: str, search_terms: List[str], max_matches: int = 100, match_fields: List[str] = None) -> List[Dict]:
    """
    Search for records in the dataset matching any of the search terms.
    
    Args:
        dataset_path: Path to the gzipped dataset file
        search_terms: List of search terms to match
        max_matches: Maximum number of matches to return
        match_fields: Fields to search in (default: title, abstract, keywords)
        
    Returns:
        List of matching records
    """
    if match_fields is None:
        match_fields = ['title', 'abstract', 'keywords']
    
    # Convert search terms to lowercase for case-insensitive matching
    search_terms_lower = [term.lower() for term in search_terms]
    
    matches = []
    processed = 0
    
    try:
        with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Searching dataset", unit="records")):
                if len(matches) >= max_matches:
                    break
                    
                if i % 10000 == 0:
                    logger.info(f"Processed {i} records, found {len(matches)} matches")
                
                processed += 1
                try:
                    record = json.loads(line.strip())
                    
                    # Check if any search term is in any of the fields
                    for field in match_fields:
                        if field in record and record[field]:
                            field_value = record[field]
                            
                            # Handle different field types
                            if isinstance(field_value, list):
                                field_value = ' '.join(str(item) for item in field_value)
                            elif not isinstance(field_value, str):
                                field_value = str(field_value)
                                
                            field_value_lower = field_value.lower()
                            
                            if any(term in field_value_lower for term in search_terms_lower):
                                matches.append(record)
                                break
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        logger.error(f"Error searching dataset {dataset_path}: {e}")
    
    logger.info(f"Processed {processed} records, found {len(matches)} matches")
    return matches

def save_matches(matches: List[Dict], output_path: str):
    """
    Save matching records to a JSON file.
    
    Args:
        matches: List of matching records
        output_path: Path to save the matches
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(matches, f, indent=2)
        logger.info(f"Saved {len(matches)} matches to {output_path}")
    except Exception as e:
        logger.error(f"Error saving matches to {output_path}: {e}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Semantic Scholar Dataset Explorer')
    parser.add_argument('--output-dir', default=DEFAULT_DATASET_DIR,
                      help=f'Directory to store datasets (default: {DEFAULT_DATASET_DIR})')
    parser.add_argument('--release-date', default=DEFAULT_RELEASE_DATE,
                      help=f'Release date for the dataset (default: {DEFAULT_RELEASE_DATE})')
    parser.add_argument('--dataset-type', default='papers',
                      choices=['papers', 'citations', 'abstracts'],
                      help='Type of dataset to download (default: papers)')
    parser.add_argument('--search-terms', nargs='+', required=False,
                      help='Search terms to look for in the dataset')
    parser.add_argument('--max-matches', type=int, default=100,
                      help='Maximum number of matches to return (default: 100)')
    parser.add_argument('--sample-only', action='store_true',
                      help='Only sample the dataset, do not search')
    parser.add_argument('--sample-size', type=int, default=10,
                      help='Number of records to sample (default: 10)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download dataset
    dataset_path = download_dataset(
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        release_date=args.release_date
    )
    
    if dataset_path and os.path.exists(dataset_path):
        logger.info(f"Dataset downloaded to {dataset_path}")
        
        # Sample the dataset
        if args.sample_only or not args.search_terms:
            logger.info(f"Sampling {args.sample_size} records from the dataset")
            samples = sample_dataset(dataset_path, args.sample_size)
            
            sample_path = os.path.join(args.output_dir, f"{args.dataset_type}_sample.json")
            save_matches(samples, sample_path)
            
            logger.info(f"Sample structure:\n{json.dumps(samples[0], indent=2)}")
        
        # Search the dataset
        if args.search_terms:
            logger.info(f"Searching for terms: {args.search_terms}")
            matches = search_dataset(
                dataset_path=dataset_path,
                search_terms=args.search_terms,
                max_matches=args.max_matches
            )
            
            # Save matches
            if matches:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                matches_path = os.path.join(
                    args.output_dir, 
                    f"{args.dataset_type}_matches_{timestamp}.json"
                )
                save_matches(matches, matches_path)
                
                logger.info(f"Found {len(matches)} matches")
                logger.info(f"First match:\n{json.dumps(matches[0], indent=2)}")
            else:
                logger.info("No matches found")
    else:
        logger.error("Failed to download dataset")

if __name__ == "__main__":
    main()
