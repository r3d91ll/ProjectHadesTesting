#!/usr/bin/env python3
"""
RAG Dataset Collector - Academic Papers and Resources

This module collects academic papers and resources focusing on anthropology of value,
Science and Technology Studies (STS), and technology architecture.
"""

import os
import sys
import json
import time
import logging
import argparse
import yaml
import requests
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import arxiv
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("academic_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("academic_collector")


class AcademicCollector:
    """
    Collector for academic papers and resources on specified topics.
    """
    
    def __init__(self, config_file: str, output_dir: str):
        """
        Initialize the academic collector.
        
        Args:
            config_file: Path to configuration file with search terms
            output_dir: Directory to save collected papers
        """
        self.config_file = config_file
        self.output_dir = output_dir
        self.config = self._load_config(config_file)
        
        # Create output directory structure
        self.papers_dir = os.path.join(output_dir, "papers")
        os.makedirs(self.papers_dir, exist_ok=True)
        
        # Create category subdirectories
        for category in self.config.get("search_terms", {}).keys():
            category_dir = os.path.join(self.papers_dir, category)
            os.makedirs(category_dir, exist_ok=True)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def collect_arxiv_papers(self, max_papers_per_category: int = 50):
        """
        Collect papers from arXiv based on search terms.
        
        Args:
            max_papers_per_category: Maximum number of papers to collect per category
        """
        logger.info("Collecting papers from arXiv")
        
        search_terms = self.config.get("search_terms", {})
        
        for category, terms in search_terms.items():
            category_dir = os.path.join(self.papers_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            logger.info(f"Collecting papers for category: {category}")
            
            for term in terms:
                # Clean up search term for logging
                term_clean = term.replace('"', '').replace(':', '')
                logger.info(f"Searching for: {term_clean}")
                
                try:
                    # Search arXiv
                    search = arxiv.Search(
                        query=term,
                        max_results=max_papers_per_category // len(terms),
                        sort_by=arxiv.SortCriterion.Relevance
                    )
                    
                    for paper in tqdm(search.results(), desc=f"Downloading {term_clean}"):
                        # Create sanitized filename
                        filename = self._sanitize_filename(paper.title)
                        
                        # Download PDF
                        pdf_path = os.path.join(category_dir, f"{filename}.pdf")
                        
                        # Skip if already downloaded
                        if os.path.exists(pdf_path):
                            logger.info(f"Paper already exists: {filename}")
                            continue
                        
                        # Download paper
                        logger.info(f"Downloading: {paper.title}")
                        try:
                            paper.download_pdf(pdf_path)
                            
                            # Create metadata file
                            metadata = {
                                "title": paper.title,
                                "authors": [author.name for author in paper.authors],
                                "abstract": paper.summary,
                                "categories": paper.categories,
                                "url": paper.pdf_url,
                                "published": paper.published.isoformat(),
                                "search_term": term,
                                "license": self._extract_license_info(paper),
                                "paper_id": paper.entry_id.split('/')[-1]
                            }
                            
                            metadata_path = os.path.join(category_dir, f"{filename}.json")
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            
                            # Add a small delay to avoid hitting rate limits
                            time.sleep(1)
                            
                        except Exception as e:
                            logger.error(f"Error downloading paper {paper.title}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error searching arXiv for {term}: {e}")
    
    def collect_from_open_access_repositories(self, max_papers_per_category: int = 50):
        """
        Collect papers from open access repositories like SSRN, JSTOR, etc.
        
        Args:
            max_papers_per_category: Maximum number of papers to collect per category
        """
        # Implementation would require specific APIs for each repository
        # This is a placeholder for future implementation
        logger.info("Collecting papers from open access repositories")
        logger.warning("This feature is not fully implemented yet")
    
    def collect_from_anthropology_value_resources(self):
        """
        Collect papers specifically on anthropology of value from dedicated resources.
        """
        # URLs for anthropology of value resources
        resources = [
            {
                "name": "Graeber Archive",
                "url": "https://davidgraeber.org/papers/",
                "selector": "div.paper a[href$='.pdf']"
            },
            {
                "name": "STS Virtual Library",
                "url": "https://stsvirtual.org/",
                "selector": "a[href$='.pdf']"
            },
            # Add more resources as needed
        ]
        
        category_dir = os.path.join(self.papers_dir, "anthropology_value")
        os.makedirs(category_dir, exist_ok=True)
        
        for resource in resources:
            logger.info(f"Collecting papers from {resource['name']}")
            
            try:
                response = requests.get(resource["url"], timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                pdf_links = soup.select(resource["selector"])
                
                for link in pdf_links:
                    pdf_url = link.get('href')
                    if not pdf_url.startswith('http'):
                        # Handle relative URLs
                        pdf_url = f"{resource['url'].rstrip('/')}/{pdf_url.lstrip('/')}"
                    
                    # Extract filename from URL
                    filename = self._sanitize_filename(os.path.basename(pdf_url))
                    
                    # Download PDF
                    pdf_path = os.path.join(category_dir, filename)
                    
                    # Skip if already downloaded
                    if os.path.exists(pdf_path):
                        logger.info(f"Paper already exists: {filename}")
                        continue
                    
                    logger.info(f"Downloading: {filename}")
                    try:
                        pdf_response = requests.get(pdf_url, timeout=30)
                        pdf_response.raise_for_status()
                        
                        with open(pdf_path, 'wb') as f:
                            f.write(pdf_response.content)
                        
                        # Create metadata file
                        metadata = {
                            "title": link.text.strip() if link.text.strip() else filename,
                            "source": resource["name"],
                            "url": pdf_url,
                            "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        metadata_path = os.path.join(category_dir, f"{os.path.splitext(filename)[0]}.json")
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        # Add a small delay to avoid hitting rate limits
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error downloading paper {filename}: {e}")
                
            except Exception as e:
                logger.error(f"Error collecting papers from {resource['name']}: {e}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Replace invalid characters with underscores
        sanitized = ''.join(c if c.isalnum() or c in '.-_ ' else '_' for c in filename)
        # Truncate if too long
        if len(sanitized) > 100:
            sanitized = sanitized[:95] + '...'
        return sanitized
    
    def run_collection(self):
        """Run the full collection process."""
        logger.info("Starting academic paper collection")
        
        # Collect papers from arXiv
        self.collect_arxiv_papers()
        
        # Collect from anthropology of value resources
        self.collect_from_anthropology_value_resources()
        
        # Collect from open access repositories (placeholder)
        # self.collect_from_open_access_repositories()
        
        logger.info("Paper collection complete")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect academic papers and resources")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to configuration file with search terms")
    parser.add_argument("--output-dir", type=str, default="../data",
                      help="Directory to save collected papers")
    parser.add_argument("--max-papers", type=int, default=50,
                      help="Maximum number of papers to collect per category")
    
    args = parser.parse_args()
    
    collector = AcademicCollector(
        config_file=args.config,
        output_dir=args.output_dir
    )
    collector.run_collection()


if __name__ == "__main__":
    main()
