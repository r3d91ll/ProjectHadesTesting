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
# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up logging to use the logs directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "academic_collector.log")),
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
        self.papers_dir = os.path.join(output_dir, "academic_papers")
        os.makedirs(self.papers_dir, exist_ok=True)
        
        # Log discovered categories and create directories
        if self.categories:
            logger.info(f"Found {len(self.categories)} categories with search terms")
            for category in self.categories.keys():
                category_dir = os.path.join(self.papers_dir, category)
                os.makedirs(category_dir, exist_ok=True)        
                logger.info(f"Created directory for category: {category}")
        else:
            logger.warning("No search terms found in configuration")
            
            # Check if we should use default search terms
            if not self.categories:
                # Only use hardcoded terms if explicitly enabled
                use_default_terms = False
                if use_default_terms:
                    logger.info("Using default search terms for testing")
                else:
                    logger.warning("No search terms found and default terms are disabled. Please check your config file.")
                    
                if use_default_terms:
                    self.categories = {
                        "anthropology_of_value": [
                            "anthropology of value",
                            "economic anthropology",
                            "gift economy"
                        ],
                        "science_technology_studies": [
                            "science and technology studies",
                            "actor network theory",
                            "technological determinism"
                        ]
                    }
                    # Create directories for hardcoded categories
                    for category in self.categories.keys():
                        category_dir = os.path.join(self.papers_dir, category)
                        os.makedirs(category_dir, exist_ok=True)
                    logger.warning("Using hardcoded default search terms")
                else:
                    logger.warning("No search terms found and default terms are disabled")
        
        # Create category subdirectories with standardized naming
        for category in self.categories.keys():
            # Convert category name to snake_case for consistency
            standardized_category = category.lower().replace(' ', '_').replace('-', '_')
            category_dir = os.path.join(self.papers_dir, standardized_category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Store the mapping of original category to standardized directory name
            if not hasattr(self, 'category_dir_mapping'):
                self.category_dir_mapping = {}
            self.category_dir_mapping[category] = standardized_category
            
        # Log the output directory structure and found categories
        logger.info(f"Papers will be saved to directory: {self.papers_dir}")
        logger.info(f"Found {len(self.categories)} categories with search terms")
    
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
            
            # Extract search terms early to support different config formats
            self._extract_search_terms(config)
            
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
            
    def _extract_search_terms(self, config: Dict[str, Any]):
        """
        Extract search terms from config in various possible formats.
        
        Args:
            config: Configuration dictionary
        """
        # Initialize categories dictionary
        self.categories = {}
        
        # Extract collection settings
        if "collection" in config:
            self.max_papers_per_term = config["collection"].get("max_papers_per_term", 0)
            self.max_documents_per_category = config["collection"].get("max_documents_per_category", 0)
            self.download_delay = config["collection"].get("download_delay", 1.0)
            self.max_download_size_mb = config["collection"].get("max_download_size_mb", 500)
            logger.info(f"Loaded collection settings: max_papers_per_term={self.max_papers_per_term}, max_documents_per_category={self.max_documents_per_category}")
        else:
            self.max_papers_per_term = 0
            self.max_documents_per_category = 0
            self.download_delay = 1.0
            self.max_download_size_mb = 500
            logger.warning("No collection settings found in config, using defaults")
        
        # Debug: print top-level keys
        logger.debug(f"Config keys: {sorted(list(config.keys()))}")
        
        # Try all possible locations for search terms
        
        # Check in the domains section at academic.domains
        if "academic" in config and isinstance(config["academic"], dict):
            academic = config["academic"]
            if "domains" in academic and isinstance(academic["domains"], dict):
                for domain, domain_config in academic["domains"].items():
                    if isinstance(domain_config, dict) and "search_terms" in domain_config and domain_config.get("enabled", False):
                        self.categories[domain] = domain_config["search_terms"]
                        logger.info(f"Found search terms for {domain} in academic.domains")
        
        # Check in the top-level domains section
        if not self.categories and "domains" in config and isinstance(config["domains"], dict):
            for domain, domain_config in config["domains"].items():
                if isinstance(domain_config, dict) and "search_terms" in domain_config and domain_config.get("enabled", False):
                    self.categories[domain] = domain_config["search_terms"]
                    logger.info(f"Found search terms for {domain} in domains")
        
        # Direct check for anthropology_of_value and other domains
        domain_names = ["anthropology_of_value", "science_technology_studies", "interdisciplinary", "ai_ethics", "computer_science"]
        for domain in domain_names:
            if domain in config and isinstance(config[domain], dict) and "search_terms" in config[domain] and config[domain].get("enabled", False):
                self.categories[domain] = config[domain]["search_terms"]
                logger.info(f"Found search terms for {domain} at top level")
        
        # Check under collection section
        if "collection" in config and isinstance(config["collection"], dict):
            collection = config["collection"]
            
            # Check for direct search_terms in collection
            if "search_terms" in collection:
                self.categories["collection"] = collection["search_terms"]
                logger.info("Found search terms directly in collection section")
            
            # Check for academic domains under collection
            for domain in domain_names:
                if domain in collection and isinstance(collection[domain], dict) and "search_terms" in collection[domain] and collection[domain].get("enabled", False):
                    self.categories[domain] = collection[domain]["search_terms"]
                    logger.info(f"Found search terms for {domain} in collection section")
        
        # Final fallback - search for any search_terms
        if not self.categories:
            def scan_for_terms(data, path=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        new_path = f"{path}.{key}" if path else key
                        
                        # Check if this dict has search_terms
                        if key == "search_terms" and isinstance(value, list) and value:
                            logger.info(f"Found search terms at {new_path}")
                            return {path.split('.')[-1] if path else "default": value}
                        
                        # Recursively scan nested dicts
                        result = scan_for_terms(value, new_path)
                        if result:
                            return result
                return None
            
            # Scan entire config recursively as last resort
            terms = scan_for_terms(config)
            if terms:
                self.categories.update(terms)
    
    def collect_arxiv_papers(self, max_papers_per_category: int = 50, papers_per_term: int = None, check_duplicates: bool = True):
        """
        Collect papers from arXiv based on search terms.
        
        Args:
            max_papers_per_category: Maximum number of papers to collect per category
            papers_per_term: Number of papers to collect per search term (overrides max_papers_per_category division)
            check_duplicates: Whether to check for and skip duplicate papers by paper ID
        """
        # This method assumes the caller has already checked if collection is enabled
        # It focuses solely on its specific job: collecting papers
            
        # Get collection parameters from config
        if "collection" in self.config and isinstance(self.config["collection"], dict):
            collection_config = self.config["collection"]
            
            # Override method parameters with config values if provided
            if "max_documents_per_category" in collection_config:
                config_max_papers = collection_config["max_documents_per_category"]
                # Ensure the value is valid (greater than 0)
                if isinstance(config_max_papers, (int, float)) and config_max_papers > 0:
                    max_papers_per_category = int(config_max_papers)
                    logger.info(f"Using max_documents_per_category from config: {max_papers_per_category}")
                else:
                    logger.warning(f"Invalid max_documents_per_category in config: {config_max_papers}. Using default: {max_papers_per_category}")
            
            if "max_papers_per_term" in collection_config and papers_per_term is None:
                config_papers_per_term = collection_config["max_papers_per_term"]
                # Ensure the value is valid (greater than 0)
                if isinstance(config_papers_per_term, (int, float)) and config_papers_per_term > 0:
                    papers_per_term = int(config_papers_per_term)
                    logger.info(f"Using max_papers_per_term from config: {papers_per_term}")
                else:
                    logger.warning(f"Invalid max_papers_per_term in config: {config_papers_per_term}. Will calculate based on max_documents_per_category.")
                
        logger.info("Collecting papers from arXiv")
        
        # Track downloaded paper IDs to avoid duplicates
        downloaded_paper_ids = set()
        
        # Check for existing paper IDs if duplicate checking is enabled
        if check_duplicates:
            logger.info("Scanning for existing papers by ID...")
            for root, _, files in os.walk(self.papers_dir):
                for file in files:
                    if file.endswith('.json'):  # Check metadata files
                        try:
                            with open(os.path.join(root, file), 'r') as f:
                                metadata = json.load(f)
                                if 'paper_id' in metadata:
                                    downloaded_paper_ids.add(metadata['paper_id'])
                        except Exception as e:
                            logger.warning(f"Error reading metadata file {file}: {e}")
            
            logger.info(f"Found {len(downloaded_paper_ids)} existing papers by ID")
        
        # Use the categories we extracted during initialization
        if not self.categories:
            logger.warning("No search terms found in the configuration. Please check your config file.")
            return
            
        for category, terms in self.categories.items():
            category_dir = os.path.join(self.papers_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            logger.info(f"Collecting papers for category: {category}")
            
            # Use class attribute if available, otherwise use the parameter
            if papers_per_term is None:
                if hasattr(self, 'max_papers_per_term') and self.max_papers_per_term > 0:
                    papers_per_term = self.max_papers_per_term
                    logger.info(f"Using max_papers_per_term from config: {papers_per_term}")
                else:
                    papers_per_term = max(1, max_papers_per_category // len(terms))
                
            logger.info(f"Will download up to {papers_per_term} papers per search term")
            
            for term in terms:
                # Clean up search term for logging
                term_clean = term.replace('"', '').replace(':', '')
                logger.info(f"Searching for: {term_clean}")
                
                try:
                    # Search arXiv with more results to increase chances of finding new papers
                    search = arxiv.Search(
                        query=term,
                        max_results=papers_per_term * 3,  # Get more results to account for duplicates
                        sort_by=arxiv.SortCriterion.Relevance
                    )
                    
                    papers_downloaded = 0
                    for paper in tqdm(search.results(), desc=f"Processing {term_clean}"):
                        # Check if we've downloaded enough papers for this term
                        if papers_downloaded >= papers_per_term:
                            break
                            
                        # Get paper ID and check for duplicates
                        paper_id = paper.entry_id.split('/')[-1]
                        
                        # Skip if we've already downloaded this paper and duplicate checking is enabled
                        if check_duplicates and paper_id in downloaded_paper_ids:
                            logger.info(f"Skipping duplicate paper (ID: {paper_id}): {paper.title}")
                            continue
                        
                        # Create sanitized filename with paper ID to ensure uniqueness
                        base_filename = self._sanitize_filename(paper.title)
                        filename = f"{base_filename}_{paper_id}"
                        
                        # Set paths for PDF and metadata using standardized category directory
                        standardized_category = self.category_dir_mapping.get(category, category.lower().replace(' ', '_').replace('-', '_'))
                        category_dir = os.path.join(self.papers_dir, standardized_category)
                        pdf_path = os.path.join(category_dir, f"{filename}.pdf")
                        
                        # Download paper
                        logger.info(f"Downloading: {paper.title}")
                        try:
                            # Create directory for PDF file if needed
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
                            
                            metadata_path = os.path.join(category_dir, f"{filename}.json")
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            
                            # Mark this paper as downloaded
                            downloaded_paper_ids.add(paper_id)
                            papers_downloaded += 1
                            
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
                "selector": "div.paper a[href$='.pdf']",
                "fallback_url": "https://archive.org/details/graeber-papers",
                "enabled": False  # Temporarily disabled until proper authorization or alternative source
            },
            {
                "name": "STS Virtual Library",
                "url": "https://stsvirtual.org/",
                "selector": "a[href$='.pdf']",
                "enabled": False  # Temporarily disabled until proper URL or alternative source
            },
            {
                "name": "STS Research",
                "url": "https://sts.hks.harvard.edu/research/papers/",
                "selector": "a[href$='.pdf']",
                "enabled": True
            },
            {
                "name": "Economic Anthropology Papers",
                "url": "https://anthrosource.onlinelibrary.wiley.com/journal/23304847",
                "selector": "a.issue-item__title",
                "enabled": True
            },
            # Add more resources as needed
        ]
        
        category_dir = os.path.join(self.papers_dir, "anthropology_value")
        os.makedirs(category_dir, exist_ok=True)
        
        for resource in resources:
            # Skip disabled resources
            if not resource.get("enabled", False):
                logger.info(f"Skipping disabled resource: {resource['name']}")
                continue
                
            logger.info(f"Collecting papers from {resource['name']}")
            
            try:
                # Add proper headers to mimic a browser request
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5"
                }
                
                # Try to fetch the resource with proper timeout and retries
                response = requests.get(
                    resource["url"], 
                    headers=headers, 
                    timeout=30,
                    allow_redirects=True
                )
                
                # Check if we got a successful response
                if response.status_code != 200:
                    logger.warning(f"Non-200 response ({response.status_code}) from {resource['name']}: {resource['url']}")
                    logger.warning(f"Response: {response.text[:500]}...") 
                    
                    # Try fallback URL if available
                    if resource.get("fallback_url"):
                        logger.info(f"Trying fallback URL for {resource['name']}: {resource.get('fallback_url')}")
                        response = requests.get(
                            resource["fallback_url"], 
                            headers=headers, 
                            timeout=30,
                            allow_redirects=True
                        )
                        
                        if response.status_code != 200:
                            logger.warning(f"Fallback also failed with status {response.status_code}")
                            continue
                    else:
                        continue
                
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                pdf_links = soup.select(resource["selector"])
                
                if not pdf_links:
                    logger.warning(f"No PDF links found at {resource['name']} using selector '{resource['selector']}'")
                    logger.debug(f"HTML preview: {response.text[:500]}...")
                    continue
                    
                logger.info(f"Found {len(pdf_links)} PDF links at {resource['name']}")
                
                # Process each PDF link
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
                    
                    logger.info(f"Downloading: {filename} from {pdf_url}")
                    try:
                        # Use the same headers as above for the PDF download
                        pdf_response = requests.get(
                            pdf_url, 
                            headers=headers, 
                            timeout=60,  # Longer timeout for PDF downloads
                            allow_redirects=True
                        )
                        
                        # Check response before writing to file
                        if pdf_response.status_code != 200:
                            logger.warning(f"Failed to download PDF: {pdf_url} (Status: {pdf_response.status_code})")
                            continue
                            
                        # Verify content type is PDF
                        content_type = pdf_response.headers.get('Content-Type', '')
                        if 'application/pdf' not in content_type and not pdf_url.endswith('.pdf'):
                            logger.warning(f"Resource not a PDF: {pdf_url} (Content-Type: {content_type})")
                            continue
                        
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
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error collecting papers from {resource['name']}: {e}")
            except Exception as e:
                logger.warning(f"Error collecting papers from {resource['name']}: {e}", exc_info=True)
    
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
    
    def run_collection(self, max_papers: int = 50, papers_per_term: int = None, check_duplicates: bool = True):
        """Run the full collection process.
        
        Args:
            max_papers: Maximum papers per category
            papers_per_term: Number of papers to collect per search term
            check_duplicates: Whether to check for and skip duplicate papers by paper ID
        """
        logger.info("Starting academic paper collection")
        
        # Collect papers from arXiv
        self.collect_arxiv_papers(
            max_papers_per_category=max_papers,
            papers_per_term=papers_per_term,
            check_duplicates=check_duplicates
        )
        
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
    parser.add_argument("--papers-per-term", type=int,
                      help="Number of papers to collect per search term")
    parser.add_argument("--force-download", action="store_true",
                      help="Force download papers even if they exist in the dataset")
    
    args = parser.parse_args()
    
    collector = AcademicCollector(
        config_file=args.config,
        output_dir=args.output_dir
    )
    collector.run_collection(
        max_papers=args.max_papers,
        papers_per_term=args.papers_per_term,
        check_duplicates=not args.force_download
    )


if __name__ == "__main__":
    main()
