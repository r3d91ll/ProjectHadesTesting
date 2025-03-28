"""
Semantic Scholar Collector

Collects papers from the Semantic Scholar API.
"""
import os
import logging
import time
import json
import requests
from pathlib import Path
import urllib.parse

from .base_academic_collector import BaseAcademicCollector

logger = logging.getLogger(__name__)

class SemanticScholarCollector(BaseAcademicCollector):
    """
    Collector for academic papers from Semantic Scholar.
    
    Uses the Semantic Scholar API to search for and download papers.
    """
    
    def __init__(self, output_dir, deduplicator=None, api_key=None):
        """
        Initialize the Semantic Scholar collector.
        
        Args:
            output_dir (str): Directory to save downloaded papers
            deduplicator: Optional deduplicator to avoid duplicate papers
            api_key (str, optional): API key for Semantic Scholar (not strictly required but recommended)
        """
        super().__init__(output_dir, deduplicator)
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.source = "semantic_scholar"
    
    def search_papers(self, query, max_results=50):
        """
        Search for papers on Semantic Scholar matching a query.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of paper metadata
        """
        # Prepare headers
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        # Define fields we want to retrieve
        fields = [
            "paperId", "title", "abstract", "year", "authors", "venue", "url", 
            "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy", "publicationTypes"
        ]
        fields_param = ",".join(fields)
        
        # Build the API URL
        encoded_query = urllib.parse.quote(query)
        url = f"{self.base_url}/paper/search?query={encoded_query}&limit={max_results}&fields={fields_param}"
        
        try:
            # Implement rate limiting with exponential backoff
            max_retries = 3
            retry_count = 0
            backoff_time = 2  # Start with 2 second wait (Semantic Scholar rate limit is 100 requests per 5 minutes)
            
            while retry_count <= max_retries:
                try:
                    # Make the request with rate limiting
                    if retry_count > 0:
                        logger.info(f"Retry attempt {retry_count}/{max_retries} for query: {query}")
                    
                    # Make the request
                    response = requests.get(url, headers=headers)
                    
                    # Handle different response codes
                    if response.status_code == 200:
                        data = response.json()
                        break  # Success, exit the retry loop
                    elif response.status_code == 429:  # Rate limit exceeded
                        retry_count += 1
                        wait_time = backoff_time * (2 ** (retry_count - 1))  # Exponential backoff
                        if retry_count <= max_retries:
                            logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry.")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Rate limit exceeded after {max_retries} retries. Aborting.")
                            return []
                    else:
                        # Other error occurred
                        response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = backoff_time * (2 ** (retry_count - 1))
                        logger.warning(f"Request error: {str(e)}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        return []
            
            # If we get here without data, return empty list
            if 'data' not in locals():
                return []
            
            # Process the results
            papers = []
            for paper in data.get("data", []):
                # Extract authors
                authors = [author.get("name", "") for author in paper.get("authors", [])]
                
                # Create paper metadata
                paper_metadata = {
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors": authors,
                    "year": paper.get("year"),
                    "venue": paper.get("venue", ""),
                    "url": paper.get("url", ""),
                    "is_open_access": paper.get("isOpenAccess", False),
                    "pdf_url": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
                    "fields_of_study": paper.get("fieldsOfStudy", []),
                    "publication_types": paper.get("publicationTypes", []),
                    "source": self.source,
                    "source_id": paper.get("paperId", ""),
                    "metadata_retrieved_at": time.time()
                }
                
                # Only add papers that have either abstract or PDF
                if paper_metadata["abstract"] or paper_metadata["pdf_url"]:
                    papers.append(paper_metadata)
            
            logger.info(f"Found {len(papers)} papers on Semantic Scholar for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def download_paper(self, paper_metadata, save_dir):
        """
        Download a paper from Semantic Scholar.
        
        For Semantic Scholar, we either save the PDF if available or create a JSON
        with metadata including the abstract.
        
        Args:
            paper_metadata (dict): Paper metadata
            save_dir (str): Directory to save the paper
            
        Returns:
            str or None: Path to downloaded file if successful, None otherwise
        """
        try:
            # Create sanitized filename from title
            title = paper_metadata.get("title", "Untitled")
            sanitized_title = self._sanitize_filename(title)
            
            if paper_metadata.get("pdf_url"):
                # Download PDF if available
                pdf_url = paper_metadata["pdf_url"]
                pdf_path = os.path.join(save_dir, f"{sanitized_title}.pdf")
                
                # Download the PDF
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()
                
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Also save metadata
                metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(paper_metadata, f, indent=2)
                
                logger.info(f"Downloaded PDF for paper: {title}")
                return pdf_path
                
            else:
                # Save abstract and metadata as JSON
                metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(paper_metadata, f, indent=2)
                
                # Also save abstract as text file if available
                if paper_metadata.get("abstract"):
                    abstract_path = os.path.join(save_dir, f"{sanitized_title}.txt")
                    with open(abstract_path, 'w') as f:
                        f.write(f"Title: {title}\n\n")
                        f.write(f"Authors: {', '.join(paper_metadata.get('authors', []))}\n\n")
                        f.write(f"Abstract: {paper_metadata['abstract']}")
                    
                    logger.info(f"Saved abstract for paper: {title}")
                    return abstract_path
                
                logger.info(f"Saved metadata for paper: {title}")
                return metadata_path
                
        except Exception as e:
            logger.error(f"Error downloading paper from Semantic Scholar: {e}")
            return None
    
    def get_paper_by_id(self, paper_id):
        """
        Get a paper by its Semantic Scholar ID.
        
        Args:
            paper_id (str): Semantic Scholar paper ID
            
        Returns:
            dict or None: Paper metadata if found, None otherwise
        """
        # Prepare headers
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        # Define fields we want to retrieve
        fields = [
            "paperId", "title", "abstract", "year", "authors", "venue", "url", 
            "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy", "publicationTypes"
        ]
        fields_param = ",".join(fields)
        
        try:
            # Make the request
            url = f"{self.base_url}/paper/{paper_id}?fields={fields_param}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            paper = response.json()
            
            # Extract authors
            authors = [author.get("name", "") for author in paper.get("authors", [])]
            
            # Create paper metadata
            paper_metadata = {
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": authors,
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "url": paper.get("url", ""),
                "is_open_access": paper.get("isOpenAccess", False),
                "pdf_url": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
                "fields_of_study": paper.get("fieldsOfStudy", []),
                "publication_types": paper.get("publicationTypes", []),
                "source": self.source,
                "source_id": paper.get("paperId", ""),
                "metadata_retrieved_at": time.time()
            }
            
            return paper_metadata
            
        except Exception as e:
            logger.error(f"Error getting paper by ID from Semantic Scholar: {e}")
            return None
