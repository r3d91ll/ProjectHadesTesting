"""
SocArXiv Collector

Collects papers from SocArXiv using the Open Science Framework (OSF) API.
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

class SocArXivCollector(BaseAcademicCollector):
    """
    Collector for academic papers from SocArXiv.
    
    Uses the OSF API to search for and download papers from SocArXiv,
    which is a social science preprint server on the Open Science Framework.
    """
    
    def __init__(self, output_dir, deduplicator=None, token=None):
        """
        Initialize the SocArXiv collector.
        
        Args:
            output_dir (str): Directory to save downloaded papers
            deduplicator: Optional deduplicator to avoid duplicate papers
            token (str, optional): OSF API token for higher rate limits
        """
        super().__init__(output_dir, deduplicator)
        self.token = token
        self.base_url = "https://api.osf.io/v2"
        self.source = "socarxiv"
    
    def search_papers(self, query, max_results=50):
        """
        Search for papers on SocArXiv matching a query.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of paper metadata
        """
        # Prepare headers
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/json"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        # Build the query URL
        # SocArXiv's provider ID on OSF is 'socarxiv'
        encoded_query = urllib.parse.quote(query)
        url = f"{self.base_url}/search/preprints/?filter[provider]=socarxiv&q={encoded_query}&page[size]={max_results}"
        
        try:
            # Make the request
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Process the results
            papers = []
            for item in data.get("data", []):
                try:
                    # Get paper ID and attributes
                    paper_id = item.get("id")
                    attributes = item.get("attributes", {})
                    
                    # Get basic metadata
                    title = attributes.get("title", "")
                    description = attributes.get("description", "")
                    date_created = attributes.get("date_created")
                    
                    # Extract year from date
                    year = None
                    if date_created:
                        try:
                            year = int(date_created.split("-")[0])
                        except (ValueError, IndexError):
                            pass
                    
                    # Get download URL
                    download_url = None
                    links = item.get("links", {})
                    if links and "download" in links:
                        download_url = links["download"]
                    
                    # Get preprint URL
                    preprint_url = None
                    if links and "html" in links:
                        preprint_url = links["html"]
                    
                    # Get detailed information for contributors (authors)
                    authors = []
                    if paper_id:
                        try:
                            contributors_url = f"{self.base_url}/preprints/{paper_id}/contributors/"
                            contributors_response = requests.get(contributors_url, headers=headers)
                            contributors_response.raise_for_status()
                            contributors_data = contributors_response.json()
                            
                            for contributor in contributors_data.get("data", []):
                                embeds = contributor.get("embeds", {})
                                user = embeds.get("user", {})
                                user_attributes = user.get("data", {}).get("attributes", {})
                                
                                full_name = user_attributes.get("full_name")
                                if full_name:
                                    authors.append(full_name)
                        except Exception as e:
                            logger.warning(f"Error getting contributors for paper {paper_id}: {e}")
                    
                    # Get subjects/tags
                    subjects = []
                    for subject in attributes.get("subjects", []):
                        if isinstance(subject, str):
                            subjects.append(subject)
                        elif isinstance(subject, list) and len(subject) > 0:
                            subjects.append(subject[-1])
                    
                    # Create paper metadata
                    paper_metadata = {
                        "title": title,
                        "abstract": description,
                        "authors": authors,
                        "year": year,
                        "url": preprint_url,
                        "pdf_url": download_url,
                        "subjects": subjects,
                        "source": self.source,
                        "source_id": paper_id,
                        "metadata_retrieved_at": time.time()
                    }
                    
                    # Only add papers with title and either abstract or PDF url
                    if paper_metadata["title"] and (paper_metadata["abstract"] or paper_metadata["pdf_url"]):
                        papers.append(paper_metadata)
                        
                except Exception as e:
                    logger.error(f"Error processing SocArXiv paper: {e}")
            
            logger.info(f"Found {len(papers)} papers on SocArXiv for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching SocArXiv: {e}")
            return []
    
    def download_paper(self, paper_metadata, save_dir):
        """
        Download a paper from SocArXiv.
        
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
                
            elif paper_metadata.get("abstract"):
                # Save abstract as text file
                abstract_path = os.path.join(save_dir, f"{sanitized_title}.txt")
                with open(abstract_path, 'w') as f:
                    f.write(f"Title: {title}\n\n")
                    f.write(f"Authors: {', '.join(paper_metadata.get('authors', []))}\n\n")
                    f.write(f"Abstract: {paper_metadata['abstract']}")
                
                # Also save metadata
                metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(paper_metadata, f, indent=2)
                
                logger.info(f"Saved abstract for paper: {title}")
                return abstract_path
                
            else:
                # Just save metadata
                metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(paper_metadata, f, indent=2)
                
                logger.info(f"Saved metadata for paper: {title}")
                return metadata_path
                
        except Exception as e:
            logger.error(f"Error downloading paper from SocArXiv: {e}")
            return None
    
    def get_paper_by_id(self, paper_id):
        """
        Get a paper by its SocArXiv ID.
        
        Args:
            paper_id (str): SocArXiv paper ID
            
        Returns:
            dict or None: Paper metadata if found, None otherwise
        """
        # Prepare headers
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/json"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        try:
            # Make the request
            url = f"{self.base_url}/preprints/{paper_id}/"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Get paper data
            item = data.get("data", {})
            attributes = item.get("attributes", {})
            
            # Get basic metadata
            title = attributes.get("title", "")
            description = attributes.get("description", "")
            date_created = attributes.get("date_created")
            
            # Extract year from date
            year = None
            if date_created:
                try:
                    year = int(date_created.split("-")[0])
                except (ValueError, IndexError):
                    pass
            
            # Get download URL
            download_url = None
            links = item.get("links", {})
            if links and "download" in links:
                download_url = links["download"]
            
            # Get preprint URL
            preprint_url = None
            if links and "html" in links:
                preprint_url = links["html"]
            
            # Get contributors (authors)
            authors = []
            try:
                contributors_url = f"{self.base_url}/preprints/{paper_id}/contributors/"
                contributors_response = requests.get(contributors_url, headers=headers)
                contributors_response.raise_for_status()
                contributors_data = contributors_response.json()
                
                for contributor in contributors_data.get("data", []):
                    embeds = contributor.get("embeds", {})
                    user = embeds.get("user", {})
                    user_attributes = user.get("data", {}).get("attributes", {})
                    
                    full_name = user_attributes.get("full_name")
                    if full_name:
                        authors.append(full_name)
            except Exception as e:
                logger.warning(f"Error getting contributors for paper {paper_id}: {e}")
            
            # Get subjects/tags
            subjects = []
            for subject in attributes.get("subjects", []):
                if isinstance(subject, str):
                    subjects.append(subject)
                elif isinstance(subject, list) and len(subject) > 0:
                    subjects.append(subject[-1])
            
            # Create paper metadata
            paper_metadata = {
                "title": title,
                "abstract": description,
                "authors": authors,
                "year": year,
                "url": preprint_url,
                "pdf_url": download_url,
                "subjects": subjects,
                "source": self.source,
                "source_id": paper_id,
                "metadata_retrieved_at": time.time()
            }
            
            return paper_metadata
            
        except Exception as e:
            logger.error(f"Error getting paper by ID from SocArXiv: {e}")
            return None
