"""
PubMed Central Collector

Collects papers from PubMed Central using the NCBI E-utilities API.
"""
import os
import logging
import time
import json
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import urllib.parse

from .base_academic_collector import BaseAcademicCollector

logger = logging.getLogger(__name__)

class PubMedCollector(BaseAcademicCollector):
    """
    Collector for academic papers from PubMed Central.
    
    Uses the NCBI E-utilities API to search for and download papers.
    """
    
    def __init__(self, output_dir, deduplicator=None, email=None, api_key=None):
        """
        Initialize the PubMed collector.
        
        Args:
            output_dir (str): Directory to save downloaded papers
            deduplicator: Optional deduplicator to avoid duplicate papers
            email (str, optional): Email for NCBI API (preferred for higher rate limits)
            api_key (str, optional): API key for NCBI (allows higher rate limits)
        """
        super().__init__(output_dir, deduplicator)
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.source = "pubmed"
        
        # Sleep duration between requests to respect rate limits
        # With API key: 10 requests/second, without: 3 requests/second
        self.sleep_duration = 0.1 if api_key else 0.33
    
    def search_papers(self, query, max_results=50):
        """
        Search for papers on PubMed Central matching a query.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of paper metadata
        """
        # Build the search URL
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "usehistory": "y"  # Use history to retrieve full records
        }
        
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            # Make the search request
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            search_data = response.json()
            
            # Extract PMC IDs and query parameters for fetching details
            id_list = search_data["esearchresult"]["idlist"]
            query_key = search_data["esearchresult"]["querykey"]
            web_env = search_data["esearchresult"]["webenv"]
            
            if not id_list:
                logger.info(f"No papers found on PubMed for query: {query}")
                return []
            
            # Build the fetch URL
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pmc",
                "query_key": query_key,
                "WebEnv": web_env,
                "retmode": "xml",
                "retmax": max_results
            }
            
            if self.email:
                fetch_params["email"] = self.email
            if self.api_key:
                fetch_params["api_key"] = self.api_key
            
            # Sleep to respect rate limits
            time.sleep(self.sleep_duration)
            
            # Make the fetch request
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()
            
            # Parse the XML response
            root = ET.fromstring(fetch_response.content)
            
            # Process the results
            papers = []
            for article in root.findall(".//article"):
                try:
                    # Extract metadata
                    pmid = article.find(".//article-id[@pub-id-type='pmid']")
                    pmid = pmid.text if pmid is not None else None
                    
                    pmcid = article.find(".//article-id[@pub-id-type='pmc']")
                    pmcid = pmcid.text if pmcid is not None else None
                    
                    doi = article.find(".//article-id[@pub-id-type='doi']")
                    doi = doi.text if doi is not None else None
                    
                    # Extract title
                    title_elem = article.find(".//article-title")
                    title = "".join(title_elem.itertext()) if title_elem is not None else "Untitled"
                    
                    # Extract authors
                    authors = []
                    for contrib in article.findall(".//contrib[@contrib-type='author']"):
                        surname = contrib.find(".//surname")
                        given_names = contrib.find(".//given-names")
                        
                        if surname is not None and given_names is not None:
                            author_name = f"{given_names.text} {surname.text}"
                            authors.append(author_name)
                        elif surname is not None:
                            authors.append(surname.text)
                    
                    # Extract abstract
                    abstract_elem = article.find(".//abstract")
                    abstract = ""
                    if abstract_elem is not None:
                        for p in abstract_elem.findall(".//p"):
                            abstract += "".join(p.itertext()) + "\n\n"
                    
                    # Extract publication year
                    pub_date = article.find(".//pub-date")
                    year = None
                    if pub_date is not None:
                        year_elem = pub_date.find(".//year")
                        if year_elem is not None and year_elem.text:
                            try:
                                year = int(year_elem.text)
                            except ValueError:
                                pass
                    
                    # Extract journal info
                    journal_title = None
                    journal_elem = article.find(".//journal-title")
                    if journal_elem is not None:
                        journal_title = journal_elem.text
                    
                    # Create paper metadata
                    paper_metadata = {
                        "title": title,
                        "abstract": abstract.strip() if abstract else "",
                        "authors": authors,
                        "year": year,
                        "venue": journal_title,
                        "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else None,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "doi": doi,
                        "source": self.source,
                        "source_id": pmcid if pmcid else pmid,
                        "metadata_retrieved_at": time.time()
                    }
                    
                    # Only add papers with sufficient metadata
                    if paper_metadata["title"] and (paper_metadata["abstract"] or paper_metadata["pmcid"]):
                        papers.append(paper_metadata)
                        
                except Exception as e:
                    logger.error(f"Error processing PubMed article: {e}")
            
            logger.info(f"Found {len(papers)} papers on PubMed for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def download_paper(self, paper_metadata, save_dir):
        """
        Download a paper from PubMed Central.
        
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
            
            # Get PMCID for downloading PDF or full text
            pmcid = paper_metadata.get("pmcid")
            
            if pmcid:
                # Try to download PDF
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
                pdf_path = os.path.join(save_dir, f"{sanitized_title}.pdf")
                
                try:
                    # Try to download the PDF
                    response = requests.get(pdf_url, stream=True)
                    
                    # Check if response is PDF
                    if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                        with open(pdf_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # Also save metadata
                        metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
                        with open(metadata_path, 'w') as f:
                            json.dump(paper_metadata, f, indent=2)
                        
                        logger.info(f"Downloaded PDF for paper: {title}")
                        return pdf_path
                except Exception as pdf_error:
                    logger.warning(f"Could not download PDF for {pmcid}, falling back to text: {pdf_error}")
                
                # If PDF download failed, try to get full text XML
                try:
                    # Build the full text URL
                    fetch_url = f"{self.base_url}/efetch.fcgi"
                    fetch_params = {
                        "db": "pmc",
                        "id": pmcid,
                        "retmode": "xml"
                    }
                    
                    if self.email:
                        fetch_params["email"] = self.email
                    if self.api_key:
                        fetch_params["api_key"] = self.api_key
                    
                    # Make the fetch request
                    fetch_response = requests.get(fetch_url, params=fetch_params)
                    fetch_response.raise_for_status()
                    
                    # Save full text XML
                    xml_path = os.path.join(save_dir, f"{sanitized_title}.xml")
                    with open(xml_path, 'wb') as f:
                        f.write(fetch_response.content)
                    
                    # Also save converted text
                    try:
                        root = ET.fromstring(fetch_response.content)
                        body = root.find(".//body")
                        
                        if body is not None:
                            text_content = ""
                            for p in body.findall(".//p"):
                                text_content += "".join(p.itertext()) + "\n\n"
                            
                            text_path = os.path.join(save_dir, f"{sanitized_title}.txt")
                            with open(text_path, 'w') as f:
                                f.write(f"Title: {title}\n\n")
                                f.write(f"Authors: {', '.join(paper_metadata.get('authors', []))}\n\n")
                                if paper_metadata.get("abstract"):
                                    f.write(f"Abstract: {paper_metadata['abstract']}\n\n")
                                f.write("Full Text:\n\n")
                                f.write(text_content)
                            
                            # Also save metadata
                            metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
                            with open(metadata_path, 'w') as f:
                                json.dump(paper_metadata, f, indent=2)
                            
                            logger.info(f"Saved full text for paper: {title}")
                            return text_path
                    except Exception as e:
                        logger.warning(f"Error converting XML to text: {e}")
                    
                    # Save XML if text conversion failed
                    # Also save metadata
                    metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(paper_metadata, f, indent=2)
                    
                    logger.info(f"Saved XML for paper: {title}")
                    return xml_path
                    
                except Exception as xml_error:
                    logger.warning(f"Could not download full text for {pmcid}: {xml_error}")
            
            # If all else fails, save abstract and metadata
            if paper_metadata.get("abstract"):
                text_path = os.path.join(save_dir, f"{sanitized_title}.txt")
                with open(text_path, 'w') as f:
                    f.write(f"Title: {title}\n\n")
                    f.write(f"Authors: {', '.join(paper_metadata.get('authors', []))}\n\n")
                    f.write(f"Abstract: {paper_metadata['abstract']}")
                
                # Also save metadata
                metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(paper_metadata, f, indent=2)
                
                logger.info(f"Saved abstract for paper: {title}")
                return text_path
            
            # If no abstract, just save metadata
            metadata_path = os.path.join(save_dir, f"{sanitized_title}.json")
            with open(metadata_path, 'w') as f:
                json.dump(paper_metadata, f, indent=2)
            
            logger.info(f"Saved metadata for paper: {title}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Error downloading paper from PubMed: {e}")
            return None
    
    def get_paper_by_id(self, paper_id, id_type="pmcid"):
        """
        Get a paper by its PubMed ID.
        
        Args:
            paper_id (str): PubMed ID (PMID) or PMC ID
            id_type (str): Type of ID ('pmid' or 'pmcid')
            
        Returns:
            dict or None: Paper metadata if found, None otherwise
        """
        # Build the fetch URL
        fetch_url = f"{self.base_url}/efetch.fcgi"
        fetch_params = {
            "db": "pmc" if id_type == "pmcid" else "pubmed",
            "id": paper_id,
            "retmode": "xml"
        }
        
        if self.email:
            fetch_params["email"] = self.email
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        
        try:
            # Make the fetch request
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()
            
            # Parse the XML response
            root = ET.fromstring(fetch_response.content)
            
            # Process the results
            article = root.find(".//article")
            
            if article is None:
                logger.warning(f"No article found for ID {paper_id}")
                return None
            
            # Extract metadata
            pmid = article.find(".//article-id[@pub-id-type='pmid']")
            pmid = pmid.text if pmid is not None else None
            
            pmcid = article.find(".//article-id[@pub-id-type='pmc']")
            pmcid = pmcid.text if pmcid is not None else None
            
            doi = article.find(".//article-id[@pub-id-type='doi']")
            doi = doi.text if doi is not None else None
            
            # Extract title
            title_elem = article.find(".//article-title")
            title = "".join(title_elem.itertext()) if title_elem is not None else "Untitled"
            
            # Extract authors
            authors = []
            for contrib in article.findall(".//contrib[@contrib-type='author']"):
                surname = contrib.find(".//surname")
                given_names = contrib.find(".//given-names")
                
                if surname is not None and given_names is not None:
                    author_name = f"{given_names.text} {surname.text}"
                    authors.append(author_name)
                elif surname is not None:
                    authors.append(surname.text)
            
            # Extract abstract
            abstract_elem = article.find(".//abstract")
            abstract = ""
            if abstract_elem is not None:
                for p in abstract_elem.findall(".//p"):
                    abstract += "".join(p.itertext()) + "\n\n"
            
            # Extract publication year
            pub_date = article.find(".//pub-date")
            year = None
            if pub_date is not None:
                year_elem = pub_date.find(".//year")
                if year_elem is not None and year_elem.text:
                    try:
                        year = int(year_elem.text)
                    except ValueError:
                        pass
            
            # Extract journal info
            journal_title = None
            journal_elem = article.find(".//journal-title")
            if journal_elem is not None:
                journal_title = journal_elem.text
            
            # Create paper metadata
            paper_metadata = {
                "title": title,
                "abstract": abstract.strip() if abstract else "",
                "authors": authors,
                "year": year,
                "venue": journal_title,
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else None,
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "source": self.source,
                "source_id": pmcid if pmcid else pmid,
                "metadata_retrieved_at": time.time()
            }
            
            return paper_metadata
            
        except Exception as e:
            logger.error(f"Error getting paper by ID from PubMed: {e}")
            return None
