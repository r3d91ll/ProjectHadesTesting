"""
Base Academic Collector

Provides the base functionality for collecting academic papers from different sources.
"""
import os
import logging
import time
import json
import requests
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAcademicCollector(ABC):
    """
    Base class for academic paper collectors.
    
    Provides common functionality for collecting papers from different academic sources.
    """
    
    def __init__(self, output_dir, deduplicator=None):
        """
        Initialize the academic collector.
        
        Args:
            output_dir (str): Directory to save downloaded papers
            deduplicator: Optional deduplicator to use for avoiding duplicate papers
        """
        self.output_dir = output_dir
        self.deduplicator = deduplicator
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Stats tracking
        self.total_papers_found = 0
        self.total_papers_downloaded = 0
        self.total_duplicates_skipped = 0
        self.search_stats = {}
    
    @abstractmethod
    def search_papers(self, query, max_results=50):
        """
        Search for papers matching a query.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of paper metadata
        """
        pass
    
    @abstractmethod
    def download_paper(self, paper_metadata, save_dir):
        """
        Download a paper based on its metadata.
        
        Args:
            paper_metadata (dict): Paper metadata
            save_dir (str): Directory to save the paper
            
        Returns:
            str or None: Path to downloaded file if successful, None otherwise
        """
        pass
    
    def process_search_results(self, results, search_term):
        """
        Process search results and check for duplicates.
        
        Args:
            results (list): List of paper metadata
            search_term (str): Search term that yielded these results
            
        Returns:
            tuple: (new_papers, duplicate_papers) lists
        """
        new_papers = []
        duplicate_papers = []
        
        for paper in results:
            # Skip if required fields are missing
            if not paper.get('title') or not paper.get('authors'):
                continue
            
            # Add search term to metadata
            if 'search_terms' not in paper:
                paper['search_terms'] = []
            if search_term not in paper['search_terms']:
                paper['search_terms'].append(search_term)
            
            # Check for duplicates if deduplicator is available
            if self.deduplicator:
                duplicate_id = self.deduplicator.find_duplicate(paper)
                if duplicate_id:
                    # Found duplicate, update metadata
                    self.deduplicator.add_search_term_to_document(duplicate_id, search_term)
                    if paper.get('source'):
                        self.deduplicator.add_source_to_document(duplicate_id, paper['source'])
                    duplicate_papers.append((duplicate_id, paper))
                    continue
            
            # No duplicate found, add to new papers
            new_papers.append(paper)
        
        logger.info(f"Found {len(results)} papers, {len(new_papers)} new, {len(duplicate_papers)} duplicates")
        return new_papers, duplicate_papers
    
    def collect_papers(self, search_terms, max_papers_per_term=50):
        """
        Collect papers matching a list of search terms.
        
        Args:
            search_terms (list): List of search terms
            max_papers_per_term (int): Maximum papers to collect per term
            
        Returns:
            int: Number of papers downloaded
        """
        start_time = time.time()
        papers_downloaded = 0
        
        for term in search_terms:
            term_start_time = time.time()
            logger.info(f"Searching for papers with term: {term}")
            
            try:
                # Search for papers
                results = self.search_papers(term, max_papers_per_term)
                self.total_papers_found += len(results)
                
                # Process results and check for duplicates
                new_papers, duplicate_papers = self.process_search_results(results, term)
                self.total_duplicates_skipped += len(duplicate_papers)
                
                # Create directory for this search term
                term_dir = os.path.join(self.output_dir, self._sanitize_filename(term))
                os.makedirs(term_dir, exist_ok=True)
                
                # Save metadata about the search
                with open(os.path.join(term_dir, 'search_metadata.json'), 'w') as f:
                    json.dump({
                        'search_term': term,
                        'timestamp': time.time(),
                        'total_results': len(results),
                        'new_papers': len(new_papers),
                        'duplicate_papers': len(duplicate_papers)
                    }, f, indent=2)
                
                # Download new papers
                downloaded = 0
                for paper in new_papers:
                    paper_path = self.download_paper(paper, term_dir)
                    
                    if paper_path:
                        downloaded += 1
                        papers_downloaded += 1
                        self.total_papers_downloaded += 1
                        
                        # Register in deduplicator if available
                        if self.deduplicator:
                            doc_id = f"{paper.get('source', 'unknown')}-{paper.get('source_id', hash(paper['title']))}"
                            self.deduplicator.register_document(doc_id, paper)
            
                # Update search stats
                self.search_stats[term] = {
                    'found': len(results),
                    'new': len(new_papers),
                    'duplicates': len(duplicate_papers),
                    'downloaded': downloaded,
                    'time_taken': time.time() - term_start_time
                }
                
                logger.info(f"Downloaded {downloaded} papers for term: {term}")
                
            except Exception as e:
                logger.error(f"Error collecting papers for term '{term}': {e}")
        
        logger.info(f"Collected {papers_downloaded} papers in {time.time() - start_time:.2f} seconds")
        return papers_downloaded
    
    def _sanitize_filename(self, filename):
        """
        Sanitize a string to be used as a filename.
        
        Args:
            filename (str): Input string
            
        Returns:
            str: Sanitized filename
        """
        # Replace invalid characters with underscores
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return ''.join(c if c in valid_chars else '_' for c in filename)
    
    def get_statistics(self):
        """
        Get statistics about the collection process.
        
        Returns:
            dict: Collection statistics
        """
        return {
            'total_papers_found': self.total_papers_found,
            'total_papers_downloaded': self.total_papers_downloaded,
            'total_duplicates_skipped': self.total_duplicates_skipped,
            'search_stats': self.search_stats
        }
