"""
Academic Paper Deduplicator

This module provides specialized deduplication for academic papers
based on title and author similarity.
"""
import os
import logging
import re
import time
import difflib
from .base import BaseDeduplicator
from .registry import DocumentRegistry

logger = logging.getLogger(__name__)

class AcademicPaperDeduplicator(BaseDeduplicator):
    """
    Deduplicator specialized for academic papers using title and author matching.
    Supports deduplication across multiple academic sources like ArXiv, 
    Semantic Scholar, PubMed Central, and SocArXiv.
    """
    
    def __init__(self, registry_path=None, title_similarity_threshold=0.85):
        """
        Initialize the academic paper deduplicator.
        
        Args:
            registry_path (str, optional): Path to the registry file
            title_similarity_threshold (float, optional): Threshold for title similarity (0-1)
        """
        self.registry = DocumentRegistry(registry_path)
        self.title_similarity_threshold = title_similarity_threshold
        logger.info(f"Initialized academic paper deduplicator with {self.registry.get_document_count()} documents")
        # Log the breakdown by source
        for source in self.registry.get_sources():
            source_count = len(self.registry.get_documents_by_source(source))
            logger.info(f"  - {source}: {source_count} documents")
    
    def _normalize_title(self, title):
        """
        Normalize a paper title for comparison.
        
        Args:
            title (str): Paper title
            
        Returns:
            str: Normalized title
        """
        if not title:
            return ""
        
        # Convert to lowercase
        normalized = title.lower()
        
        # Remove punctuation except hyphens in compound words
        normalized = re.sub(r'[^\w\s-]', ' ', normalized)
        
        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Trim
        normalized = normalized.strip()
        
        return normalized
    
    def _normalize_author(self, author):
        """
        Normalize an author name for comparison.
        
        Args:
            author (str): Author name
            
        Returns:
            str: Normalized author name (last name only)
        """
        if not author:
            return ""
        
        # Get last name (assume it's the last word)
        parts = author.split()
        if len(parts) > 0:
            return parts[-1].lower()
        return ""
    
    def _get_title_similarity(self, title1, title2):
        """
        Calculate the similarity between two titles.
        
        Args:
            title1 (str): First title
            title2 (str): Second title
            
        Returns:
            float: Similarity score (0-1)
        """
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        norm_title1 = self._normalize_title(title1)
        norm_title2 = self._normalize_title(title2)
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, norm_title1, norm_title2).ratio()
        
        return similarity
    
    def find_duplicate(self, document_metadata):
        """
        Find a duplicate paper in the registry based on title and author similarity.
        
        Args:
            document_metadata (dict): Paper metadata including:
                - title: Paper title
                - authors: List of author names
                - source: Source repository
                - source_id: ID in the source repository
                - year: Publication year (optional)
                
        Returns:
            str or None: Document ID of the duplicate if found, None otherwise
        """
        if not document_metadata.get("title") or not document_metadata.get("authors"):
            logger.warning("Missing title or authors in document metadata")
            return None
        
        # First, try to find an exact match by DOI if available
        if document_metadata.get("doi"):
            for doc_id, doc in self.registry.get_all_documents().items():
                if doc["metadata"].get("doi") == document_metadata["doi"]:
                    logger.info(f"Found duplicate by DOI: {doc_id}")
                    return doc_id
        
        # Next, try to find a match by source ID if from the same source
        source = document_metadata.get("source")
        source_id = document_metadata.get("source_id")
        if source and source_id:
            for doc_id, doc in self.registry.get_all_documents().items():
                if doc["metadata"].get("source") == source and doc["metadata"].get("source_id") == source_id:
                    logger.info(f"Found duplicate by source ID: {doc_id}")
                    return doc_id
        
        # If no exact matches, try title + first author
        title = document_metadata["title"]
        first_author = document_metadata["authors"][0] if document_metadata["authors"] else None
        
        if not first_author:
            return None
        
        first_author_last = self._normalize_author(first_author)
        
        for doc_id, doc in self.registry.get_all_documents().items():
            doc_title = doc["metadata"].get("title", "")
            doc_authors = doc["metadata"].get("authors", [])
            
            if not doc_authors:
                continue
            
            doc_first_author_last = self._normalize_author(doc_authors[0])
            
            # Check if first author's last name matches
            if first_author_last == doc_first_author_last:
                # Check title similarity
                similarity = self._get_title_similarity(title, doc_title)
                if similarity >= self.title_similarity_threshold:
                    logger.info(f"Found duplicate by title+author: {doc_id} (similarity: {similarity:.2f})")
                    return doc_id
        
        # No duplicates found
        return None
    
    def register_document(self, document_id, document_metadata):
        """
        Register a new academic paper in the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            document_metadata (dict): Document metadata
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        # Ensure required fields are present
        required_fields = ["title", "authors", "source"]
        for field in required_fields:
            if field not in document_metadata:
                logger.warning(f"Missing required field '{field}' in document metadata")
                return False
        
        # Add additional fields
        document_metadata.setdefault("registered_at", time.time())
        document_metadata.setdefault("sources", [document_metadata.get("source")])
        document_metadata.setdefault("search_terms", [])
        
        # Add to registry
        success = self.registry.add_document(document_id, document_metadata)
        
        # Save registry
        if success:
            self.registry.save()
        
        return success
    
    def update_document_metadata(self, document_id, new_metadata):
        """
        Update metadata for an existing academic paper in the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            new_metadata (dict): New/additional metadata to merge
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.registry.document_exists(document_id):
            logger.warning(f"Document {document_id} not found in registry")
            return False
        
        # Update in registry
        success = self.registry.update_document(document_id, new_metadata)
        
        # Save registry
        if success:
            self.registry.save()
        
        return success
    
    def get_document_metadata(self, document_id):
        """
        Retrieve metadata for an academic paper from the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            
        Returns:
            dict or None: Document metadata if found, None otherwise
        """
        doc = self.registry.get_document(document_id)
        if doc:
            return doc["metadata"]
        return None
    
    def get_document_sources(self, document_id):
        """
        Get sources where a document was found.
        
        Args:
            document_id (str): Unique identifier for the document
            
        Returns:
            list: List of sources
        """
        doc = self.registry.get_document(document_id)
        if doc:
            return doc.get("sources", [])
        return []
    
    def get_document_search_terms(self, document_id):
        """
        Get search terms that found a document.
        
        Args:
            document_id (str): Unique identifier for the document
            
        Returns:
            list: List of search terms
        """
        doc = self.registry.get_document(document_id)
        if doc:
            return doc.get("search_terms", [])
        return []
    
    def add_source_to_document(self, document_id, source):
        """
        Add a source to a document's metadata.
        
        Args:
            document_id (str): Unique identifier for the document
            source (str): Source name
            
        Returns:
            bool: True if successful, False otherwise
        """
        doc = self.registry.get_document(document_id)
        if not doc:
            return False
        
        if source not in doc.get("sources", []):
            if "sources" not in doc:
                doc["sources"] = []
            doc["sources"].append(source)
            self.registry.save()
        
        return True
    
    def add_search_term_to_document(self, document_id, search_term):
        """
        Add a search term to a document's metadata.
        
        Args:
            document_id (str): Unique identifier for the document
            search_term (str): Search term
            
        Returns:
            bool: True if successful, False otherwise
        """
        doc = self.registry.get_document(document_id)
        if not doc:
            return False
        
        if search_term not in doc.get("search_terms", []):
            if "search_terms" not in doc:
                doc["search_terms"] = []
            doc["search_terms"].append(search_term)
            self.registry.save()
        
        return True
    
    # Additional utility methods
    
    def get_statistics(self):
        """
        Get statistics about the registry.
        
        Returns:
            dict: Statistics about the registry
        """
        sources = self.registry.get_sources()
        search_terms = self.registry.get_search_terms()
        
        stats = {
            "total_documents": self.registry.get_document_count(),
            "sources": {
                source: len(self.registry.get_documents_by_source(source))
                for source in sources
            },
            "search_terms": {
                term: len(self.registry.get_documents_by_search_term(term))
                for term in search_terms
            }
        }
        
        return stats
