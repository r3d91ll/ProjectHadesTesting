"""
Document Registry

This module provides a persistent registry for tracking documents across multiple collection runs.
"""
import json
import os
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentRegistry:
    """Registry for tracking documents and their metadata across collection runs."""
    
    def __init__(self, registry_path=None):
        """
        Initialize the document registry.
        
        Args:
            registry_path (str, optional): Path to the registry file.
                If not provided, a default path will be used.
        """
        self.registry_path = registry_path or os.path.join(
            os.getcwd(), "data", "registry", "document_registry.json"
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        # Load existing registry if it exists
        self.registry = {}
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded registry with {len(self.registry)} documents")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                # Create a backup of the corrupted file
                if os.path.getsize(self.registry_path) > 0:
                    backup_path = f"{self.registry_path}.backup.{int(time.time())}"
                    try:
                        os.rename(self.registry_path, backup_path)
                        logger.info(f"Created backup of corrupted registry at {backup_path}")
                    except Exception as e:
                        logger.error(f"Failed to create backup: {e}")
    
    def save(self):
        """
        Save the registry to disk.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            logger.info(f"Saved registry with {len(self.registry)} documents")
            return True
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            return False
    
    def add_document(self, document_id, metadata):
        """
        Add a document to the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            metadata (dict): Document metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if document_id in self.registry:
            logger.warning(f"Document {document_id} already exists in registry")
            return False
        
        self.registry[document_id] = {
            "metadata": metadata,
            "added_at": time.time(),
            "updated_at": time.time(),
            "sources": metadata.get("sources", []),
            "search_terms": metadata.get("search_terms", [])
        }
        return True
    
    def update_document(self, document_id, new_metadata):
        """
        Update a document in the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            new_metadata (dict): New metadata to merge
            
        Returns:
            bool: True if successful, False otherwise
        """
        if document_id not in self.registry:
            logger.warning(f"Document {document_id} not found in registry")
            return False
        
        # Update metadata
        self.registry[document_id]["metadata"].update(new_metadata)
        
        # Update sources if provided
        if "sources" in new_metadata:
            for source in new_metadata["sources"]:
                if source not in self.registry[document_id]["sources"]:
                    self.registry[document_id]["sources"].append(source)
        
        # Update search terms if provided
        if "search_terms" in new_metadata:
            for term in new_metadata["search_terms"]:
                if term not in self.registry[document_id]["search_terms"]:
                    self.registry[document_id]["search_terms"].append(term)
        
        # Update timestamp
        self.registry[document_id]["updated_at"] = time.time()
        
        return True
    
    def get_document(self, document_id):
        """
        Get a document from the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            
        Returns:
            dict or None: Document data if found, None otherwise
        """
        return self.registry.get(document_id)
    
    def get_all_documents(self):
        """
        Get all documents in the registry.
        
        Returns:
            dict: All documents in the registry
        """
        return self.registry
    
    def document_exists(self, document_id):
        """
        Check if a document exists in the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            
        Returns:
            bool: True if the document exists, False otherwise
        """
        return document_id in self.registry
    
    def get_document_count(self):
        """
        Get the number of documents in the registry.
        
        Returns:
            int: Number of documents
        """
        return len(self.registry)
    
    def get_sources(self):
        """
        Get all unique sources in the registry.
        
        Returns:
            list: List of unique sources
        """
        sources = set()
        for doc in self.registry.values():
            sources.update(doc.get("sources", []))
        return list(sources)
    
    def get_search_terms(self):
        """
        Get all unique search terms in the registry.
        
        Returns:
            list: List of unique search terms
        """
        terms = set()
        for doc in self.registry.values():
            terms.update(doc.get("search_terms", []))
        return list(terms)
    
    def get_documents_by_source(self, source):
        """
        Get documents from a specific source.
        
        Args:
            source (str): Source name
            
        Returns:
            dict: Documents from the specified source
        """
        return {
            doc_id: doc for doc_id, doc in self.registry.items()
            if source in doc.get("sources", [])
        }
    
    def get_documents_by_search_term(self, search_term):
        """
        Get documents matching a specific search term.
        
        Args:
            search_term (str): Search term
            
        Returns:
            dict: Documents matching the search term
        """
        return {
            doc_id: doc for doc_id, doc in self.registry.items()
            if search_term in doc.get("search_terms", [])
        }
