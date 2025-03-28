"""
Base Deduplicator Interface

This module defines the abstract base class for document deduplication.
"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseDeduplicator(ABC):
    """Abstract base class for document deduplication."""
    
    @abstractmethod
    def find_duplicate(self, document_metadata):
        """
        Find a duplicate document in the registry based on metadata.
        
        Args:
            document_metadata (dict): Metadata for the document to check
                
        Returns:
            str or None: Document ID of the duplicate if found, None otherwise
        """
        pass
    
    @abstractmethod
    def register_document(self, document_id, document_metadata):
        """
        Register a new document in the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            document_metadata (dict): Metadata for the document
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def update_document_metadata(self, document_id, new_metadata):
        """
        Update metadata for an existing document in the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            new_metadata (dict): New/additional metadata to merge
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_document_metadata(self, document_id):
        """
        Retrieve metadata for a document from the registry.
        
        Args:
            document_id (str): Unique identifier for the document
            
        Returns:
            dict or None: Document metadata if found, None otherwise
        """
        pass
