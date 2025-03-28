#!/usr/bin/env python3
"""
Component Factory for RAG Dataset Builder

This module provides factory functions for creating different components
of the RAG Dataset Builder framework based on configuration.
"""

import logging
from typing import Dict, Any, Optional, Type, List

from .interfaces import (
    DocumentProcessor,
    TextChunker,
    Embedder,
    OutputFormatter,
    StorageBackend,
    RetrievalSystem,
    PerformanceTracker
)
from .plugin import (
    get_registered_plugins,
    create_document_processor,
    create_text_chunker,
    create_embedder,
    create_output_formatter,
    create_storage_backend,
    create_retrieval_system,
    create_performance_tracker
)

logger = logging.getLogger("rag_dataset_builder.component_factory")

class ComponentFactory:
    """
    Factory for creating components based on configuration.
    
    This class provides methods for creating different components of the
    RAG Dataset Builder framework based on configuration values.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ComponentFactory with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def create_document_processor(self, processor_type: str) -> DocumentProcessor:
        """
        Create a document processor of the specified type.
        
        Args:
            processor_type: Type of document processor to create
            
        Returns:
            Document processor instance
        """
        processor_config = self.config.get("processors", {}).get(processor_type, {})
        return create_document_processor(processor_type, config=processor_config)
    
    def create_text_chunker(self, chunker_type: str) -> TextChunker:
        """
        Create a text chunker of the specified type.
        
        Args:
            chunker_type: Type of text chunker to create
            
        Returns:
            Text chunker instance
        """
        chunker_config = self.config.get("chunkers", {}).get(chunker_type, {})
        return create_text_chunker(chunker_type, config=chunker_config)
    
    def create_embedder(self, embedder_type: str) -> Embedder:
        """
        Create an embedder of the specified type.
        
        Args:
            embedder_type: Type of embedder to create
            
        Returns:
            Embedder instance
        """
        embedder_config = self.config.get("embedders", {}).get(embedder_type, {})
        return create_embedder(embedder_type, config=embedder_config)
    
    def create_storage_backend(self, backend_type: str) -> StorageBackend:
        """
        Create a storage backend of the specified type.
        
        Args:
            backend_type: Type of storage backend to create
            
        Returns:
            Storage backend instance
        """
        backend_config = self.config.get("storage", {}).get(backend_type, {})
        return create_storage_backend(backend_type, config=backend_config)
    
    def create_output_formatter(self, formatter_type: str) -> OutputFormatter:
        """
        Create an output formatter of the specified type.
        
        Args:
            formatter_type: Type of output formatter to create
            
        Returns:
            Output formatter instance
        """
        formatter_config = self.config.get("formatters", {}).get(formatter_type, {})
        return create_output_formatter(formatter_type, config=formatter_config)
    
    def create_retrieval_system(self, system_type: str, tracker: Optional[PerformanceTracker] = None) -> RetrievalSystem:
        """
        Create a retrieval system of the specified type.
        
        Args:
            system_type: Type of retrieval system to create
            tracker: Optional performance tracker
            
        Returns:
            Retrieval system instance
        """
        system_config = self.config.get("retrieval_systems", {}).get(system_type, {})
        system = create_retrieval_system(system_type, tracker=tracker)
        
        # Configure the system with its specific configuration
        system.configure(system_config)
        
        return system
    
    def create_performance_tracker(self, tracker_type: str) -> Optional[PerformanceTracker]:
        """
        Create a performance tracker of the specified type.
        
        Args:
            tracker_type: Type of performance tracker to create
            
        Returns:
            Performance tracker instance or None if disabled
        """
        tracker_config = self.config.get("monitoring", {}).get(tracker_type, {})
        
        # Skip if tracker is disabled
        if not tracker_config.get("enabled", True):
            return None
            
        return create_performance_tracker(tracker_type, config=tracker_config)

def create_component_factory(config: Dict[str, Any]) -> ComponentFactory:
    """
    Create a ComponentFactory instance with the specified configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ComponentFactory instance
    """
    return ComponentFactory(config)
