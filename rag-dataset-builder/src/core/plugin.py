#!/usr/bin/env python3
"""
Plugin System for RAG Dataset Builder

This module provides functionality for registering and loading plugins
for the RAG Dataset Builder framework. This allows for easy extension
with custom document processors, chunkers, embedders, storage backends,
and retrieval systems.
"""

import os
import sys
import logging
import importlib
import pkgutil
from typing import Dict, List, Any, Type, Optional, Callable, Union, TypeVar

from .interfaces import (
    DocumentProcessor,
    TextChunker,
    Embedder,
    StorageBackend,
    OutputFormatter,
    DatasetCollector,
    PerformanceTracker,
    RetrievalSystem
)

logger = logging.getLogger("rag_dataset_builder.plugin")

# Type variable for plugin types
T = TypeVar('T')

# Registry for plugin implementations
_plugin_registry = {
    'document_processors': {},
    'chunkers': {},
    'embedders': {},
    'storage_backends': {},
    'formatters': {},
    'collectors': {},
    'trackers': {},
    'retrieval_systems': {}
}

def register_plugin(
    plugin_type: str, 
    name: str, 
    implementation: Type[T]
) -> None:
    """
    Register a plugin implementation.
    
    Args:
        plugin_type: Type of plugin (e.g., 'document_processors', 'chunkers')
        name: Name of the implementation
        implementation: Implementation class
    """
    if plugin_type not in _plugin_registry:
        raise ValueError(f"Unknown plugin type: {plugin_type}")
    
    _plugin_registry[plugin_type][name] = implementation
    logger.info(f"Registered {plugin_type} plugin: {name}")

def get_plugin(
    plugin_type: str, 
    name: str
) -> Optional[Type[T]]:
    """
    Get a plugin implementation by name.
    
    Args:
        plugin_type: Type of plugin
        name: Name of the implementation
        
    Returns:
        Plugin implementation class, or None if not found
    """
    if plugin_type not in _plugin_registry:
        raise ValueError(f"Unknown plugin type: {plugin_type}")
    
    return _plugin_registry[plugin_type].get(name)

def list_plugins(plugin_type: str) -> List[str]:
    """
    List available plugins of a specific type.
    
    Args:
        plugin_type: Type of plugin
        
    Returns:
        List of plugin names
    """
    if plugin_type not in _plugin_registry:
        raise ValueError(f"Unknown plugin type: {plugin_type}")
    
    return list(_plugin_registry[plugin_type].keys())

def discover_plugins() -> None:
    """
    Discover and register plugins from the src/implementations, src/processors, 
    src/chunkers, src/embedders, and src/formatters directories.
    """
    plugin_dirs = [
        'implementations',
        'processors',
        'chunkers', 
        'embedders',
        'formatters',
        'collectors',
        'trackers',
        'storage'
    ]
    
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for plugin_dir in plugin_dirs:
        full_dir = os.path.join(src_dir, plugin_dir)
        if not os.path.exists(full_dir):
            continue
        
        logger.info(f"Discovering plugins in {full_dir}")
        
        # Add directory to Python path if it's not already there
        if full_dir not in sys.path:
            sys.path.append(full_dir)
        
        # Discover and import all modules in the directory
        for _, name, is_pkg in pkgutil.iter_modules([full_dir]):
            if is_pkg:
                continue  # Skip packages, only import modules
            
            try:
                module = importlib.import_module(f"{plugin_dir}.{name}")
                logger.info(f"Loaded plugin module: {plugin_dir}.{name}")
            except ImportError as e:
                logger.error(f"Error importing plugin module {plugin_dir}.{name}: {e}")

def create_processor(name: str, **kwargs) -> DocumentProcessor:
    """
    Create a document processor by name.
    
    Args:
        name: Name of the processor
        **kwargs: Additional arguments to pass to the processor constructor
        
    Returns:
        Document processor instance
    """
    processor_cls = get_plugin('document_processors', name)
    if not processor_cls:
        raise ValueError(f"Document processor not found: {name}")
    
    return processor_cls(**kwargs)

def create_chunker(name: str, **kwargs) -> TextChunker:
    """
    Create a text chunker by name.
    
    Args:
        name: Name of the chunker
        **kwargs: Additional arguments to pass to the chunker constructor
        
    Returns:
        Text chunker instance
    """
    chunker_cls = get_plugin('chunkers', name)
    if not chunker_cls:
        raise ValueError(f"Text chunker not found: {name}")
    
    return chunker_cls(**kwargs)

def create_embedder(name: str, **kwargs) -> Embedder:
    """
    Create an embedder by name.
    
    Args:
        name: Name of the embedder
        **kwargs: Additional arguments to pass to the embedder constructor
        
    Returns:
        Embedder instance
    """
    embedder_cls = get_plugin('embedders', name)
    if not embedder_cls:
        raise ValueError(f"Embedder not found: {name}")
    
    return embedder_cls(**kwargs)

def create_storage_backend(name: str, **kwargs) -> StorageBackend:
    """
    Create a storage backend by name.
    
    Args:
        name: Name of the storage backend
        **kwargs: Additional arguments to pass to the storage backend constructor
        
    Returns:
        Storage backend instance
    """
    backend_cls = get_plugin('storage_backends', name)
    if not backend_cls:
        raise ValueError(f"Storage backend not found: {name}")
    
    return backend_cls(**kwargs)

def create_formatter(name: str, **kwargs) -> OutputFormatter:
    """
    Create an output formatter by name.
    
    Args:
        name: Name of the formatter
        **kwargs: Additional arguments to pass to the formatter constructor
        
    Returns:
        Formatter instance
    """
    formatter_cls = get_plugin('formatters', name)
    if not formatter_cls:
        raise ValueError(f"Formatter not found: {name}")
    
    return formatter_cls(**kwargs)

def create_collector(name: str, **kwargs) -> DatasetCollector:
    """
    Create a dataset collector by name.
    
    Args:
        name: Name of the collector
        **kwargs: Additional arguments to pass to the collector constructor
        
    Returns:
        Dataset collector instance
    """
    collector_cls = get_plugin('collectors', name)
    if not collector_cls:
        raise ValueError(f"Dataset collector not found: {name}")
    
    return collector_cls(**kwargs)

def create_tracker(name: str, **kwargs) -> PerformanceTracker:
    """
    Create a performance tracker by name.
    
    Args:
        name: Name of the tracker
        **kwargs: Additional arguments to pass to the tracker constructor
        
    Returns:
        Performance tracker instance
    """
    tracker_cls = get_plugin('trackers', name)
    if not tracker_cls:
        raise ValueError(f"Performance tracker not found: {name}")
    
    return tracker_cls(**kwargs)

def create_retrieval_system(name: str, **kwargs) -> RetrievalSystem:
    """
    Create a retrieval system by name.
    
    Args:
        name: Name of the retrieval system
        **kwargs: Additional arguments to pass to the retrieval system constructor
        
    Returns:
        Retrieval system instance
    """
    system_cls = get_plugin('retrieval_systems', name)
    if not system_cls:
        raise ValueError(f"Retrieval system not found: {name}")
    
    return system_cls(**kwargs)

# Decorator for registering plugins
def register(plugin_type: str, name: str = None):
    """
    Decorator for registering plugin implementations.
    
    Args:
        plugin_type: Type of plugin (e.g., 'document_processors', 'chunkers')
        name: Name of the implementation (defaults to class name)
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        nonlocal name
        if name is None:
            name = cls.__name__
        register_plugin(plugin_type, name, cls)
        return cls
    return decorator

# Register standard implementations
from .base import (
    BaseDocumentProcessor,
    BaseTextChunker,
    BaseEmbedder,
    NetworkXStorageBackend,
    BaseOutputFormatter,
    BasePerformanceTracker,
    BaseRetrievalSystem
)

# Register base implementations
register_plugin('document_processors', 'base', BaseDocumentProcessor)
register_plugin('chunkers', 'base', BaseTextChunker)
register_plugin('embedders', 'base', BaseEmbedder)
register_plugin('storage_backends', 'networkx', NetworkXStorageBackend)
register_plugin('formatters', 'base', BaseOutputFormatter)
register_plugin('trackers', 'base', BasePerformanceTracker)
register_plugin('retrieval_systems', 'base', BaseRetrievalSystem)

# Discover plugins
discover_plugins()
