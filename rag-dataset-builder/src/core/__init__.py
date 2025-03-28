"""
Core module for the RAG Dataset Builder framework.

This package contains the fundamental interfaces and base classes
that define the extension points for the framework.
"""

# Export interfaces
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

# Export base implementations
from .base import (
    BaseDocumentProcessor,
    BaseTextChunker,
    BaseEmbedder,
    NetworkXStorageBackend,
    BaseOutputFormatter,
    BasePerformanceTracker,
    BaseRetrievalSystem
)

# Export configuration utilities
from .config import (
    load_config,
    validate_required_config,
    get_nested_config,
    merge_configs,
    create_rag_config,
    ConfigurationError
)

# Export plugin system
from .plugin import (
    register_plugin,
    get_plugin,
    list_plugins,
    discover_plugins,
    create_processor,
    create_chunker,
    create_embedder,
    create_storage_backend,
    create_formatter,
    create_collector,
    create_tracker,
    create_retrieval_system,
    register
)

__all__ = [
    # Interfaces
    'DocumentProcessor',
    'TextChunker',
    'Embedder',
    'StorageBackend',
    'OutputFormatter',
    'DatasetCollector',
    'PerformanceTracker',
    'RetrievalSystem',
    
    # Base implementations
    'BaseDocumentProcessor',
    'BaseTextChunker',
    'BaseEmbedder',
    'NetworkXStorageBackend',
    'BaseOutputFormatter',
    'BasePerformanceTracker',
    'BaseRetrievalSystem',
    
    # Configuration utilities
    'load_config',
    'validate_required_config',
    'get_nested_config',
    'merge_configs',
    'create_rag_config',
    'ConfigurationError',
    
    # Plugin system
    'register_plugin',
    'get_plugin',
    'list_plugins',
    'discover_plugins',
    'create_processor',
    'create_chunker',
    'create_embedder',
    'create_storage_backend',
    'create_formatter',
    'create_collector',
    'create_tracker',
    'create_retrieval_system',
    'register'
]
