"""
RAG Dataset Builder

A flexible, memory-efficient tool for building datasets for 
Retrieval-Augmented Generation (RAG) systems.
"""

from .builder import (
    BaseProcessor,
    BaseChunker,
    BaseEmbedder,
    BaseOutputFormatter
)

from .embedders import (
    SentenceTransformerEmbedder,
    OpenAIEmbedder
)

from .formatters import (
    PathRAGFormatter,
    VectorDBFormatter,
    HuggingFaceDatasetFormatter
)

from .processors import (
    SimpleTextProcessor,
    PDFProcessor,
    CodeProcessor
)

from .chunkers import (
    SlidingWindowChunker,
    SemanticChunker,
    FixedSizeChunker
)

__version__ = "0.1.0"
