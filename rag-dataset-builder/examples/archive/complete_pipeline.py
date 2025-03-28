#!/usr/bin/env python3
"""
Complete RAG Dataset Builder Pipeline Example

This script demonstrates a complete pipeline for building a RAG dataset:
1. Loading configuration
2. Collecting academic papers
3. Processing documents
4. Chunking text
5. Generating embeddings
6. Formatting output
7. Tracking performance with Arize Phoenix

Usage:
    python complete_pipeline.py --config ../config/default_config.yaml
"""

import os
import sys
import time
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processors import SimpleTextProcessor, PDFProcessor, CodeProcessor
from src.chunkers import SlidingWindowChunker, SemanticChunker, FixedSizeChunker
from src.collectors.academic_collector import AcademicCollector
from src.formatters import PathRAGFormatter, VectorDBFormatter, HuggingFaceDatasetFormatter
from src.utils.arize_integration import get_arize_adapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_dataset_builder.log')
    ]
)
logger = logging.getLogger('rag_dataset_builder')

# Try to import SentenceTransformerEmbedder, with fallback to a mock version
try:
    from src.embedders import SentenceTransformerEmbedder, OpenAIEmbedder
except ImportError:
    # Mock embedders for demonstration
    logger.warning("Embedders module not found, using mock embedders")
    
    class SentenceTransformerEmbedder:
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            self.model_name = model_name
            logger.info(f"Initialized mock SentenceTransformerEmbedder with model {model_name}")
        
        def embed_texts(self, texts):
            return [[0.1] * 384 for _ in texts]
    
    class OpenAIEmbedder:
        def __init__(self, model_name="text-embedding-ada-002"):
            self.model_name = model_name
            logger.info(f"Initialized mock OpenAIEmbedder with model {model_name}")
        
        def embed_texts(self, texts):
            return [[0.1] * 1536 for _ in texts]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def get_processor(config: Dict[str, Any]):
    """Get document processor based on configuration."""
    processor_type = config.get('processor', {}).get('type', 'simple_text')
    
    if processor_type == 'pdf':
        return PDFProcessor()
    elif processor_type == 'code':
        return CodeProcessor()
    else:
        return SimpleTextProcessor()


def get_chunker(config: Dict[str, Any]):
    """Get text chunker based on configuration."""
    chunker_config = config.get('chunker', {})
    chunker_type = chunker_config.get('type', 'sliding_window')
    
    if chunker_type == 'semantic':
        return SemanticChunker(
            max_chunk_size=chunker_config.get('max_chunk_size', 500),
            min_chunk_size=chunker_config.get('min_chunk_size', 100)
        )
    elif chunker_type == 'fixed_size':
        return FixedSizeChunker(
            chunk_size=chunker_config.get('chunk_size', 500),
            overlap=chunker_config.get('overlap', 50)
        )
    else:
        return SlidingWindowChunker(
            chunk_size=chunker_config.get('chunk_size', 500),
            chunk_overlap=chunker_config.get('overlap', 50)
        )


def get_embedder(config: Dict[str, Any]):
    """Get embedder based on configuration."""
    embedder_config = config.get('embedder', {})
    embedder_type = embedder_config.get('type', 'sentence_transformer')
    
    if embedder_type == 'openai':
        return OpenAIEmbedder(
            model_name=embedder_config.get('model_name', 'text-embedding-ada-002')
        )
    else:
        return SentenceTransformerEmbedder(
            model_name=embedder_config.get('model_name', 'all-MiniLM-L6-v2')
        )


def get_formatter(config: Dict[str, Any], output_dir: str):
    """Get output formatter based on configuration."""
    formatter_type = config.get('output', {}).get('format', 'pathrag')
    
    if formatter_type == 'vector_db':
        return VectorDBFormatter(
            output_dir=output_dir,
            vector_db_type=config.get('output', {}).get('vector_db_type', 'faiss')
        )
    elif formatter_type == 'huggingface':
        return HuggingFaceDatasetFormatter(output_dir=output_dir)
    else:
        return PathRAGFormatter(output_dir=output_dir)


def collect_documents(config: Dict[str, Any], data_dir: str):
    """Collect documents based on configuration."""
    collection_config = config.get('collection', {})
    
    if collection_config.get('enabled', False):
        collector = AcademicCollector(data_dir)
        
        # Track performance with Arize Phoenix if enabled
        arize_adapter = get_arize_adapter(config)
        collection_start_time = time.time()
        
        # Collect papers based on search terms
        search_terms = collection_config.get('search_terms', [])
        max_papers = collection_config.get('max_papers_per_term', 50)
        collector.collect_arxiv_papers(search_terms, max_papers_per_category=max_papers)
        
        # Track collection performance
        if arize_adapter:
            arize_adapter.track_collection_performance(
                collector_id=f"collection_{int(time.time())}",
                collector_type="academic",
                source="arxiv",
                num_queries=len(search_terms),
                num_documents_found=collector.total_papers_found,
                num_documents_processed=collector.total_papers_downloaded,
                collection_time=time.time() - collection_start_time,
                success=True
            )
            arize_adapter.flush()


def process_documents(config: Dict[str, Any]):
    """Process documents and build the dataset."""
    # Get paths
    data_dir = config.get('data_dir')
    output_dir = config.get('output_dir')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get components based on configuration
    processor = get_processor(config)
    chunker = get_chunker(config)
    embedder = get_embedder(config)
    formatter = get_formatter(config, output_dir)
    
    # Get Arize Phoenix adapter if enabled
    arize_adapter = get_arize_adapter(config)
    
    # Get document paths
    doc_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.txt', '.pdf', '.py', '.js', '.java', '.md')):
                doc_paths.append(os.path.join(root, file))
    
    logger.info(f"Found {len(doc_paths)} documents to process")
    
    # Process each document
    for doc_path in doc_paths:
        doc_id = Path(doc_path).stem
        
        # Track document processing time
        doc_start_time = time.time()
        
        try:
            # Process document
            logger.info(f"Processing document: {doc_path}")
            doc_content, doc_metadata = processor.process(doc_path)
            
            # Track document processing in Arize Phoenix
            if arize_adapter:
                arize_adapter.track_document_processing(
                    document_id=doc_id,
                    document_path=doc_path,
                    document_type=Path(doc_path).suffix[1:],  # Remove dot from extension
                    processing_time=time.time() - doc_start_time,
                    document_size=os.path.getsize(doc_path),
                    metadata=doc_metadata,
                    success=True
                )
            
            # Track chunking time
            chunk_start_time = time.time()
            
            # Chunk document
            logger.info(f"Chunking document: {doc_path}")
            chunks = chunker.chunk_text(doc_content, doc_metadata)
            
            # Track chunking in Arize Phoenix
            if arize_adapter and chunks:
                chunk_sizes = [len(chunk.get('content', '')) for chunk in chunks]
                arize_adapter.track_chunking(
                    document_id=doc_id,
                    chunker_type=chunker.__class__.__name__,
                    num_chunks=len(chunks),
                    chunk_sizes=chunk_sizes,
                    chunking_time=time.time() - chunk_start_time,
                    avg_chunk_size=sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                    success=True
                )
            
            if not chunks:
                logger.warning(f"No chunks generated for document: {doc_path}")
                continue
            
            # Generate embeddings
            logger.info(f"Generating embeddings for document: {doc_path}")
            embed_start_time = time.time()
            texts = [chunk.get('content', '') for chunk in chunks]
            embeddings = embedder.embed_texts(texts)
            
            # Track embedding generation in Arize Phoenix
            if arize_adapter:
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    arize_adapter.track_embedding_generation(
                        chunk_id=f"{doc_id}_chunk_{i}",
                        document_id=doc_id,
                        embedder_type=embedder.__class__.__name__,
                        embedding_model=embedder.model_name,
                        embedding_dimensions=len(embedding),
                        embedding_time=(time.time() - embed_start_time) / len(chunks),
                        embedding=embedding,
                        success=True
                    )
            
            # Format output
            logger.info(f"Formatting output for document: {doc_path}")
            output_start_time = time.time()
            
            # Create document metadata
            document_data = {
                "id": doc_id,
                "metadata": {
                    "filename": os.path.basename(doc_path),
                    "path": doc_path,
                    **doc_metadata
                }
            }
            
            # Format and save
            formatter.format_output(chunks, embeddings, document_data)
            
            # Track output generation in Arize Phoenix
            if arize_adapter:
                total_chunks_size = sum(len(chunk.get('content', '')) for chunk in chunks)
                arize_adapter.track_output_generation(
                    output_id=f"{doc_id}_output",
                    formatter_type=formatter.__class__.__name__,
                    num_documents=1,
                    num_chunks=len(chunks),
                    total_chunks_size=total_chunks_size,
                    output_size=0,  # Not easily calculable, could be estimated
                    processing_time=time.time() - output_start_time,
                    success=True
                )
            
            logger.info(f"Successfully processed document: {doc_path}")
        
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            
            # Track error in Arize Phoenix
            if arize_adapter:
                arize_adapter.track_document_processing(
                    document_id=doc_id,
                    document_path=doc_path,
                    document_type=Path(doc_path).suffix[1:],
                    processing_time=time.time() - doc_start_time,
                    document_size=os.path.getsize(doc_path),
                    metadata={},
                    success=False,
                    error=str(e)
                )
    
    # Flush any remaining Arize Phoenix records
    if arize_adapter:
        arize_adapter.flush()
    
    logger.info("Document processing complete")


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Dataset Builder")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Collect documents if enabled
    collect_documents(config, config.get('data_dir'))
    
    # Process documents
    process_documents(config)


if __name__ == "__main__":
    main()
