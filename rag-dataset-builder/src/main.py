#!/usr/bin/env python3
"""
RAG Dataset Builder - Main Entry Point

A flexible, memory-efficient tool for building datasets for 
Retrieval-Augmented Generation (RAG) systems.
"""

import os
import sys
import json
import time
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import necessary components from our module
from builder import BaseProcessor, BaseChunker, BaseEmbedder, BaseOutputFormatter
from embedders import SentenceTransformerEmbedder, OpenAIEmbedder
from formatters import PathRAGFormatter, VectorDBFormatter, HuggingFaceDatasetFormatter
from processors import SimpleTextProcessor, PDFProcessor, CodeProcessor
from chunkers import SlidingWindowChunker, SemanticChunker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_dataset_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_dataset_builder")


class RAGDatasetBuilder:
    """Main class for building RAG datasets."""
    
    def __init__(self, config_file: str):
        """
        Initialize the RAG dataset builder.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config(config_file)
        
        # Set up input and output directories
        self.data_dir = self.config.get("data_dir", "./data")
        self.output_dir = self.config.get("output_dir", "./output")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.processor = self._init_processor()
        self.chunker = self._init_chunker()
        self.embedder = self._init_embedder()
        self.formatter = self._init_formatter()
        
        # Load checkpoint if it exists
        self.checkpoint_file = os.path.join(self.output_dir, "checkpoint.json")
        self.checkpoint = self._load_checkpoint()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _init_processor(self) -> BaseProcessor:
        """
        Initialize document processor based on configuration.
        
        Returns:
            Document processor
        """
        processor_config = self.config.get("processor", {})
        processor_type = processor_config.get("type", "simple_text")
        
        if processor_type == "simple_text":
            return SimpleTextProcessor()
        elif processor_type == "pdf":
            return PDFProcessor()
        elif processor_type == "code":
            return CodeProcessor()
        else:
            logger.error(f"Unknown processor type: {processor_type}")
            sys.exit(1)
    
    def _init_chunker(self) -> BaseChunker:
        """
        Initialize text chunker based on configuration.
        
        Returns:
            Text chunker
        """
        chunker_config = self.config.get("chunker", {})
        chunker_type = chunker_config.get("type", "sliding_window")
        
        if chunker_type == "sliding_window":
            chunk_size = chunker_config.get("chunk_size", 300)
            chunk_overlap = chunker_config.get("chunk_overlap", 50)
            return SlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif chunker_type == "semantic":
            return SemanticChunker()
        else:
            logger.error(f"Unknown chunker type: {chunker_type}")
            sys.exit(1)
    
    def _init_embedder(self) -> BaseEmbedder:
        """
        Initialize embedder based on configuration.
        
        Returns:
            Text embedder
        """
        embedder_config = self.config.get("embedder", {})
        embedder_type = embedder_config.get("type", "sentence_transformer")
        
        if embedder_type == "sentence_transformer":
            model_name = embedder_config.get("model_name", "all-MiniLM-L6-v2")
            batch_size = embedder_config.get("batch_size", 32)
            use_gpu = embedder_config.get("use_gpu", True)
            return SentenceTransformerEmbedder(model_name=model_name, batch_size=batch_size, use_gpu=use_gpu)
        elif embedder_type == "openai":
            model_name = embedder_config.get("model_name", "text-embedding-3-small")
            batch_size = embedder_config.get("batch_size", 100)
            return OpenAIEmbedder(model_name=model_name, batch_size=batch_size)
        else:
            logger.error(f"Unknown embedder type: {embedder_type}")
            sys.exit(1)
    
    def _init_formatter(self) -> BaseOutputFormatter:
        """
        Initialize output formatter based on configuration.
        
        Returns:
            Output formatter
        """
        formatter_config = self.config.get("formatter", {})
        formatter_type = formatter_config.get("type", "pathrag")
        
        if formatter_type == "pathrag":
            return PathRAGFormatter(output_dir=self.output_dir)
        elif formatter_type == "vector_db":
            vector_db_type = formatter_config.get("vector_db_type", "faiss")
            return VectorDBFormatter(output_dir=self.output_dir, vector_db_type=vector_db_type)
        elif formatter_type == "huggingface":
            return HuggingFaceDatasetFormatter(output_dir=self.output_dir)
        else:
            logger.error(f"Unknown formatter type: {formatter_type}")
            sys.exit(1)
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint from disk if it exists.
        
        Returns:
            Checkpoint data
        """
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted checkpoint file. Starting fresh.")
        
        # Initialize new checkpoint
        return {
            "processed_files": [],
            "processed_chunks": 0,
            "total_chunks": 0,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        self.checkpoint["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def find_all_documents(self) -> List[str]:
        """
        Find all documents in the data directory based on configuration.
        
        Returns:
            List of document paths
        """
        include_patterns = self.config.get("input", {}).get("include", ["**/*.txt", "**/*.pdf"])
        exclude_patterns = self.config.get("input", {}).get("exclude", [])
        
        document_paths = []
        
        # Support glob patterns for file discovery
        for pattern in include_patterns:
            matched_files = list(Path(self.data_dir).glob(pattern))
            document_paths.extend([str(f) for f in matched_files])
        
        # Apply exclusion patterns
        for pattern in exclude_patterns:
            exclusion_files = list(Path(self.data_dir).glob(pattern))
            exclusion_paths = [str(f) for f in exclusion_files]
            document_paths = [p for p in document_paths if p not in exclusion_paths]
        
        # Filter out already processed files
        document_paths = [
            path for path in document_paths 
            if path not in self.checkpoint["processed_files"]
        ]
        
        logger.info(f"Found {len(document_paths)} unprocessed documents")
        return document_paths
    
    def process_document(self, doc_path: str) -> None:
        """
        Process a single document.
        
        Args:
            doc_path: Path to the document
        """
        try:
            logger.info(f"Processing document: {doc_path}")
            
            # Process document
            doc_data = self.processor.process_document(doc_path)
            
            # Skip if processing failed
            if not doc_data or not doc_data.get("text"):
                logger.warning(f"Failed to process document: {doc_path}")
                return
            
            # Generate chunks
            chunks = self.chunker.chunk_text(doc_data["text"], doc_data["metadata"])
            logger.info(f"Generated {len(chunks)} chunks from {doc_path}")
            
            # Skip if no chunks were generated
            if not chunks:
                logger.warning(f"No chunks generated from document: {doc_path}")
                return
            
            # Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(chunk_texts)
            
            # Format and save output
            self.formatter.format_output(chunks, embeddings, doc_data)
            
            # Update checkpoint
            self.checkpoint["processed_files"].append(doc_path)
            self.checkpoint["processed_chunks"] += len(chunks)
            self.checkpoint["total_chunks"] += len(chunks)
            self._save_checkpoint()
            
            logger.info(f"Successfully processed document: {doc_path}")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
    
    def process_documents_in_batches(self, document_paths: List[str]) -> None:
        """
        Process documents in batches to limit memory usage.
        
        Args:
            document_paths: List of document paths
        """
        batch_size = self.config.get("processing", {}).get("batch_size", 5)
        total_documents = len(document_paths)
        
        for i in range(0, total_documents, batch_size):
            batch = document_paths[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_documents + batch_size - 1)//batch_size}")
            
            for doc_path in batch:
                self.process_document(doc_path)
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
    
    def build_dataset(self) -> None:
        """Build the RAG dataset."""
        logger.info("Starting RAG dataset build")
        
        # Find all documents
        document_paths = self.find_all_documents()
        
        if not document_paths:
            logger.info("No new documents to process")
            return
        
        # Process documents in batches
        self.process_documents_in_batches(document_paths)
        
        # Final save
        self._save_checkpoint()
        
        logger.info(f"Dataset build complete. Processed {len(self.checkpoint['processed_files'])} documents "
                  f"with {self.checkpoint['total_chunks']} chunks.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build RAG dataset with configurable components")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to configuration file (YAML or JSON)")
    args = parser.parse_args()
    
    # Build dataset
    builder = RAGDatasetBuilder(config_file=args.config)
    builder.build_dataset()


if __name__ == "__main__":
    main()
