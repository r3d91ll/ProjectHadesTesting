#!/usr/bin/env python3
"""
Fixed RAG Dataset Builder - Main Entry Point

A modified version of the main entry script that fixes import issues.
Following the project organization guidelines, this temporary utility
is placed in the dev-utilities directory.
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

# Add rag-dataset-builder src directory to path
rag_builder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag-dataset-builder'))
src_path = os.path.join(rag_builder_path, 'src')
sys.path.insert(0, src_path)

# Import necessary components with absolute imports
# Define abstract base classes required by the implementation
class BaseProcessor:
    def process_document(self, doc_path: str) -> Dict[str, Any]:
        raise NotImplementedError()

class BaseChunker:
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError()

class BaseEmbedder:
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        raise NotImplementedError()

class BaseOutputFormatter:
    def format_output(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], metadata: Dict[str, Any]) -> None:
        raise NotImplementedError()
        
    def finalize(self) -> None:
        """Finalize the output. Called after all documents are processed."""
        pass

# Import actual implementations
from src.builder import BaseProcessor as OrigBaseProcessor, BaseChunker as OrigBaseChunker, BaseEmbedder as OrigBaseEmbedder, BaseOutputFormatter as OrigBaseOutputFormatter
from src.embedders import SentenceTransformerEmbedder, OpenAIEmbedder
from src.formatters import PathRAGFormatter as OrigPathRAGFormatter, VectorDBFormatter as OrigVectorDBFormatter, HuggingFaceDatasetFormatter as OrigHuggingFaceDatasetFormatter

# Create wrapper classes with the expected interface
class PathRAGFormatter(OrigPathRAGFormatter):
    def finalize(self) -> None:
        """Finalize the PathRAG output."""
        # Save the knowledge graph one final time
        try:
            self._save_graph()
            logger.info("Finalized PathRAG output and saved knowledge graph")
        except Exception as e:
            logger.error(f"Error finalizing PathRAG output: {e}")

class VectorDBFormatter(OrigVectorDBFormatter):
    def finalize(self) -> None:
        logger.info("Finalized VectorDB output")
        pass

class HuggingFaceDatasetFormatter(OrigHuggingFaceDatasetFormatter):
    def finalize(self) -> None:
        logger.info("Finalized HuggingFace dataset output")
        pass
from src.processors import SimpleTextProcessor as OrigSimpleTextProcessor, PDFProcessor as OrigPDFProcessor, CodeProcessor as OrigCodeProcessor

# Create wrapper classes with the expected interface
class SimpleTextProcessor(OrigSimpleTextProcessor):
    def process(self, doc_path: str) -> Dict[str, Any]:
        return self.process_document(doc_path)

class PDFProcessor(OrigPDFProcessor):
    def process(self, doc_path: str) -> Dict[str, Any]:
        return self.process_document(doc_path)

class CodeProcessor(OrigCodeProcessor):
    def process(self, doc_path: str) -> Dict[str, Any]:
        return self.process_document(doc_path)
from src.chunkers import SlidingWindowChunker, SemanticChunker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(rag_builder_path, "logs", "rag_dataset_builder.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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
        
        # Set up input and output directories with absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Convert relative paths to absolute paths
        source_documents_config = self.config.get("source_documents_dir", "../source_documents")
        output_dir_config = self.config.get("output_dir", "../rag_databases/current")
        
        # Handle relative paths by converting them to absolute
        if source_documents_config.startswith("../"):
            self.source_documents_dir = os.path.abspath(os.path.join(base_dir, source_documents_config.replace("../", "")))
        else:
            self.source_documents_dir = source_documents_config
            
        if output_dir_config.startswith("../"):
            self.output_dir = os.path.abspath(os.path.join(base_dir, output_dir_config.replace("../", "")))
        else:
            self.output_dir = output_dir_config
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Source documents directory: {self.source_documents_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
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
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_file}")
        return config
    
    def _init_processor(self) -> BaseProcessor:
        """
        Initialize document processor based on configuration.
        
        Returns:
            Document processor
        """
        processor_config = self.config.get("processor", {})
        processor_type = processor_config.get("type", "auto")
        
        if processor_type == "simple_text":
            processor = SimpleTextProcessor()
        elif processor_type == "pdf":
            processor = PDFProcessor(
                extract_metadata=processor_config.get("extract_metadata", True),
                fallback_to_pdfminer=processor_config.get("fallback_to_pdfminer", True)
            )
        elif processor_type == "code":
            processor = CodeProcessor()
        elif processor_type == "auto":
            processor = SimpleTextProcessor()  # Default, will be overridden based on file extension
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")
        
        logger.info(f"Initialized processor: {processor.__class__.__name__}")
        return processor
    
    def _init_chunker(self) -> BaseChunker:
        """
        Initialize text chunker based on configuration.
        
        Returns:
            Text chunker
        """
        chunker_config = self.config.get("chunker", {})
        chunker_type = chunker_config.get("type", "sliding_window")
        
        if chunker_type == "sliding_window":
            chunker = SlidingWindowChunker(
                chunk_size=chunker_config.get("chunk_size", 500),
                chunk_overlap=chunker_config.get("chunk_overlap", 100),
                respect_sentences=chunker_config.get("respect_sentences", True)
            )
        elif chunker_type == "semantic":
            chunker = SemanticChunker(
                max_chunk_size=chunker_config.get("chunk_size", 300),
                min_chunk_size=chunker_config.get("min_chunk_size", 100)
            )
        else:
            raise ValueError(f"Unsupported chunker type: {chunker_type}")
        
        logger.info(f"Initialized chunker: {chunker.__class__.__name__}")
        return chunker
    
    def _init_embedder(self) -> BaseEmbedder:
        """
        Initialize embedder based on configuration.
        
        Returns:
            Text embedder
        """
        embedder_config = self.config.get("embedder", {})
        embedder_type = embedder_config.get("type", "sentence_transformer")
        
        if embedder_type == "sentence_transformer":
            embedder = SentenceTransformerEmbedder(
                model_name=embedder_config.get("model_name", "all-MiniLM-L6-v2"),
                batch_size=embedder_config.get("batch_size", 32),
                use_gpu=embedder_config.get("use_gpu", True)
            )
        elif embedder_type == "openai":
            embedder = OpenAIEmbedder(
                model_name=embedder_config.get("model_name", "text-embedding-3-small"),
                batch_size=embedder_config.get("batch_size", 100)
            )
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")
        
        logger.info(f"Initialized embedder: {embedder.__class__.__name__}")
        return embedder
    
    def _init_formatter(self) -> BaseOutputFormatter:
        """
        Initialize output formatter based on configuration.
        
        Returns:
            Output formatter
        """
        output_config = self.config.get("output", {})
        formats_config = output_config.get("formats", {})
        
        if formats_config.get("pathrag", {}).get("enabled", False):
            formatter = PathRAGFormatter(
                output_dir=self.output_dir
            )
            # Note: PathRAGFormatter in the actual implementation doesn't accept these parameters
            # They are read from the config but not passed to the formatter
        elif formats_config.get("vector_db", {}).get("enabled", False):
            formatter = VectorDBFormatter(
                output_dir=self.output_dir,
                vector_db_type=formats_config.get("vector_db", {}).get("type", "faiss"),
                include_metadata=formats_config.get("vector_db", {}).get("include_metadata", True)
            )
        elif formats_config.get("huggingface", {}).get("enabled", False):
            formatter = HuggingFaceDatasetFormatter(
                output_dir=self.output_dir,
                dataset_name=formats_config.get("huggingface", {}).get("dataset_name", "rag_dataset"),
                include_embeddings=formats_config.get("huggingface", {}).get("include_embeddings", True),
                include_metadata=formats_config.get("huggingface", {}).get("include_metadata", True)
            )
        else:
            # Default to PathRAG format
            formatter = PathRAGFormatter(
                output_dir=self.output_dir
            )
        
        logger.info(f"Initialized formatter: {formatter.__class__.__name__}")
        return formatter
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint from disk if it exists.
        
        Returns:
            Checkpoint data
        """
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            return checkpoint
        
        return {
            "processed_documents": [],
            "last_updated": time.time()
        }
    
    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        self.checkpoint["last_updated"] = time.time()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint to {self.checkpoint_file}")
    
    def find_all_documents(self) -> List[str]:
        """
        Find all documents in the data directory based on configuration.
        
        Returns:
            List of document paths
        """
        input_config = self.config.get("input", {})
        include_patterns = input_config.get("include", ["**/*.txt", "**/*.pdf", "**/*.md"])
        exclude_patterns = input_config.get("exclude", ["**/.*", "**/node_modules/**"])
        
        # Make sure the source documents directory exists
        if not os.path.exists(self.source_documents_dir):
            logger.warning(f"Source documents directory not found: {self.source_documents_dir}")
            return []
        
        # Find all documents matching the patterns
        all_docs = []
        for pattern in include_patterns:
            for path in Path(self.source_documents_dir).glob(pattern):
                if path.is_file():
                    all_docs.append(str(path))
        
        # Filter out excluded files
        filtered_docs = []
        for doc_path in all_docs:
            excluded = False
            for pattern in exclude_patterns:
                if Path(doc_path).match(pattern):
                    excluded = True
                    break
            
            if not excluded:
                filtered_docs.append(doc_path)
        
        # Filter out already processed documents if in incremental mode
        if self.config.get("processing", {}).get("incremental", True):
            processed_docs = set(self.checkpoint.get("processed_documents", []))
            filtered_docs = [doc for doc in filtered_docs if doc not in processed_docs]
        
        logger.info(f"Found {len(filtered_docs)} documents to process")
        return filtered_docs
    
    def process_document(self, doc_path: str):
        """
        Process a single document.
        
        Args:
            doc_path: Path to the document
        """
        logger.info(f"Processing document: {doc_path}")
        
        try:
            # Process document based on its type
            doc_extension = os.path.splitext(doc_path)[1].lower()
            
            # Select processor based on file extension
            processor = self.processor
            processor_config = self.config.get("processor", {})
            if processor_config.get("type", "auto") == "auto":
                if doc_extension == '.pdf':
                    processor = PDFProcessor(
                        extract_metadata=processor_config.get("extract_metadata", True),
                        fallback_to_pdfminer=processor_config.get("fallback_to_pdfminer", True)
                    )
                elif doc_extension in ['.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp']:
                    processor = CodeProcessor()
                else:
                    processor = SimpleTextProcessor()
            
            # Process the document
            processed_doc = processor.process(doc_path)
            
            # Extract text and metadata from the processed document
            document_text = processed_doc.get('text', '')
            metadata = processed_doc.get('metadata', {})
            
            # Chunk the document
            chunks = self.chunker.chunk_document({'text': document_text, 'metadata': metadata})
            
            # Generate embeddings
            embeddings = self.embedder.embed_chunks(chunks)
            
            # Save to output format
            self.formatter.format_output(chunks, embeddings, processed_doc)
            
            # Add to processed documents
            self.checkpoint["processed_documents"].append(doc_path)
            self._save_checkpoint()
            
            logger.info(f"Successfully processed document: {doc_path}")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
    
    def process_documents_in_batches(self, document_paths: List[str]):
        """
        Process documents in batches to limit memory usage.
        
        Args:
            document_paths: List of document paths
        """
        batch_size = self.config.get("processing", {}).get("batch_size", 10)
        
        for i in range(0, len(document_paths), batch_size):
            batch = document_paths[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{len(document_paths) // batch_size + 1}")
            
            for doc_path in batch:
                self.process_document(doc_path)
    
    def build_dataset(self):
        """Build the RAG dataset."""
        logger.info("Starting RAG dataset build process")
        
        # Find all documents
        document_paths = self.find_all_documents()
        
        if not document_paths:
            logger.warning("No documents found to process")
            return
        
        # Process documents in batches
        self.process_documents_in_batches(document_paths)
        
        # Finalize the dataset
        self.formatter.finalize()
        
        logger.info("Completed RAG dataset build process")


def main(config_path: Optional[str] = None):
    """
    Main function.
    
    Args:
        config_path: Path to configuration file (optional)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Dataset Builder")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Use provided config path, command line argument, or default
    if config_path:
        config_file = config_path
    elif args.config:
        config_file = args.config
    else:
        config_file = os.path.join(rag_builder_path, "config", "config.yaml")
    
    # Create RAG dataset builder
    builder = RAGDatasetBuilder(config_file)
    
    # Build the dataset
    builder.build_dataset()


if __name__ == "__main__":
    main()
