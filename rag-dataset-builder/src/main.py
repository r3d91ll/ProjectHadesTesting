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
from src.builder import BaseProcessor, BaseChunker, BaseEmbedder, BaseOutputFormatter
from src.embedders import SentenceTransformerEmbedder, OpenAIEmbedder
from src.formatters import PathRAGFormatter, VectorDBFormatter, HuggingFaceDatasetFormatter
from src.processors import SimpleTextProcessor, PDFProcessor, CodeProcessor
from src.chunkers import SlidingWindowChunker, SemanticChunker
# Import collectors
from src.collectors import SemanticScholarCollector, PubMedCollector, SocArXivCollector
from src.collectors.academic_collector import AcademicCollector
# Import Arize Phoenix integration
from src.utils.arize_integration import get_arize_adapter

# Set up logging
# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "rag_dataset_builder.log")),
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
        self.data_dir = self.config.get("source_documents_dir", "../source_documents")
        self.output_dir = self.config.get("output_dir", "../rag_databases/current")
        
        # Convert relative paths to absolute if needed
        # Handle paths relative to the script execution location
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(script_dir)
        
        if self.data_dir.startswith("/"):
            # Absolute path, leave as is
            pass
        elif self.data_dir.startswith("../"):
            # Path relative to rag-dataset-builder directory
            relative_path = self.data_dir.replace("../", "")
            self.data_dir = os.path.join(project_root, relative_path)
        elif self.data_dir.startswith("./"):
            # Path relative to rag-dataset-builder directory
            relative_path = self.data_dir.replace("./", "")
            self.data_dir = os.path.join(script_dir, relative_path)
        else:
            # Default to project root if no prefix
            self.data_dir = os.path.join(project_root, self.data_dir)
        
        if self.output_dir.startswith("/"):
            # Absolute path, leave as is
            pass
        elif self.output_dir.startswith("../"):
            # Path relative to rag-dataset-builder directory
            relative_path = self.output_dir.replace("../", "")
            self.output_dir = os.path.join(project_root, relative_path)
        elif self.output_dir.startswith("./"):
            # Path relative to rag-dataset-builder directory
            relative_path = self.output_dir.replace("./", "")
            self.output_dir = os.path.join(script_dir, relative_path)
        else:
            # Default to project root if no prefix
            self.output_dir = os.path.join(project_root, self.output_dir)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.processor = self._init_processor()
        self.chunker = self._init_chunker()
        self.embedder = self._init_embedder()
        self.formatter = self._init_formatter()
        
        # Initialize performance tracking with Arize Phoenix
        self.arize_adapter = get_arize_adapter(self.config)
        if self.arize_adapter:
            logger.info(f"Arize Phoenix integration enabled for project: {self.arize_adapter.project_name}")
        
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
            extract_metadata = processor_config.get("extract_metadata", True)
            fallback_to_pdfminer = processor_config.get("fallback_to_pdfminer", True)
            return PDFProcessor(extract_metadata=extract_metadata, fallback_to_pdfminer=fallback_to_pdfminer)
        elif processor_type == "code":
            return CodeProcessor()
        elif processor_type == "auto":
            # For auto type, we'll use SimpleTextProcessor as the default but will
            # determine the processor type during document processing
            logger.info("Using 'auto' processor type - will select appropriate processor based on file extension")
            return SimpleTextProcessor()
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
        start_time = time.time()
        doc_size = os.path.getsize(doc_path) if os.path.exists(doc_path) else 0
        doc_type = os.path.splitext(doc_path)[1].lstrip('.').lower() if '.' in doc_path else 'unknown'
        doc_id = None
        success = False
        error_msg = None
        metadata = {
            "path": doc_path,
            "type": doc_type,
            "size": doc_size
        }
        
        try:
            logger.info(f"Processing document: {doc_path}")
            
            # Process document
            doc_data = self.processor.process_document(doc_path)
            
            # Skip if processing failed
            if not doc_data or not doc_data.get("text"):
                logger.warning(f"Failed to process document: {doc_path}")
                return
                
            # Ensure document has an ID
            if "id" not in doc_data:
                # Generate a document ID based on its path if missing
                import hashlib
                doc_data["id"] = hashlib.md5(doc_path.encode()).hexdigest()
                logger.info(f"Generated ID for document: {doc_path}")
                
            # Ensure metadata exists
            if "metadata" not in doc_data:
                doc_data["metadata"] = {}
            
            # Move ID into metadata for chunker compatibility
            if "id" in doc_data:
                doc_data["metadata"]["id"] = doc_data["id"]
                
            # Measure chunking time
            chunk_start_time = time.time()
            
            # Generate chunks using the appropriate method
            if hasattr(self.chunker, 'chunk_text'):
                chunks = self.chunker.chunk_text(doc_data["text"], doc_data["metadata"])
            elif hasattr(self.chunker, 'chunk_document'):
                # Pass the whole doc_data, assuming chunk_document handles it or expects id in metadata
                chunk_input = {"text": doc_data["text"], "metadata": doc_data["metadata"]}
                chunks = self.chunker.chunk_document(chunk_input)
            else:
                raise AttributeError(f"Chunker {self.chunker.__class__.__name__} has no chunk_text or chunk_document method")
            
            # Calculate chunking time
            chunking_time = time.time() - chunk_start_time
            logger.info(f"Chunking completed in {chunking_time:.2f}s")
            
            # Skip if chunking failed
            if not chunks:
                logger.warning(f"No chunks generated from document: {doc_path}")
                return
                
            # Track chunking performance if Arize adapter is enabled
            if self.arize_adapter:
                # Calculate chunk sizes
                chunk_sizes = [len(chunk["text"]) for chunk in chunks]
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
                
                self.arize_adapter.track_chunking(
                    document_id=doc_data["id"],
                    chunker_type=self.chunker.__class__.__name__,
                    num_chunks=len(chunks),
                    chunk_sizes=chunk_sizes,
                    chunking_time=chunking_time,
                    avg_chunk_size=avg_chunk_size,
                    success=True
                )
            
            # Generate embeddings with timing
            embedding_start_time = time.time()
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(chunk_texts)
            embedding_time = time.time() - embedding_start_time
            logger.info(f"Embedding generation completed in {embedding_time:.2f}s for {len(chunk_texts)} chunks")
            
            # Track embedding performance if Arize adapter is enabled
            if self.arize_adapter and embeddings:
                # Track a sample of embeddings (first 5)
                for i, (chunk, embedding) in enumerate(zip(chunks[:5], embeddings[:5])):
                    # Generate a chunk ID
                    chunk_id = f"{doc_data['id']}_chunk_{i}"
                    
                    self.arize_adapter.track_embedding_generation(
                        chunk_id=chunk_id,
                        document_id=doc_data["id"],
                        embedder_type=self.embedder.__class__.__name__,
                        embedding_model=getattr(self.embedder, 'model_name', type(self.embedder).__name__),
                        embedding_dimensions=len(embedding),
                        embedding_time=embedding_time / len(embeddings),  # Average time per embedding
                        embedding=embedding,
                        success=True
                    )
            
            # Format and save output
            self.formatter.format_output(chunks, embeddings, doc_data)
            
            # Update checkpoint
            self.checkpoint["processed_files"].append(doc_path)
            self.checkpoint["processed_chunks"] += len(chunks)
            self.checkpoint["total_chunks"] += len(chunks)
            self._save_checkpoint()
            
            logger.info(f"Successfully processed document: {doc_path}")
            
            # Document processed successfully
            success = True
            doc_id = doc_data.get("id", None)
            if not doc_id and "metadata" in doc_data and "id" in doc_data["metadata"]:
                doc_id = doc_data["metadata"]["id"]
                
            # Add chunking and embedding metadata
            metadata.update({
                "chunks_count": len(chunks),
                "embeddings_count": len(embeddings),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "embedding_model": self.embedder.__class__.__name__,
                "chunker_type": self.chunker.__class__.__name__
            })
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing document {doc_path}: {e}")
        
        # Track document processing performance with Arize Phoenix
        if self.arize_adapter:
            processing_time = time.time() - start_time
            # Use hash of path as document ID if none was generated
            if not doc_id:
                import hashlib
                doc_id = hashlib.md5(doc_path.encode()).hexdigest()
                
            self.arize_adapter.track_document_processing(
                document_id=doc_id,
                document_path=doc_path,
                document_type=doc_type,
                processing_time=processing_time,
                document_size=doc_size,
                metadata=metadata,
                success=success,
                error=error_msg
            )
    
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
    
    def _run_collectors(self) -> None:
        """Initialize and run academic collectors based on config."""
        collection_config = self.config.get("collection", {})
        if not collection_config.get("enabled", False):
            logger.info("Academic collection is disabled in the configuration.")
            return

        logger.info("Starting academic paper collection...")
        sources_config = collection_config.get("sources", {})
        domains_config = collection_config.get("domains", {})
        max_papers_per_term = collection_config.get("max_papers_per_term", 10)

        # Initialize collectors
        active_collectors = []
        
        # Check for arxiv specifically, since it uses the AcademicCollector
        if sources_config.get("arxiv", {}).get("enabled", False):
            try:
                # Create a temporary config file for the AcademicCollector
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
                    # Create a simplified config with just search terms
                    arxiv_config = {
                        "search_terms": {}
                    }
                    
                    # Add search terms from domains
                    for domain_name, domain_config in domains_config.items():
                        if domain_config.get("enabled", True):
                            arxiv_config["search_terms"][domain_name] = domain_config.get("search_terms", [])
                    
                    # Write config to temp file
                    yaml.dump(arxiv_config, temp_config)
                    temp_config_path = temp_config.name
                
                # Initialize academic collector
                academic_collector = AcademicCollector(temp_config_path, self.data_dir)
                
                # Run the ArXiv collection
                max_papers = sources_config.get("arxiv", {}).get("max_papers_per_category", 10)
                academic_collector.collect_arxiv_papers(max_papers_per_category=max_papers)
                
                # Clean up temp file
                try:
                    os.remove(temp_config_path)
                except:
                    pass
                
                logger.info(f"Completed ArXiv collection")
            except Exception as e:
                logger.error(f"Error during ArXiv collection: {e}")
                # Print the full error traceback for debugging
                import traceback
                logger.error(traceback.format_exc())
        
        # Process other collectors
        for source_name, source_config in sources_config.items():
            if source_name == "arxiv":
                # Skip arxiv as it's handled separately above
                continue
                
            if source_config.get("enabled", False):
                if source_name == "semantic_scholar":
                    try:
                        api_key = source_config.get("api_key")
                        collector_instance = SemanticScholarCollector(self.data_dir, deduplicator=None, api_key=api_key)
                        active_collectors.append(collector_instance)
                        logger.info(f"Initialized collector: {source_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize collector {source_name}: {e}")
                elif source_name == "pubmed":
                    try:
                        email = source_config.get("email")
                        collector_instance = PubMedCollector(self.data_dir, deduplicator=None, email=email)
                        active_collectors.append(collector_instance)
                        logger.info(f"Initialized collector: {source_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize collector {source_name}: {e}")
                elif source_name == "socarxiv":
                    try:
                        collector_instance = SocArXivCollector(self.data_dir, deduplicator=None)
                        active_collectors.append(collector_instance)
                        logger.info(f"Initialized collector: {source_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize collector {source_name}: {e}")
                else:
                    logger.warning(f"Collector for source '{source_name}' not found or implemented.")

        if not active_collectors:
            logger.warning("No active collectors were initialized.")
            return

        total_downloaded = 0
        for domain_name, domain_config in domains_config.items():
            if domain_config.get("enabled", True):
                search_terms = domain_config.get("search_terms", [])
                if not search_terms:
                    logger.warning(f"No search terms found for enabled domain: {domain_name}")
                    continue
                
                logger.info(f"Collecting papers for domain: {domain_name} using {len(search_terms)} terms")
                for collector in active_collectors:
                    try:
                        # Call the collect_papers method with the search terms list
                        download_count = collector.collect_papers(search_terms, max_papers_per_term)
                        total_downloaded += download_count
                        logger.info(f"Collector {collector.__class__.__name__} downloaded {download_count} papers for domain {domain_name}.")
                    except AttributeError as ae:
                        logger.error(f"Collector {collector.__class__.__name__} does not have a 'collect_papers' method: {ae}")
                    except Exception as e:
                        logger.error(f"Error during collection with {collector.__class__.__name__} for domain {domain_name}: {e}")
                        # Print the full error traceback for debugging
                        import traceback
                        logger.error(traceback.format_exc())
        
        logger.info(f"Academic paper collection finished. Total papers downloaded across all domains: {total_downloaded}")

    def build_dataset(self) -> None:
        """Build the RAG dataset."""
        logger.info("Starting RAG dataset build")
        
        # Run collectors first to download documents
        self._run_collectors()
        
        # Find all documents (including newly downloaded ones)
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


def main(config_path=None):
    """Main function."""
    if config_path is None:
        parser = argparse.ArgumentParser(description="Build RAG dataset with configurable components")
        parser.add_argument("--config", type=str, required=True,
                          help="Path to configuration file (YAML or JSON)")
        args = parser.parse_args()
        config_path = args.config
        
    # Log key information
    logger.info(f"Using config file: {config_path}")
    
    # Build dataset
    builder = RAGDatasetBuilder(config_file=config_path)
    builder.build_dataset()


if __name__ == "__main__":
    main()
