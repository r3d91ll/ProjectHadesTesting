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
from .builder import BaseProcessor, BaseChunker, BaseEmbedder, BaseOutputFormatter
from .embedders import SentenceTransformerEmbedder, OpenAIEmbedder, OllamaEmbedder
from .formatters import PathRAGFormatter, VectorDBFormatter, HuggingFaceDatasetFormatter
from .processors import SimpleTextProcessor, PDFProcessor, CodeProcessor
from .chunkers import SlidingWindowChunker, SemanticChunker
# Import collectors
from .collectors import SemanticScholarCollector, PubMedCollector, SocArXivCollector
from .collectors.academic_collector import AcademicCollector
# Import Arize Phoenix integration
from .utils.arize_integration import get_arize_adapter

# Set up logging
# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Initial basic logging setup - will be replaced by config-based setup later
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "rag_dataset_builder.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_dataset_builder")

# Note: This basic configuration will be replaced when the config is loaded
# The actual logging configuration comes from the config.d/12-logging.yaml file


class RAGDatasetBuilder:
    """Main class for building RAG datasets."""
    
    def __init__(self, config_file: str = None, config: Dict[str, Any] = None):
        """
        Initialize the RAG dataset builder.
        
        Args:
            config_file: Path to configuration file
            config: Configuration dictionary (alternative to config_file)
        """
        self.config_file = config_file
        
        if config is not None:
            # Use provided config dictionary
            self.config = config
        elif config_file is not None:
            # Load config from file
            self.config = self._load_config(config_file)
        else:
            raise ValueError("Either config_file or config must be provided")
        
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
        elif embedder_type == "ollama":
            # First check if there's a specific ollama config in the embedders section
            ollama_config = self.config.get("embedders", {}).get("ollama", {})
            
            # Use the specific ollama config if available, otherwise fall back to embedder config
            model_name = ollama_config.get("model_name", embedder_config.get("model_name", "tinyllama"))
            host = ollama_config.get("host", embedder_config.get("host", "localhost"))
            port = ollama_config.get("port", embedder_config.get("port", 11434))
            batch_size = ollama_config.get("batch_size", embedder_config.get("batch_size", 16))
            max_workers = ollama_config.get("max_workers", 8)
            
            # Get GPU usage setting from config
            use_gpu = ollama_config.get("use_gpu", embedder_config.get("use_gpu", True))
            
            logger.info(f"Using Ollama embedder with model {model_name} at {host}:{port} with {max_workers} workers")
            logger.info(f"Ollama GPU usage: {'Enabled' if use_gpu else 'Disabled (using CPU)'}")
            
            return OllamaEmbedder(model_name=model_name, host=host, port=port, 
                                 batch_size=batch_size, max_workers=max_workers, use_gpu=use_gpu)
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
            
    def _filter_by_enabled_domains(self, document_paths: List[str]) -> List[str]:
        """
        Filter documents based on enabled domains in the configuration.
        
        Args:
            document_paths: List of document paths to filter
            
        Returns:
            Filtered list of document paths
        """
        # Check if we should filter by domains
        if not self.config.get("filter_by_domains", True):
            logger.info("Domain filtering disabled, processing all documents")
            return document_paths
            
        # Get domains configuration
        domains_config = {}
        
        # Check in academic.domains section
        academic_config = self.config.get("collection", {}).get("academic", {})
        if "domains" in academic_config and isinstance(academic_config["domains"], dict):
            domains_config.update(academic_config["domains"])
            
        # Check in top-level domains section
        if "domains" in self.config and isinstance(self.config["domains"], dict):
            domains_config.update(self.config["domains"])
            
        # If no domains found, return all documents
        if not domains_config:
            logger.warning("No domains configuration found, processing all documents")
            return document_paths
            
        # Get enabled domains
        enabled_domains = []
        for domain_name, domain_config in domains_config.items():
            if isinstance(domain_config, dict) and domain_config.get("enabled", True):
                enabled_domains.append(domain_name)
                
        if not enabled_domains:
            logger.warning("No enabled domains found, processing all documents")
            return document_paths
            
        logger.info(f"Filtering documents for enabled domains: {', '.join(enabled_domains)}")
        
        # Filter documents by domain
        filtered_paths = []
        for doc_path in document_paths:
            # Extract domain from path
            path_parts = Path(doc_path).parts
            
            # Check if any enabled domain is in the path
            for domain in enabled_domains:
                domain_path = domain.replace('_', '-')  # Convert underscores to hyphens for path matching
                if any(domain in part or domain_path in part for part in path_parts):
                    filtered_paths.append(doc_path)
                    break
                    
        logger.info(f"Filtered {len(document_paths)} documents to {len(filtered_paths)} based on enabled domains")
        return filtered_paths
    
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
            
        # Filter documents based on enabled domains
        document_paths = self._filter_by_enabled_domains(document_paths)
        
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
        # Get the collection configuration
        collection_config = self.config.get("collection", {})
        
        # CRITICAL: Check the master switch first - if collection is disabled, don't proceed
        # This is the top-level switch that controls all document collection
        collection_enabled = collection_config.get("enabled", False)
        
        # Log the status clearly for debugging
        logger.info(f"COLLECTION MASTER SWITCH STATUS: {collection_enabled}")
        
        # If collection is disabled, skip all collection steps
        if not collection_enabled:
            logger.info("Academic collection is DISABLED in the configuration (master switch is off).")
            logger.info("No documents will be downloaded. Skipping all collection steps.")
            return
        
        # If we get here, collection is enabled
        logger.info(f"Collection ENABLED: {collection_enabled} (master switch is on)")

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
                    # Create a simplified config with the proper structure
                    arxiv_config = {
                        "domains": {},
                        "collection": {
                            "enabled": True,  # Always set to True here since we already checked the master switch
                            "max_papers_per_term": collection_config.get("max_papers_per_term", 0),
                            "max_documents_per_category": collection_config.get("max_documents_per_category", 0),
                            "download_delay": collection_config.get("download_delay", 1.0),
                            "max_download_size_mb": collection_config.get("max_download_size_mb", 500)
                        }
                    }
                    
                    # Log whether collection is enabled or disabled
                    logger.info(f"Collection enabled: {collection_config.get('enabled', False)}")
                    
                    # Log collection settings for debugging
                    logger.info(f"Collection settings: max_papers_per_term={collection_config.get('max_papers_per_term', 0)}, "
                               f"max_documents_per_category={collection_config.get('max_documents_per_category', 0)}")
                    
                    # Debug the domains configuration
                    logger.info(f"Domains config type: {type(domains_config)}, keys: {list(domains_config.keys()) if isinstance(domains_config, dict) else 'Not a dict'}")
                    
                    # Add search terms from domains with proper structure
                    enabled_domains = []
                    
                    # If domains_config is not a dictionary or is empty, try to load it directly from the domains file
                    if not isinstance(domains_config, dict) or not domains_config:
                        logger.warning("Domains config is not properly loaded, attempting to load directly from config.d/19-domains.yaml")
                        # Try to find the domains file in the temp config directory
                        domains_file = None
                        
                        # Check if we have a config_dir attribute
                        if hasattr(self, 'config_dir') and self.config_dir:
                            domains_file = os.path.join(self.config_dir, "19-domains.yaml")
                        else:
                            # Try to find the domains file in the original config.d directory
                            original_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.d")
                            domains_file = os.path.join(original_config_dir, "19-domains.yaml")
                            # Also check the temp directory
                            temp_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.d", "temp")
                            if os.path.exists(os.path.join(temp_config_dir, "19-domains.yaml")):
                                domains_file = os.path.join(temp_config_dir, "19-domains.yaml")
                        
                        logger.info(f"Looking for domains file at: {domains_file}")
                        
                        if domains_file and os.path.exists(domains_file):
                            try:
                                with open(domains_file, 'r') as f:
                                    domains_data = yaml.safe_load(f)
                                    if isinstance(domains_data, dict) and 'domains' in domains_data:
                                        domains_config = domains_data['domains']
                                        logger.info(f"Loaded domains directly from file: {list(domains_config.keys())}")
                                    else:
                                        logger.warning(f"Domains file does not contain 'domains' key: {list(domains_data.keys()) if isinstance(domains_data, dict) else 'Not a dict'}")
                            except Exception as e:
                                logger.error(f"Error loading domains file: {e}")
                        else:
                            logger.error(f"Domains file not found at {domains_file}")
                    
                    # Now process the domains
                    for domain_name, domain_config in domains_config.items():
                        if domain_config.get("enabled", True):
                            search_terms = domain_config.get("search_terms", [])
                            if search_terms:  # Only add domains that have search terms
                                arxiv_config["domains"][domain_name] = {
                                    "enabled": True,
                                    "search_terms": search_terms
                                }
                                enabled_domains.append(domain_name)
                                # Log the search terms for debugging
                                logger.info(f"Domain '{domain_name}' has {len(search_terms)} search terms: {search_terms[:3]}...")
                            else:
                                logger.warning(f"Domain '{domain_name}' is enabled but has no search terms")
                    
                    # Log the domains being used
                    logger.info(f"Using domains for ArXiv collection: {enabled_domains}")
                    
                    # Write config to temp file
                    yaml.dump(arxiv_config, temp_config)
                    temp_config_path = temp_config.name
                    
                    # Debug the temporary config file
                    logger.info(f"Created temporary config file at {temp_config_path}")
                    with open(temp_config_path, 'r') as f:
                        logger.debug(f"Temporary config file contents:\n{f.read()}")
                
                # Initialize academic collector
                academic_collector = AcademicCollector(temp_config_path, self.data_dir)
                
                # Run the ArXiv collection
                max_papers = collection_config.get("max_papers_per_term", 1000)
                
                # Double-check that collection is enabled before proceeding
                if collection_enabled:
                    academic_collector.collect_arxiv_papers(papers_per_term=max_papers)
                else:
                    logger.info("Skipping ArXiv collection as collection is disabled.")
                
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
        import time
        from datetime import datetime
        
        # Record start time
        start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting RAG dataset build at {start_datetime}")
        
        # Log GPU/CPU configuration
        gpu_status = "GPU" if self.config.get("embedder", {}).get("use_gpu", False) else "CPU"
        logger.info(f"Using {gpu_status} for embedding generation")
        
        # Run collectors first to download documents
        self._run_collectors()
        
        # Find all documents (including newly downloaded ones)
        document_paths = self.find_all_documents()
        doc_count = len(document_paths)
        logger.info(f"Found {doc_count} documents to process")
        
        if not document_paths:
            logger.info("No new documents to process")
            return
        
        # Process documents in batches
        processing_start = time.time()
        logger.info(f"Starting document processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.process_documents_in_batches(document_paths)
        processing_end = time.time()
        processing_time = processing_end - processing_start
        
        # Final save
        self._save_checkpoint()
        
        # Calculate and log timing information
        end_time = time.time()
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Create performance summary
        logger.info(f"Dataset build complete at {end_datetime}")
        logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Documents processed: {len(self.checkpoint['processed_files'])}")
        logger.info(f"Total chunks: {self.checkpoint['total_chunks']}")
        logger.info(f"Chunks per second: {self.checkpoint['total_chunks'] / total_time:.2f}")
        logger.info(f"Documents per second: {len(self.checkpoint['processed_files']) / total_time:.2f}")
        
        # Save timing information to a file
        timing_file = os.path.join(self.output_dir, "timing_info.txt")
        with open(timing_file, "w") as f:
            f.write(f"Build Configuration: {gpu_status}-based embedding\n")
            f.write(f"Start time: {start_datetime}\n")
            f.write(f"End time: {end_datetime}\n")
            f.write(f"Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s ({total_time:.2f} seconds)\n")
            f.write(f"Documents processed: {len(self.checkpoint['processed_files'])}\n")
            f.write(f"Total chunks: {self.checkpoint['total_chunks']}\n")
            f.write(f"Chunks per second: {self.checkpoint['total_chunks'] / total_time:.2f}\n")
            f.write(f"Documents per second: {len(self.checkpoint['processed_files']) / total_time:.2f}\n")
        
        logger.info(f"Timing information saved to {timing_file}")


def main(config_path=None, threads=None, config_dir=None, use_cpu=None, rag_impl=None):
    """Main function."""
    if config_path is None and config_dir is None:
        parser = argparse.ArgumentParser(description="Build RAG dataset with configurable components")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--config", type=str,
                          help="Path to configuration file (YAML or JSON)")
        group.add_argument("--config_dir", type=str,
                          help="Path to configuration directory (config.d)")
        parser.add_argument("--threads", type=int, default=None,
                          help="Number of worker threads to use for processing")
        
        # Processing mode options
        proc_group = parser.add_argument_group("Processing mode options")
        proc_group.add_argument("--gpu", action="store_true",
                          help="Use GPU-based processing (CPU is the default)")
        
        # RAG implementation options
        rag_group = parser.add_argument_group("RAG implementation options")
        rag_group.add_argument("--pathrag", action="store_true",
                          help="Use PathRAG implementation (default)")
        rag_group.add_argument("--graphrag", action="store_true",
                          help="Use GraphRAG implementation")
        rag_group.add_argument("--literag", action="store_true",
                          help="Use LiteRAG implementation")
                          
        # Document filtering options
        filter_group = parser.add_argument_group("Document filtering options")
        filter_group.add_argument("--filter-domains", action="store_true",
                          help="Only process documents from enabled domains (default)")
        filter_group.add_argument("--no-filter-domains", action="store_true",
                          help="Process all documents regardless of domain enabled status")
        
        args = parser.parse_args()
        config_path = args.config
        config_dir = args.config_dir
        threads = args.threads
        
        # Determine CPU/GPU usage based on arguments
        if args.gpu:
            use_cpu = False
            logger.info("Using GPU mode as specified")
        else:
            # Default to CPU mode
            use_cpu = True
            logger.info("Using CPU mode (default)")
                
        # Determine RAG implementation
        if args.graphrag and args.literag:
            logger.warning("Multiple RAG implementations specified, defaulting to PathRAG")
            rag_impl = "pathrag"
        elif args.graphrag:
            rag_impl = "graphrag"
            logger.info("Using GraphRAG implementation")
        elif args.literag:
            rag_impl = "literag"
            logger.info("Using LiteRAG implementation")
        else:
            # Default to PathRAG
            rag_impl = "pathrag"
            logger.info("Using PathRAG implementation (default)")
            
        # Determine domain filtering setting
        filter_domains = None
        if args.filter_domains and args.no_filter_domains:
            logger.warning("Both --filter-domains and --no-filter-domains specified, defaulting to filtering enabled")
            filter_domains = True
        elif args.filter_domains:
            filter_domains = True
            logger.info("Domain filtering enabled - only processing documents from enabled domains")
        elif args.no_filter_domains:
            filter_domains = False
            logger.info("Domain filtering disabled - processing all documents regardless of domain status")
    
    # Set up PyTorch environment variables for multithreading
    # This is now simpler since CPU is the default mode
    if not args.gpu:
        # Determine number of threads to use
        cpu_threads = threads or os.cpu_count()
        logger.info(f"Setting up PyTorch for CPU processing with {cpu_threads} threads")
        
        # Set PyTorch environment variables for multithreading
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)
    
    # Log key information
    if config_path:
        logger.info(f"Using config file: {config_path}")
        # Load configuration from file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Apply domain filtering setting if specified
        if filter_domains is not None:
            config["filter_by_domains"] = filter_domains
            
        # Apply minimal processing mode settings
        _apply_processing_mode(config, use_cpu, threads)
            
        # Apply RAG implementation settings if specified
        if rag_impl is not None:
            # Modify the configuration for the specified RAG implementation
            _apply_rag_implementation(config, rag_impl)
        
        # Create builder with the modified configuration
        builder = RAGDatasetBuilder(config_file=config_path, config_override=config)
    else:
        logger.info(f"Using config directory: {config_dir}")
        # Load configuration from config.d directory
        from src.core.config_loader import get_configuration
        config = get_configuration(config_dir=config_dir)
        
        # Apply CPU/GPU settings if specified
        if use_cpu is not None:
            # Modify the configuration for CPU/GPU
            _apply_processing_mode(config, use_cpu, threads)
            
        # Apply RAG implementation settings if specified
        if rag_impl is not None:
            # Modify the configuration for the specified RAG implementation
            _apply_rag_implementation(config, rag_impl)
        
        # Create builder with the merged configuration
        builder = RAGDatasetBuilder(config=config)
        logger.info("Created builder with merged configuration from config directory")
        
    if threads:
        logger.info(f"Using {threads} worker threads for processing")
        # Set the number of worker threads for concurrent.futures
        os.environ["PYTHONEXECUTIONPOOL_MAX_WORKERS"] = str(threads)
    
    # Builder is already created above based on config_path or config_dir
    # Build dataset
    builder.build_dataset()


def _apply_processing_mode(config, use_cpu, threads=None):
    """Apply minimal processing mode settings to the configuration.
    Most configuration is now handled by config files.
    
    Args:
        config: Configuration dictionary to modify
        use_cpu: Whether to use CPU (True) or GPU (False)
        threads: Number of threads to use for CPU processing (if None, uses all available cores)
    """
    # Set up embedder configuration based on CPU/GPU flag
    if "embedder" not in config:
        config["embedder"] = {}
    
    # Set use_gpu based on the use_cpu parameter (inverted logic)
    config["embedder"]["use_gpu"] = not use_cpu
    
    # Configure embedder based on CPU/GPU mode
    if use_cpu:
        logger.info("Using CPU mode for embedding generation with Ollama")
        
        # Configure for CPU with Ollama
        if "embedders" not in config:
            config["embedders"] = {}
        
        if "ollama" not in config["embedders"]:
            config["embedders"]["ollama"] = {}
            
        # Set Ollama configuration for CPU
        config["embedders"]["ollama"]["model_name"] = "nomic-embed-text"
        config["embedders"]["ollama"]["host"] = "localhost"
        config["embedders"]["ollama"]["port"] = 11434
        config["embedders"]["ollama"]["batch_size"] = 32
        config["embedders"]["ollama"]["max_workers"] = 8
        config["embedders"]["ollama"]["use_gpu"] = False  # Explicitly disable GPU for CPU mode
        config["embedder"]["type"] = "ollama"
    else:
        logger.info("Using GPU mode for embedding generation with Ollama")
        
        # Configure for GPU with Ollama
        if "embedders" not in config:
            config["embedders"] = {}
        
        if "ollama" not in config["embedders"]:
            config["embedders"]["ollama"] = {}
            
        # Set Ollama configuration
        config["embedders"]["ollama"]["model_name"] = "nomic-embed-text"  # Use specialized embedding model for better performance
        config["embedders"]["ollama"]["host"] = "localhost"
        config["embedders"]["ollama"]["port"] = 11434
        config["embedders"]["ollama"]["batch_size"] = 32
        config["embedders"]["ollama"]["use_gpu"] = True  # Explicitly enable GPU for GPU mode
        
        # Set the embedder type to ollama
        config["embedder"]["type"] = "ollama"
    
    # We respect the collection.enabled setting from the config file
    # No override here - if collection.enabled is false, we won't download papers
    
    return config


def _apply_rag_implementation(config, rag_impl):
    """Apply RAG implementation-specific settings to the configuration.
    
    Args:
        config: Configuration dictionary to modify
        rag_impl: RAG implementation to use ("pathrag", "graphrag", or "literag")
    """
    logger.info(f"Applying {rag_impl.upper()} implementation settings")
    
    # Set output directory based on RAG implementation
    if "output_dir" in config:
        # Check if we should use a custom output directory without appending RAG implementation
        # This is controlled by the CUSTOM_OUTPUT_DIR environment variable or the custom_output_dir config setting
        custom_output_dir = os.environ.get("CUSTOM_OUTPUT_DIR", "false").lower() == "true"
        if "custom_output_dir" in config:
            custom_output_dir = config["custom_output_dir"]
            
        if custom_output_dir:
            # Use the output directory as is without appending RAG implementation
            base_dir = config["output_dir"]
            logger.info(f"Using custom output directory: {base_dir}")
        else:
            # Update the base output directory to include the RAG implementation
            base_dir = config["output_dir"]
            # If the base directory already has a RAG implementation name, replace it
            if any(impl in base_dir for impl in ["pathrag", "graphrag", "literag"]):
                for impl in ["pathrag", "graphrag", "literag"]:
                    base_dir = base_dir.replace(impl, rag_impl)
            else:
                # Otherwise, append the RAG implementation name
                base_dir = os.path.join(base_dir, rag_impl)
        
        config["output_dir"] = base_dir
        logger.info(f"Set output directory to: {base_dir}")
    
    # Apply PathRAG-specific settings
    if rag_impl == "pathrag":
        # Enable PathRAG-specific features
        if "pathrag" not in config:
            config["pathrag"] = {}
        
        config["pathrag"]["enabled"] = True
        
        # Set default PathRAG formatter if not already set
        if "output" not in config:
            config["output"] = {}
        if "formatters" not in config["output"]:
            config["output"]["formatters"] = {}
        if "pathrag" not in config["output"]["formatters"]:
            config["output"]["formatters"]["pathrag"] = {
                "enabled": True,
                "type": "pathrag"
            }
        
        # Set PathRAG as the default formatter
        config["default_output_formatter"] = "pathrag"
    
    # Apply GraphRAG-specific settings
    elif rag_impl == "graphrag":
        # Enable GraphRAG-specific features
        if "graphrag" not in config:
            config["graphrag"] = {}
        
        config["graphrag"]["enabled"] = True
        
        # Set default GraphRAG formatter if not already set
        if "output" not in config:
            config["output"] = {}
        if "formatters" not in config["output"]:
            config["output"]["formatters"] = {}
        if "graphrag" not in config["output"]["formatters"]:
            config["output"]["formatters"]["graphrag"] = {
                "enabled": True,
                "type": "graphrag"
            }
        
        # Set GraphRAG as the default formatter
        config["default_output_formatter"] = "graphrag"
    
    # Apply LiteRAG-specific settings
    elif rag_impl == "literag":
        # Enable LiteRAG-specific features
        if "literag" not in config:
            config["literag"] = {}
        
        config["literag"]["enabled"] = True
        
        # Set default LiteRAG formatter if not already set
        if "output" not in config:
            config["output"] = {}
        if "formatters" not in config["output"]:
            config["output"]["formatters"] = {}
        if "literag" not in config["output"]["formatters"]:
            config["output"]["formatters"]["literag"] = {
                "enabled": True,
                "type": "literag"
            }
        
        # Set LiteRAG as the default formatter
        config["default_output_formatter"] = "literag"
    
    return config


if __name__ == "__main__":
    main()
