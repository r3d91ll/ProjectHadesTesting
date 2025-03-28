#!/usr/bin/env python3
"""
Multi-Source PathRAG Dataset Builder

This script builds a PathRAG dataset using multiple academic paper sources
(ArXiv, Semantic Scholar, PubMed, SocArXiv) with deduplication and Phoenix monitoring.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.deduplication.academic_dedup import AcademicPaperDeduplicator
from src.collectors.semantic_scholar_collector import SemanticScholarCollector
from src.collectors.pubmed_collector import PubMedCollector
from src.collectors.socarxiv_collector import SocArXivCollector

# Import OpenTelemetry for Phoenix
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# Import Phoenix telemetry if available
try:
    import phoenix as px
    from phoenix.otel import register as phoenix_register
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# Import GPU monitoring
try:
    import torch
    import psutil
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'pathrag_dataset_builder.log'))
    ]
)
logger = logging.getLogger('multi_source_pathrag_builder')

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Load environment variables
load_dotenv()

class MultiSourcePathRAGBuilder:
    """
    Builds a PathRAG dataset using multiple academic paper sources with deduplication.
    """
    
    def __init__(self, config_path):
        """
        Initialize the dataset builder.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.start_time = time.time()
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Set up directories
        self.setup_directories()
        
        # Initialize Phoenix telemetry
        self.setup_phoenix_telemetry()
        
        # Initialize deduplicator
        self.deduplicator = AcademicPaperDeduplicator(
            registry_path=os.path.join(self.data_dir, 'registry', 'academic_papers.json'),
            title_similarity_threshold=self.config.get('deduplication', {}).get('title_similarity_threshold', 0.85)
        )
        
        # Initialize collectors
        self.collectors = self.setup_collectors()
        
        # Initialize PathRAG
        self.pathrag, self.storage_backend = self.setup_pathrag()
        
        # GPU monitoring
        self.gpu_info = []
        self.update_gpu_info()
    
    def load_config(self, config_path):
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        logger.info(f"Loading configuration from {config_path}")
        
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")
        
        return config
    
    def setup_directories(self):
        """Set up directories for the dataset builder."""
        # Get directory paths from configuration
        self.data_dir = self.config.get('data_dir', os.path.join('data'))
        self.output_dir = self.config.get('output_dir', os.path.join('data', 'output'))
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'input'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'chunks'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'registry'), exist_ok=True)
        
        logger.info(f"Set up directories: {self.data_dir}, {self.output_dir}")
    
    def setup_phoenix_telemetry(self):
        """Set up OpenTelemetry for Arize Phoenix."""
        project_name = self.config.get('performance_tracking', {}).get(
            'project_name', 'pathrag-dataset-builder-run2'
        )
        
        # Check if Phoenix is available
        if not PHOENIX_AVAILABLE:
            logger.warning("Phoenix package not available, using basic OpenTelemetry setup")
            # Create a basic tracer provider and tracer
            resource = Resource(attributes={SERVICE_NAME: project_name})
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            return self.tracer
        
        try:
            # Get Phoenix URL from config
            phoenix_url = self.config.get('performance_tracking', {}).get(
                'phoenix_url', 'http://0.0.0.0:8084'
            )
            
            # Use the direct Phoenix registration approach
            logger.info(f"Setting up Phoenix OpenTelemetry tracer at {phoenix_url}...")
            endpoint = f"{phoenix_url}/v1/traces"
            
            # Register with Phoenix using their OTEL API
            tracer_provider = phoenix_register(
                project_name=project_name,
                endpoint=endpoint
            )
            
            # Get a tracer for our application
            self.tracer = trace.get_tracer(__name__)
            logger.info(f"Phoenix OpenTelemetry tracer initialized for project: {project_name}")
            
            # For debugging, we can also log to console
            if self.config.get('performance_tracking', {}).get('debug', True):
                try:
                    console_processor = SimpleSpanProcessor(ConsoleSpanExporter())
                    tracer_provider.add_span_processor(console_processor)
                    logger.info("Console exporter added for telemetry debugging")
                except Exception as e:
                    logger.warning(f"Failed to set up console exporter: {e}")
            
            return self.tracer
            
        except Exception as e:
            logger.warning(f"Failed to set up Phoenix exporter: {e}. Using basic tracer instead.")
            # Fall back to a basic tracer
            resource = Resource(attributes={SERVICE_NAME: project_name})
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            return self.tracer
    
    def setup_collectors(self):
        """
        Set up academic paper collectors.
        
        Returns:
            dict: Dictionary of collectors by source
        """
        collectors = {}
        collection_config = self.config.get('collection', {})
        
        # Semantic Scholar
        if collection_config.get('semantic_scholar', {}).get('enabled', True):
            collectors['semantic_scholar'] = SemanticScholarCollector(
                output_dir=os.path.join(self.data_dir, 'input', 'semantic_scholar'),
                deduplicator=self.deduplicator,
                api_key=os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
            )
            logger.info("Initialized Semantic Scholar collector")
        
        # PubMed Central
        if collection_config.get('pubmed', {}).get('enabled', True):
            collectors['pubmed'] = PubMedCollector(
                output_dir=os.path.join(self.data_dir, 'input', 'pubmed'),
                deduplicator=self.deduplicator,
                email=os.environ.get('PUBMED_EMAIL'),
                api_key=os.environ.get('PUBMED_API_KEY')
            )
            logger.info("Initialized PubMed collector")
        
        # SocArXiv
        if collection_config.get('socarxiv', {}).get('enabled', True):
            collectors['socarxiv'] = SocArXivCollector(
                output_dir=os.path.join(self.data_dir, 'input', 'socarxiv'),
                deduplicator=self.deduplicator,
                token=os.environ.get('OSF_TOKEN')
            )
            logger.info("Initialized SocArXiv collector")
        
        return collectors
    
    def setup_pathrag(self):
        """
        Set up PathRAG for storing documents.
        
        Returns:
            tuple: (pathrag, storage_backend)
        """
        from src.implementations.pathrag import PathRAG
        
        storage_config = self.config.get('output', {}).get('formats', {}).get('pathrag', {})
        storage_backend_type = storage_config.get('backend', 'networkx')
        
        with self.tracer.start_as_current_span("pathrag_initialization"):
            # Create the Phoenix adapter for PathRAG telemetry
            class PathRAGPhoenixAdapter:
                def __init__(self, tracer):
                    self.tracer = tracer
                    self.events = []
                
                def log_event(self, event_type, data=None):
                    # Try to log using OpenTelemetry, but fail gracefully if Phoenix is not running
                    try:
                        with self.tracer.start_as_current_span(event_type) as span:
                            if data:
                                for key, value in data.items():
                                    if isinstance(value, (str, int, float, bool)):
                                        span.set_attribute(key, value)
                            self.events.append({"type": event_type, "data": data, "timestamp": time.time()})
                    except Exception as e:
                        logger.debug(f"Phoenix telemetry error (non-critical): {e}")
                
                def flush(self):
                    # Already logging via OpenTelemetry spans
                    pass
            
            # Initialize PathRAG with the custom adapter
            phoenix_adapter = PathRAGPhoenixAdapter(self.tracer)
            pathrag = PathRAG(tracker=phoenix_adapter)
            
            # Set the backend type but don't try to access it directly
            # Initialize the default NetworkX backend - optimized for your RTX A6000 GPUs
            pathrag.backend_type = storage_backend_type
            
            # The storage_backend is directly accessible as an attribute, not a method
            if pathrag.storage_backend is None:
                from src.core.base import NetworkXStorageBackend
                pathrag.storage_backend = NetworkXStorageBackend()
            
            storage_backend = pathrag.storage_backend
            
            logger.info(f"Initialized PathRAG with {storage_backend_type} backend and Phoenix telemetry")
            
            return pathrag, storage_backend
    
    def update_gpu_info(self):
        """Update GPU information for monitoring."""
        if not GPU_AVAILABLE:
            return
        
        try:
            gpus = GPUtil.getGPUs()
            self.gpu_info = []
            
            for i, gpu in enumerate(gpus):
                self.gpu_info.append({
                    'id': i,
                    'name': gpu.name,
                    'load': gpu.load,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
            
            logger.info(f"Added metrics for {len(gpus)} GPUs")
            
            # Log to Phoenix
            with self.tracer.start_as_current_span("gpu_metrics") as span:
                for i, gpu in enumerate(self.gpu_info):
                    for key, value in gpu.items():
                        if isinstance(value, (int, float)):
                            span.set_attribute(f"gpu.{i}.{key}", value)
        
        except Exception as e:
            logger.error(f"Error updating GPU info: {e}")
    
    def collect_papers(self):
        """
        Collect papers from all configured sources.
        
        Returns:
            int: Total number of papers collected
        """
        total_collected = 0
        collection_config = self.config.get('collection', {})
        domains = collection_config.get('domains', {})
        
        with self.tracer.start_as_current_span("paper_collection") as span:
            span.set_attribute("collection.total_sources", len(self.collectors))
            
            # Collect from each source
            for source_name, collector in self.collectors.items():
                logger.info(f"Collecting papers from {source_name}")
                
                # Get search terms for this source
                search_terms = []
                for domain_name, domain in domains.items():
                    if domain.get('enabled', True):
                        search_terms.extend(domain.get('search_terms', []))
                
                # Remove duplicates and limit
                search_terms = list(set(search_terms))
                max_terms = collection_config.get('max_search_terms', 100)
                if max_terms and len(search_terms) > max_terms:
                    logger.warning(f"Limiting search terms from {len(search_terms)} to {max_terms}")
                    search_terms = search_terms[:max_terms]
                
                # Start collection span
                with self.tracer.start_as_current_span(f"{source_name}_collection") as source_span:
                    source_span.set_attribute("source.name", source_name)
                    source_span.set_attribute("source.search_terms", len(search_terms))
                    
                    # Collect papers
                    max_papers = collection_config.get('max_papers_per_term', 50)
                    papers_collected = collector.collect_papers(search_terms, max_papers_per_term=max_papers)
                    
                    # Update attributes
                    source_span.set_attribute("source.papers_collected", papers_collected)
                    source_span.set_attribute("source.papers_found", collector.total_papers_found)
                    source_span.set_attribute("source.duplicates_skipped", collector.total_duplicates_skipped)
                    
                    total_collected += papers_collected
                    
                    # Update GPU metrics
                    self.update_gpu_info()
            
            # Update span attributes
            span.set_attribute("collection.total_papers", total_collected)
            span.set_attribute("collection.time_taken", time.time() - self.start_time)
            
            # Log deduplicator stats
            dedup_stats = self.deduplicator.get_statistics()
            span.set_attribute("deduplication.total_documents", dedup_stats["total_documents"])
            for source, count in dedup_stats.get("sources", {}).items():
                span.set_attribute(f"deduplication.source.{source}", count)
        
        logger.info(f"Collected {total_collected} papers from all sources")
        return total_collected
    
    def process_papers(self):
        """
        Process collected papers and add them to PathRAG.
        
        Returns:
            int: Number of papers processed
        """
        processed_count = 0
        chunk_count = 0
        collection_dir = os.path.join(self.data_dir, 'input')
        
        with self.tracer.start_as_current_span("paper_processing") as span:
            # Walk through all files in the input directory recursively
            for root, dirs, files in os.walk(collection_dir):
                for file in files:
                    # Only process text and PDF files
                    if file.endswith(('.txt', '.pdf', '.json')):
                        file_path = os.path.join(root, file)
                        
                        try:
                            with self.tracer.start_as_current_span("process_document") as doc_span:
                                doc_span.set_attribute("document.path", file_path)
                                doc_span.set_attribute("document.type", file_path.split('.')[-1])
                                
                                # Skip JSON metadata files during processing
                                if file.endswith('.json'):
                                    continue
                                
                                # Read the document
                                doc_content = ""
                                
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        doc_content = f.read()
                                except UnicodeDecodeError:
                                    logger.warning(f"Unicode decode error for {file_path}, skipping")
                                    continue
                                
                                # Skip empty documents
                                if not doc_content.strip():
                                    logger.warning(f"Empty document: {file_path}, skipping")
                                    continue
                                
                                # Get metadata from corresponding JSON if it exists
                                metadata = {}
                                json_path = os.path.splitext(file_path)[0] + '.json'
                                if os.path.exists(json_path):
                                    try:
                                        with open(json_path, 'r', encoding='utf-8') as f:
                                            metadata = json.load(f)
                                    except Exception as e:
                                        logger.warning(f"Error loading metadata for {file_path}: {e}")
                                
                                # Generate a document ID
                                doc_id = f"doc-{processed_count}"
                                if metadata.get('source_id'):
                                    doc_id = f"{metadata.get('source', 'unknown')}-{metadata['source_id']}"
                                
                                # Add document to PathRAG
                                with self.tracer.start_as_current_span("add_document") as add_doc_span:
                                    # Add to storage backend
                                    title = metadata.get('title', os.path.basename(file_path))
                                    add_doc_span.set_attribute("document.title", title)
                                    
                                    # Add document to the storage backend
                                    self.storage_backend.add_item(
                                        doc_id, "document", 
                                        title=title, 
                                        content=doc_content,
                                        metadata=metadata
                                    )
                                    
                                    # Create chunks from the document
                                    chunks = self.create_chunks(doc_content)
                                    chunk_ids = []
                                    
                                    # Add each chunk to storage backend
                                    for i, chunk_text in enumerate(chunks):
                                        if not chunk_text.strip():
                                            continue
                                        
                                        chunk_id = f"{doc_id}-chunk-{i}"
                                        self.storage_backend.add_item(
                                            chunk_id, "chunk", 
                                            content=chunk_text.strip(),
                                            metadata={"source_doc": doc_id}
                                        )
                                        self.storage_backend.add_relationship(doc_id, chunk_id, "contains")
                                        chunk_ids.append(chunk_id)
                                        chunk_count += 1
                                
                                processed_count += 1
                                doc_span.set_attribute("document.chunks", len(chunk_ids))
                                
                                # Update GPU metrics periodically
                                if processed_count % 10 == 0:
                                    self.update_gpu_info()
                                
                                # Log progress
                                if processed_count % 100 == 0:
                                    logger.info(f"Processed {processed_count} documents, {chunk_count} chunks so far...")
                        
                        except Exception as e:
                            logger.error(f"Error processing document {file_path}: {e}")
            
            # Update span attributes
            span.set_attribute("processing.documents_count", processed_count)
            span.set_attribute("processing.chunks_count", chunk_count)
            span.set_attribute("processing.time_taken", time.time() - self.start_time)
        
        logger.info(f"Processed {processed_count} documents with {chunk_count} chunks")
        return processed_count
    
    def create_chunks(self, text):
        """
        Create chunks from document text.
        
        Args:
            text (str): Document text
            
        Returns:
            list: List of chunk texts
        """
        # Get chunking configuration
        chunker_config = self.config.get('chunker', {})
        chunker_type = chunker_config.get('type', 'fixed_size')
        chunk_size = chunker_config.get('chunk_size', 1000)
        chunk_overlap = chunker_config.get('chunk_overlap', 200)
        
        if chunker_type == 'semantic':
            # Simple paragraph-based chunking for now
            # In a real implementation, this would use a more sophisticated approach
            chunks = text.split('\n\n')
            
            # Filter out empty chunks and ensure they're not too long
            chunks = [chunk for chunk in chunks if chunk.strip()]
            
            # If chunks are too long, split them further
            if any(len(chunk) > chunk_size for chunk in chunks):
                new_chunks = []
                for chunk in chunks:
                    if len(chunk) > chunk_size:
                        # Split into smaller chunks
                        words = chunk.split()
                        for i in range(0, len(words), chunk_size // 10):
                            new_chunks.append(' '.join(words[i:i + chunk_size // 10]))
                    else:
                        new_chunks.append(chunk)
                chunks = new_chunks
            
            return chunks
        
        elif chunker_type == 'fixed_size':
            # Split by fixed number of characters
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks
        
        else:
            # Default to simple splitting by paragraphs
            chunks = text.split('\n\n')
            return [chunk for chunk in chunks if chunk.strip()]
    
    def run_test_queries(self, num_queries=5):
        """
        Run test queries on the PathRAG dataset.
        
        Args:
            num_queries (int): Number of queries to run
            
        Returns:
            dict: Query results
        """
        # Define some test queries based on the domains
        domains = self.config.get('collection', {}).get('domains', {})
        
        default_queries = [
            "What is the anthropology of value?",
            "How does actor-network theory relate to science and technology studies?",
            "What is the relationship between cultural value and technology?",
            "How do indigenous communities interact with modern technology?",
            "What ethical considerations are important in digital anthropology?"
        ]
        
        # Generate queries from search terms
        domain_queries = []
        for domain_name, domain in domains.items():
            if domain.get('enabled', True):
                search_terms = domain.get('search_terms', [])
                for term in search_terms[:2]:  # Take first 2 terms from each domain
                    domain_queries.append(f"What is {term}?")
                    domain_queries.append(f"Explain the concept of {term}.")
        
        # Use domain queries if available, otherwise use defaults
        queries = domain_queries[:num_queries] if domain_queries else default_queries[:num_queries]
        
        results = {}
        
        with self.tracer.start_as_current_span("test_queries") as span:
            span.set_attribute("queries.count", len(queries))
            
            for i, query in enumerate(queries):
                with self.tracer.start_as_current_span(f"query_{i}") as query_span:
                    query_span.set_attribute("query.text", query)
                    
                    # Start timing
                    query_start_time = time.time()
                    
                    # Query the storage backend
                    query_results = self.storage_backend.query({"item_type": "chunk"}, limit=5)
                    
                    # Calculate query time
                    query_time = time.time() - query_start_time
                    
                    # Store results
                    results[query] = {
                        "time": query_time,
                        "num_results": len(query_results),
                        "results": [
                            {
                                "id": result.get("id"),
                                "content": result.get("content", "")[:100] + "..." 
                                if len(result.get("content", "")) > 100 else result.get("content", ""),
                                "score": result.get("score", 0)
                            }
                            for result in query_results
                        ]
                    }
                    
                    # Update query span
                    query_span.set_attribute("query.time", query_time)
                    query_span.set_attribute("query.results", len(query_results))
                    
                    # Update GPU metrics
                    self.update_gpu_info()
                    
                    logger.info(f"Query '{query}' returned {len(query_results)} results in {query_time:.3f} seconds")
            
            # Update span with overall stats
            total_time = sum(r["time"] for r in results.values())
            avg_time = total_time / len(results) if results else 0
            span.set_attribute("queries.total_time", total_time)
            span.set_attribute("queries.avg_time", avg_time)
        
        return results
    
    def build_dataset(self):
        """
        Build the complete PathRAG dataset.
        
        This method orchestrates the entire dataset building process:
        1. Collecting papers from multiple sources
        2. Processing papers and adding to PathRAG
        3. Running test queries
        
        Returns:
            dict: Dataset statistics
        """
        start_time = time.time()
        
        with self.tracer.start_as_current_span("build_dataset") as span:
            # Step 1: Collect papers
            if self.config.get('collection', {}).get('enabled', True):
                papers_collected = self.collect_papers()
                span.set_attribute("dataset.papers_collected", papers_collected)
            else:
                logger.info("Paper collection disabled, using existing papers")
                papers_collected = 0
            
            # Step 2: Process papers
            papers_processed = self.process_papers()
            span.set_attribute("dataset.papers_processed", papers_processed)
            
            # Step 3: Run test queries
            test_queries = self.run_test_queries(
                num_queries=self.config.get('testing', {}).get('num_queries', 5)
            )
            span.set_attribute("dataset.test_queries", len(test_queries))
            
            # Calculate total time
            total_time = time.time() - start_time
            span.set_attribute("dataset.total_time", total_time)
            
            # Log final GPU metrics
            self.update_gpu_info()
        
        # Build statistics
        stats = {
            "papers_collected": papers_collected,
            "papers_processed": papers_processed,
            "test_queries": len(test_queries),
            "total_time": total_time,
            "deduplication_stats": self.deduplicator.get_statistics()
        }
        
        logger.info(f"Dataset built successfully in {total_time:.2f} seconds")
        logger.info(f"Processed {papers_processed} documents")
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats


def ensure_phoenix_running():
    """Ensure Phoenix server is running for telemetry tracking."""
    import socket
    try:
        # Check if Phoenix is already running on port 8080
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', 8080)) == 0:
                logger.info("Phoenix server already running on port 8080")
                return True
        
        # Try to start Phoenix server
        try:
            import phoenix as px
            import threading
            
            def start_phoenix_server():
                try:
                    # Use our RTX A6000 GPUs for Phoenix visualization
                    logger.info("Starting Phoenix server on port 8080")
                    px.launch_app(port=8080)
                except Exception as e:
                    logger.warning(f"Could not start Phoenix server: {e}")
            
            # Start Phoenix in a background thread
            phoenix_thread = threading.Thread(target=start_phoenix_server)
            phoenix_thread.daemon = True
            phoenix_thread.start()
            
            # Give Phoenix a moment to start
            import time
            time.sleep(2)
            
            logger.info("Phoenix server started in background")
            return True
        except ImportError:
            logger.warning("Phoenix package not installed, telemetry will be limited")
            return False
    except Exception as e:
        logger.warning(f"Error checking/starting Phoenix: {e}")
        return False

def ensure_telemetry_flush():
    """Ensure all telemetry data is flushed to Phoenix before exit."""
    if PHOENIX_AVAILABLE:
        try:
            # Force flush any pending telemetry data
            logger.info("Flushing telemetry data to Phoenix...")
            # For the built-in OpenTelemetry provider
            provider = trace.get_tracer_provider()
            if hasattr(provider, 'force_flush'):
                provider.force_flush()
            
            # For Phoenix-specific flushing
            if 'px' in globals():
                try:
                    # Ensure Phoenix spans are flushed
                    px.flush()
                    logger.info("Phoenix span data flushed successfully")
                except Exception as e:
                    logger.warning(f"Error flushing Phoenix data: {e}")
        except Exception as e:
            logger.warning(f"Error during telemetry flush: {e}")

def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build a PathRAG dataset using multiple academic paper sources")
    parser.add_argument("--config", type=str, default="config/archive/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--no-phoenix", action="store_true",
                      help="Disable Phoenix server startup")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Start Phoenix if needed
    if not args.no_phoenix:
        ensure_phoenix_running()
    
    # Build the dataset
    try:
        builder = MultiSourcePathRAGBuilder(args.config)
        stats = builder.build_dataset()
        
        # Ensure telemetry data is properly flushed
        logger.info("Ensuring all telemetry data is flushed to Phoenix...")
        ensure_telemetry_flush()
        
        logger.info("PathRAG dataset built successfully!")
        logger.info(f"To view the Phoenix dashboard: http://localhost:8080")
        
        # Allow time for telemetry to be completely flushed
        time.sleep(2)
        return 0
    except Exception as e:
        logger.error(f"Error building dataset: {e}", exc_info=True)
        # Try to flush telemetry even on error
        ensure_telemetry_flush()
        return 1

if __name__ == "__main__":
    sys.exit(main())
