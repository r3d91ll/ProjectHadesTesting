"""
PathRAG Arize Phoenix Integration Adapter

This module implements a PathRAG adapter with Arize Phoenix integration for
performance tracking and evaluation. It extends the original PathRAG adapter
with telemetry capabilities for logging LLM prompts, responses, and metrics.
"""

import os
import sys
import time
import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests

# Arize Phoenix imports for telemetry
try:
    import pandas as pd
    from arize.phoenix.session import Session
    from arize.phoenix.trace import trace, response_converter
    from arize.phoenix.trace.trace import LLMTrace
    from arize.phoenix.trace import model_call_converter
except ImportError:
    raise ImportError(
        "Failed to import Arize Phoenix. Install it with: pip install arize-phoenix"
    )

# Add the original PathRAG implementation to the Python path
PATHRAG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "temp", "pathrag")
sys.path.append(PATHRAG_DIR)

# Import the original PathRAG implementation
try:
    from PathRAG.PathRAG import PathRAG as OriginalPathRAG
    from PathRAG.utils import init_model, init_tokenizer
    import networkx as nx
except ImportError as e:
    raise ImportError(
        f"Import error: {e}. Original PathRAG implementation or NetworkX not found. "
        f"Make sure the code is available at: {PATHRAG_DIR} and NetworkX is installed."
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PathRAGArizeAdapter:
    """Adapter for PathRAG with Arize Phoenix integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PathRAG adapter with Arize Phoenix integration.
        
        Args:
            config: Configuration dictionary for PathRAG and Arize Phoenix integration
        """
        self.config = config
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.pathrag = None
        self.graph = None
        self.initialized = False
        
        # Arize Phoenix configuration
        self.phoenix_host = config.get("phoenix_host", "arize-phoenix")
        self.phoenix_port = config.get("phoenix_port", 8080)
        self.phoenix_url = f"http://{self.phoenix_host}:{self.phoenix_port}"
        self.track_performance = config.get("track_performance", True)
        
        # Initialize Phoenix session if tracking is enabled
        if self.track_performance:
            try:
                self.phoenix_session = Session(url=self.phoenix_url)
                logger.info(f"Connected to Arize Phoenix at {self.phoenix_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Arize Phoenix: {e}")
                self.track_performance = False
    
    def initialize(self) -> None:
        """Initialize the PathRAG system and verify Phoenix connection."""
        # Initialize the model and tokenizer
        model = init_model(self.model_name)
        tokenizer = init_tokenizer(self.model_name)
        
        # Initialize the original PathRAG implementation
        self.pathrag = OriginalPathRAG(
            model=model,
            tokenizer=tokenizer,
            **{k: v for k, v in self.config.items() if k not in ["model_name", "phoenix_host", "phoenix_port", "track_performance"]}
        )
        
        # Verify Phoenix connection if tracking is enabled
        if self.track_performance:
            try:
                response = requests.get(f"{self.phoenix_url}/health")
                if response.status_code == 200:
                    logger.info("Arize Phoenix health check successful")
                else:
                    logger.warning(f"Arize Phoenix health check failed: {response.status_code}")
                    self.track_performance = False
            except Exception as e:
                logger.error(f"Failed to connect to Arize Phoenix: {e}")
                self.track_performance = False
        
        self.initialized = True
    
    def _log_to_phoenix(self, trace_data: Dict[str, Any]) -> None:
        """
        Log trace data to Arize Phoenix.
        
        Args:
            trace_data: Dictionary containing trace data
        """
        if not self.track_performance:
            return
        
        try:
            # Create and save trace to Phoenix
            trace_obj = LLMTrace(
                id=trace_data.get("id", str(uuid.uuid4())),
                name=trace_data.get("name", "PathRAG Query"),
                model=trace_data.get("model", self.model_name),
                input=trace_data.get("input", ""),
                output=trace_data.get("output", ""),
                prompt_tokens=trace_data.get("prompt_tokens", 0),
                completion_tokens=trace_data.get("completion_tokens", 0),
                latency_ms=trace_data.get("latency_ms", 0),
                metadata=trace_data.get("metadata", {}),
                spans=trace_data.get("spans", [])
            )
            
            self.phoenix_session.log_trace(trace_obj)
            logger.info(f"Logged trace {trace_obj.id} to Arize Phoenix")
        except Exception as e:
            logger.error(f"Failed to log to Arize Phoenix: {e}")
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query using PathRAG and log to Arize Phoenix.
        
        Args:
            query: The query string
            **kwargs: Additional arguments for the query
            
        Returns:
            Dict containing the response and related information
        """
        if not self.initialized:
            self.initialize()
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Create a trace for this query
        trace_metadata = {
            "user_id": kwargs.get("user_id", "anonymous"),
            "session_id": kwargs.get("session_id", str(uuid.uuid4())),
            "query_params": json.dumps({k: v for k, v in kwargs.items() 
                                      if k not in ["user_id", "session_id"]})
        }
        
        try:
            # Call the original implementation
            result = self.pathrag.answer(query, **kwargs)
            
            # Calculate performance metrics
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Extract relevant information for standardization
            answer = result.get("answer", "")
            paths = result.get("paths", [])
            context = result.get("context", "")
            
            # Log to Arize Phoenix if tracking is enabled
            if self.track_performance:
                trace_data = {
                    "id": trace_id,
                    "name": "PathRAG Query",
                    "model": self.model_name,
                    "input": query,
                    "output": answer,
                    "prompt_tokens": len(query.split()),
                    "completion_tokens": len(answer.split()),
                    "latency_ms": latency_ms,
                    "metadata": {
                        **trace_metadata,
                        "num_paths": len(paths),
                        "context_length": len(context),
                        "timestamp": datetime.now().isoformat()
                    },
                    "spans": [
                        {
                            "name": "retrieve_paths",
                            "start_time": start_time,
                            "end_time": start_time + (latency_ms * 0.6 / 1000),  # Estimate
                            "metadata": {"num_paths": len(paths)}
                        },
                        {
                            "name": "generate_answer",
                            "start_time": start_time + (latency_ms * 0.6 / 1000),
                            "end_time": end_time,
                            "metadata": {"context_length": len(context)}
                        }
                    ]
                }
                self._log_to_phoenix(trace_data)
            
            # Return in a standardized format for our framework
            return {
                "answer": answer,
                "paths": paths,
                "context": context,
                "raw_result": result,
                "metrics": {
                    "latency_ms": latency_ms,
                    "trace_id": trace_id
                }
            }
        
        except Exception as e:
            # Log the error to Arize Phoenix if tracking is enabled
            if self.track_performance:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                
                trace_data = {
                    "id": trace_id,
                    "name": "PathRAG Query Error",
                    "model": self.model_name,
                    "input": query,
                    "output": f"Error: {str(e)}",
                    "prompt_tokens": len(query.split()),
                    "completion_tokens": len(str(e).split()),
                    "latency_ms": latency_ms,
                    "metadata": {
                        **trace_metadata,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                self._log_to_phoenix(trace_data)
            
            # Re-raise the exception
            raise e
    
    def get_paths(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve paths for a query without generating an answer.
        
        Args:
            query: The query string
            top_k: Number of paths to retrieve
            **kwargs: Additional arguments for path retrieval
            
        Returns:
            List of retrieved paths
        """
        if not self.initialized:
            self.initialize()
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Adapt to the original implementation's method for retrieving paths
        paths = self.pathrag.retrieve_paths(query, top_k=top_k, **kwargs)
        
        # Log to Arize Phoenix if tracking is enabled
        if self.track_performance:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            trace_data = {
                "id": trace_id,
                "name": "PathRAG Path Retrieval",
                "model": self.model_name,
                "input": query,
                "output": json.dumps(paths),
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(json.dumps(paths).split()),
                "latency_ms": latency_ms,
                "metadata": {
                    "user_id": kwargs.get("user_id", "anonymous"),
                    "session_id": kwargs.get("session_id", str(uuid.uuid4())),
                    "top_k": top_k,
                    "num_paths_retrieved": len(paths),
                    "timestamp": datetime.now().isoformat()
                }
            }
            self._log_to_phoenix(trace_data)
        
        return paths
    
    def ingest_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest a document into the knowledge base.
        
        Args:
            document: Document text to ingest
            metadata: Optional metadata for the document
            
        Returns:
            Dict containing information about the ingestion result
        """
        if not self.initialized:
            self.initialize()
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Use the original implementation's method for ingesting documents
        result = self.pathrag.ingest_text(document, metadata=metadata or {})
        
        # Log to Arize Phoenix if tracking is enabled
        if self.track_performance:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            trace_data = {
                "id": trace_id,
                "name": "PathRAG Document Ingestion",
                "model": self.model_name,
                "input": f"Document: {document[:100]}... (Length: {len(document)})",
                "output": json.dumps(result),
                "latency_ms": latency_ms,
                "metadata": {
                    "document_length": len(document),
                    "nodes_created": result.get("nodes_created", 0),
                    "edges_created": result.get("edges_created", 0),
                    "document_id": result.get("document_id", ""),
                    "timestamp": datetime.now().isoformat()
                }
            }
            self._log_to_phoenix(trace_data)
        
        return {
            "success": True,
            "nodes_created": result.get("nodes_created", 0),
            "edges_created": result.get("edges_created", 0),
            "document_id": result.get("document_id", ""),
            "raw_result": result,
            "metrics": {
                "latency_ms": latency_ms,
                "trace_id": trace_id
            }
        }
    
    def ingest_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Ingest a file into the knowledge base.
        
        Args:
            file_path: Path to the file to ingest
            **kwargs: Additional arguments for ingestion
            
        Returns:
            Dict containing information about the ingestion result
        """
        if not self.initialized:
            self.initialize()
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get metadata from the file path
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            **kwargs
        }
        
        # Ingest the document
        return self.ingest_document(content, metadata)
        
    def evaluate_query(self, query: str, ground_truth: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query using PathRAG and evaluate against ground truth.
        
        Args:
            query: The query string
            ground_truth: Ground truth answer for evaluation
            **kwargs: Additional arguments for the query
            
        Returns:
            Dict containing the response, evaluation metrics, and related information
        """
        # First get the normal query result
        result = self.query(query, **kwargs)
        answer = result["answer"]
        
        # Calculate evaluation metrics
        # This could be expanded with more sophisticated metrics
        from nltk.translate.bleu_score import sentence_bleu
        from rouge import Rouge
        import nltk
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Tokenize the texts
        answer_tokens = nltk.word_tokenize(answer.lower())
        ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
        
        # Calculate BLEU score
        bleu_score = sentence_bleu([ground_truth_tokens], answer_tokens)
        
        # Calculate ROUGE scores
        try:
            rouge = Rouge()
            rouge_scores = rouge.get_scores(answer, ground_truth)[0]
        except Exception as e:
            logger.warning(f"Error calculating ROUGE scores: {e}")
            rouge_scores = {
                "rouge-1": {"f": 0, "p": 0, "r": 0},
                "rouge-2": {"f": 0, "p": 0, "r": 0},
                "rouge-l": {"f": 0, "p": 0, "r": 0}
            }
        
        # Log to Arize Phoenix if tracking is enabled
        if self.track_performance:
            trace_id = str(uuid.uuid4())
            
            trace_data = {
                "id": trace_id,
                "name": "PathRAG Evaluation",
                "model": self.model_name,
                "input": query,
                "output": answer,
                "expected_output": ground_truth,
                "latency_ms": result["metrics"]["latency_ms"],
                "metadata": {
                    "user_id": kwargs.get("user_id", "anonymous"),
                    "session_id": kwargs.get("session_id", str(uuid.uuid4())),
                    "evaluation_scores": {
                        "bleu": bleu_score,
                        "rouge-1-f": rouge_scores["rouge-1"]["f"],
                        "rouge-2-f": rouge_scores["rouge-2"]["f"],
                        "rouge-l-f": rouge_scores["rouge-l"]["f"]
                    },
                    "timestamp": datetime.now().isoformat()
                }
            }
            self._log_to_phoenix(trace_data)
        
        # Add evaluation metrics to result
        result["evaluation"] = {
            "bleu_score": bleu_score,
            "rouge_scores": rouge_scores
        }
        
        return result
