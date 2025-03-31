#!/usr/bin/env python3
"""
RAG Dataset Builder - Embedding Modules

This module contains implementations of text embedders for the RAG Dataset Builder.
"""

import os
import logging
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.builder import BaseEmbedder

# Configure logging
logger = logging.getLogger("rag_dataset_builder.embedders")


class SentenceTransformerEmbedder(BaseEmbedder):
    """Text embedder using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None,
                 batch_size: int = 32, use_gpu: bool = True):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use (cpu or cuda)
            batch_size: Batch size for embedding generation
            use_gpu: Whether to use GPU acceleration
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Determine device
            if device is None:
                if use_gpu and torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            
            self.device = device
            self.batch_size = batch_size
            self.model_name = model_name
            
            logger.info(f"Loading embedding model {model_name} on {device}")
            start_time = time.time()
            self.model = SentenceTransformer(model_name, device=device)
            load_time = time.time() - start_time
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded in {load_time:.2f}s. Embedding dimension: {self.embedding_dim}")
            
            # Warm up the model
            self._warm_up()
            
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it with: pip install sentence-transformers")
            raise
    
    def _warm_up(self):
        """Warm up the model with a sample input."""
        try:
            logger.info("Warming up embedding model...")
            sample_text = "This is a sample text to warm up the embedding model."
            sample_embedding = self.model.encode([sample_text], batch_size=1, show_progress_bar=False)
            logger.info(f"Model warm-up complete. Sample embedding shape: {sample_embedding.shape}")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
            
            start_time = time.time()
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_tensor=False,  # Return numpy array
                normalize_embeddings=True  # L2 normalization
            )
            
            # Convert to Python list for JSON serialization
            embeddings_list = embeddings.tolist()
            
            generation_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {generation_time:.2f}s")
            
            return embeddings_list
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return empty embeddings as fallback
            return [[] for _ in range(len(texts))]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size
        }


class OpenAIEmbedder(BaseEmbedder):
    """Text embedder using OpenAI API."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 100):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the OpenAI embedding model
            batch_size: Batch size for embedding generation
        """
        try:
            import openai
            self.openai = openai
            
            # Check for API key
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY environment variable not set.")
            
            self.model_name = model_name
            self.batch_size = batch_size
            
            # Model dimensions mapping
            self.dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            
            self.embedding_dim = self.dimensions.get(model_name, 1536)
            logger.info(f"Using OpenAI model {model_name} with dimension {self.embedding_dim}")
            
        except ImportError:
            logger.error("openai package not installed. Please install it with: pip install openai")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        try:
            all_embeddings = []
            logger.info(f"Generating embeddings for {len(texts)} texts using OpenAI {self.model_name}")
            
            # Process in batches
            for i in tqdm(range(0, len(texts), self.batch_size)):
                batch_texts = texts[i:i + self.batch_size]
                
                # Call OpenAI API
                response = self.openai.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Add delay to respect rate limits
                if i + self.batch_size < len(texts):
                    time.sleep(0.5)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings using OpenAI API")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            # Return empty embeddings as fallback
            return [[] for _ in range(len(texts))]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "provider": "OpenAI",
            "batch_size": self.batch_size
        }


class OllamaEmbedder(BaseEmbedder):
    """Text embedder using Ollama API."""
    
    def __init__(self, model_name: str = "tinyllama", host: str = "localhost", port: int = 11434,
                 batch_size: int = 16, max_workers: int = 8, use_gpu: bool = True):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the Ollama model
            host: Ollama host
            port: Ollama port
            batch_size: Batch size for embedding generation
            max_workers: Number of concurrent workers
            use_gpu: Whether to use GPU for embeddings (False to force CPU)
        """
        try:
            import requests
            import concurrent.futures
            self.requests = requests
            self.concurrent_futures = concurrent.futures
            
            # Force using nomic-embed-text for optimal embedding performance
            # This ensures we always use the specialized embedding model regardless of configuration
            self.model_name = "nomic-embed-text"
            self.batch_size = batch_size
            self.host = host
            self.port = port
            self.max_workers = max_workers
            self.use_gpu = use_gpu
            self.api_url = f"http://{host}:{port}"
            self.embedding_endpoint = f"{self.api_url}/api/embeddings"
            
            # Default embedding dimension for Ollama models
            # This will be updated after the first API call
            if "nomic-embed-text" in model_name:
                self.embedding_dim = 768  # nomic-embed-text has 768 dimensions
            elif "tinyllama" in model_name:
                self.embedding_dim = 2048  # TinyLlama has 2048 dimensions
            else:
                self.embedding_dim = 4096  # Default for most Ollama models
            
            logger.info(f"Forcing use of nomic-embed-text model for optimal embedding performance")
            logger.info(f"Using Ollama model {self.model_name} at {self.api_url}")
            logger.info(f"GPU usage for embeddings: {'Enabled' if self.use_gpu else 'Disabled (using CPU)'}")
            
            # Test connection
            self._test_connection()
            
        except ImportError:
            logger.error("requests package not installed. Please install it with: pip install requests")
            raise
    
    def _test_connection(self):
        """Test connection to Ollama API."""
        try:
            response = self.requests.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                if self.model_name in model_names:
                    logger.info(f"Successfully connected to Ollama. Model {self.model_name} is available.")
                else:
                    available_models = ", ".join(model_names)
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {available_models}")
            else:
                logger.warning(f"Failed to connect to Ollama API: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error testing connection to Ollama: {e}")
    
    def _get_embedding_for_text(self, text: str) -> List[float]:
        """
        Get embedding for a single text using Ollama API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Truncate very long texts to improve performance
            if len(text) > 1024:
                text = text[:1024]
                
            # Construct payload to match the CLI behavior
            # ollama embed nomic-embed-text --use-cpu
            # Construct payload with more explicit CPU/GPU control
            payload = {
                "model": self.model_name,
                "prompt": text,
                "options": {
                    "use_cpu": not self.use_gpu,  # Set use_cpu based on use_gpu setting
                    "num_gpu": 0 if not self.use_gpu else 1  # Explicitly set num_gpu to 0 for CPU mode
                }
            }
            
            # Log the first few requests to confirm settings
            if len(text) < 50:  # Only log short texts to avoid cluttering logs
                logger.info(f"Ollama embedding request options: {payload['options']}")
            
            # This endpoint is equivalent to the 'embed' subcommand in CLI
            response = self.requests.post(self.embedding_endpoint, json=payload)
            
            if response.status_code == 200:
                return response.json().get("embedding", [])
            else:
                logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
                return [0.0] * self.embedding_dim
        except Exception as e:
            logger.error(f"Error in _get_embedding_for_text: {e}")
            return [0.0] * self.embedding_dim
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using Ollama API with concurrent processing.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        try:
            all_embeddings = []
            logger.info(f"Generating embeddings for {len(texts)} texts using Ollama {self.model_name} with {self.max_workers} workers")
            
            # Process in batches with concurrent execution
            for i in tqdm(range(0, len(texts), self.batch_size)):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = [None] * len(batch_texts)  # Pre-allocate results list
                
                # Use ThreadPoolExecutor for concurrent API calls
                with self.concurrent_futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all tasks
                    future_to_index = {executor.submit(self._get_embedding_for_text, text): idx 
                                      for idx, text in enumerate(batch_texts)}
                    
                    # Process results as they complete
                    for future in self.concurrent_futures.as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            embedding = future.result()
                            batch_embeddings[idx] = embedding
                            
                            # Update embedding dimension based on first response
                            if len(all_embeddings) == 0 and idx == 0:
                                self.embedding_dim = len(embedding)
                                logger.info(f"Ollama embedding dimension: {self.embedding_dim}")
                        except Exception as e:
                            logger.error(f"Error processing embedding result: {e}")
                            batch_embeddings[idx] = [0.0] * self.embedding_dim
                
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings using Ollama API")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Ollama: {e}")
            # Return empty embeddings as fallback
            return [[] for _ in range(len(texts))]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "provider": "Ollama",
            "batch_size": self.batch_size,
            "host": self.host,
            "port": self.port,
            "max_workers": self.max_workers
        }
