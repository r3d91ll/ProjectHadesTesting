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
