#!/usr/bin/env python3
"""
Configuration Utilities for RAG Dataset Builder

This module provides utilities for loading and validating configuration
for the RAG Dataset Builder framework.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import importlib.util

# Try to import the config_loader if it exists
try:
    from .config_loader import get_configuration
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False

logger = logging.getLogger("rag_dataset_builder")

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Deprecated: Consider using get_configuration() from config_loader.py for the new config.d approach.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If there's a problem loading the configuration
    """
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If there's a problem loading the configuration
    """
    if not os.path.exists(config_path):
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {config_path}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration from {config_path}: {str(e)}")

def validate_required_config(config: Dict[str, Any], required_keys: Dict[str, type]) -> None:
    """
    Validate that required configuration keys are present and of the correct type.
    
    Args:
        config: Configuration dictionary
        required_keys: Dictionary mapping required keys to their expected types
        
    Raises:
        ConfigurationError: If a required key is missing or of the wrong type
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ConfigurationError(f"Required configuration key '{key}' not found")
        
        if not isinstance(config[key], expected_type):
            raise ConfigurationError(f"Configuration key '{key}' must be of type {expected_type.__name__}")

def get_nested_config(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path to the configuration value (e.g., "output.formats.pathrag.enabled")
        default: Default value to return if the path doesn't exist
        
    Returns:
        The configuration value at the specified path, or the default if not found
    """
    keys = path.split('.')
    result = config
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    
    return result

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with values from override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary to override values in base_config
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, override_value in override_config.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(override_value, dict)
        ):
            # Recursively merge dictionaries
            result[key] = merge_configs(result[key], override_value)
        else:
            # Override the value
            result[key] = override_value
    
    return result

def create_rag_config(
    rag_type: str,
    source_dir: str,
    output_dir: str,
    additional_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for a specific RAG implementation.
    
    Args:
        rag_type: Type of RAG implementation (e.g., "pathrag", "graphrag")
        source_dir: Directory containing source documents
        output_dir: Directory to save RAG output
        additional_config: Additional configuration options
        
    Returns:
        RAG configuration dictionary
    """
    # Create base configuration
    config = {
        "directories": {
            "input": source_dir,
            "output": output_dir,
        }
    }
    
    # Add default configuration based on RAG type
    if rag_type == "pathrag":
        config.update({
            "retrieval_systems": {
                "pathrag": {
                    "type": "pathrag",
                    "storage_backend": "networkx",
                    "embedder": "openai",
                    "chunker": "sliding_window",
                    "processor": "pdf"
                }
            }
        })
    elif rag_type == "vectorrag":
        config.update({
            "retrieval_systems": {
                "vectorrag": {
                    "type": "vectorrag",
                    "storage_backend": "faiss",
                    "embedder": "sentence_transformers",
                    "chunker": "fixed_size",
                    "processor": "pdf"
                }
            }
        })
    elif rag_type == "graphrag":
        config.update({
            "retrieval_systems": {
                "graphrag": {
                    "type": "graphrag",
                    "storage_backend": "neo4j",
                    "embedder": "openai",
                    "chunker": "semantic",
                    "processor": "pdf"
                }
            }
        })
    
    # Merge with additional configuration
    if additional_config:
        config = merge_configs(config, additional_config)
    
    return config
