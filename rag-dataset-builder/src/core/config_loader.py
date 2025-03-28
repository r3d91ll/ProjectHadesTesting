#!/usr/bin/env python3
"""
Configuration Loader for RAG Dataset Builder

This module provides functionality for loading and merging configurations
from the config.d/ directory in the correct order.
"""

import os
import re
import logging
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path

from .config import merge_configs, ConfigurationError

logger = logging.getLogger("rag_dataset_builder.config_loader")

def load_config_directory(config_dir: str = "config.d") -> Dict[str, Any]:
    """
    Load all configuration files from the config.d directory in numeric order.
    
    Args:
        config_dir: Path to the configuration directory (default: "config.d")
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        ConfigurationError: If there's a problem loading configurations
    """
    if not os.path.exists(config_dir) or not os.path.isdir(config_dir):
        raise ConfigurationError(f"Configuration directory not found: {config_dir}")
    
    # Dictionary to store the final configuration
    config = {}
    
    # Get all YAML files from the config directory
    config_files = []
    for f in os.listdir(config_dir):
        file_path = os.path.join(config_dir, f)
        if os.path.isfile(file_path) and f.endswith(('.yaml', '.yml')):
            config_files.append(file_path)
    
    # Sort configuration files by numeric prefix
    def get_numeric_prefix(file_path):
        filename = os.path.basename(file_path)
        match = re.match(r'^(\d+)', filename)
        if match:
            return int(match.group(1))
        return float('inf')  # Files without numeric prefix go last
    
    config_files.sort(key=get_numeric_prefix)
    
    # Load and merge configurations in order
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                
            if file_config:  # Skip empty files
                logger.info(f"Loaded configuration from {config_file}")
                config = merge_configs(config, file_config)
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            raise ConfigurationError(f"Error loading {config_file}: {str(e)}")
    
    return config

def load_user_config(config_path: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a user-provided configuration file and merge it with the base configuration.
    
    Args:
        config_path: Path to the user configuration file
        base_config: Base configuration to merge with
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        ConfigurationError: If there's a problem loading the user configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith(('.yaml', '.yml')):
                user_config = yaml.safe_load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration format: {config_path}")
        
        logger.info(f"Loaded user configuration from {config_path}")
        return merge_configs(base_config, user_config)
    
    except Exception as e:
        logger.error(f"Error loading user configuration from {config_path}: {e}")
        raise ConfigurationError(f"Error loading user configuration: {str(e)}")

def resolve_environment_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve environment variables in the configuration.
    
    Args:
        config: Configuration dictionary with possible environment variable references
        
    Returns:
        Configuration with environment variables resolved
    """
    import os
    import re
    
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            result[key] = resolve_environment_variables(value)
        return result
    elif isinstance(config, list):
        return [resolve_environment_variables(item) for item in config]
    elif isinstance(config, str):
        # Pattern to match ${VARIABLE_NAME} or $VARIABLE_NAME
        pattern = r'\${([^}]+)}|\$([A-Za-z0-9_]+)'
        
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, f"${var_name}")  # Keep original if not found
        
        return re.sub(pattern, replace_var, config)
    else:
        return config

def resolve_config_references(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve references to other parts of the configuration using ${path.to.value} syntax.
    
    Args:
        config: Configuration dictionary with possible cross-references
        
    Returns:
        Configuration with references resolved
    """
    import re
    
    # Convert config to string, replace references, then convert back
    config_str = yaml.dump(config)
    
    # Pattern to match ${path.to.value}
    pattern = r'\${([^}]+)}'
    
    def get_value_by_path(path, cfg):
        """Get a value from the config using dot notation path."""
        parts = path.split('.')
        current = cfg
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return f"${{{path}}}"  # Path not found, return original reference
        return current
    
    def replace_ref(match):
        path = match.group(1)
        value = get_value_by_path(path, config)
        
        # If the value is a complex object, convert it to YAML string
        if isinstance(value, (dict, list)):
            return yaml.dump(value)
        
        return str(value)
    
    # Replace all references
    config_str = re.sub(pattern, replace_ref, config_str)
    
    # Convert back to dictionary
    return yaml.safe_load(config_str)

def get_configuration(
    user_config: Optional[str] = None,
    config_dir: str = "config.d",
    validate: bool = True
) -> Dict[str, Any]:
    """
    Get the complete configuration by loading and merging all config files.
    
    Args:
        user_config: Path to a user-provided configuration file (optional)
        config_dir: Path to the configuration directory (default: "config.d")
        validate: Whether to validate the configuration
        
    Returns:
        Complete configuration dictionary
    """
    # Load configurations from config.d
    config = load_config_directory(config_dir)
    
    # Merge with user-provided configuration
    if user_config:
        config = load_user_config(user_config, config)
    
    # Resolve environment variables
    config = resolve_environment_variables(config)
    
    # Resolve config cross-references
    config = resolve_config_references(config)
    
    if validate:
        validate_configuration(config)
    
    return config

def validate_configuration(config: Dict[str, Any]) -> None:
    """
    Validate that the configuration has all required sections and values.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    # Check for required top-level sections
    required_sections = ["directories", "processing", "logging"]
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Required configuration section '{section}' not found")
    
    # Check for required directory settings
    required_dirs = ["input", "output"]
    for dir_name in required_dirs:
        if dir_name not in config.get("directories", {}):
            raise ConfigurationError(f"Required directory setting '{dir_name}' not found")
            
    # Additional validation could be added here
    
def get_retrieval_system_config(config: Dict[str, Any], system_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific retrieval system.
    
    Args:
        config: Complete configuration dictionary
        system_name: Name of the retrieval system (e.g., "pathrag")
        
    Returns:
        Configuration dictionary for the specified retrieval system
        
    Raises:
        ConfigurationError: If the specified retrieval system is not configured
    """
    retrieval_systems = config.get("retrieval_systems", {})
    if system_name not in retrieval_systems:
        raise ConfigurationError(f"Retrieval system '{system_name}' not found in configuration")
    
    return retrieval_systems[system_name]
