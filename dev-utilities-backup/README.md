# Development Utilities

This directory contains one-time repair scripts, temporary utilities, and other scripts that shouldn't be confused with core operational code. These utilities help with development, testing, and maintenance but are not part of the main application flow.

## Directory Structure

- `ollama-setup/` - Scripts for setting up and configuring Ollama for local LLM inference
- `pathrag-monitor/` - Utilities for monitoring PathRAG performance with Arize Phoenix
- `test_phoenix_trace.py` - Script for testing Phoenix trace logging

## Ollama Setup Utilities

The `ollama-setup/` directory contains scripts for configuring Ollama as a local LLM provider:

- `setup_qwen_model_properly.py` - Creates a custom Qwen model with extended context length (64K)
- `update_ollama_env.py` - Updates the PathRAG .env file to use Ollama with TinyLlama
- `update_qwen_env.py` - Updates the PathRAG .env file to use Ollama with Qwen2.5-128k
- `update_qwen_alt_env.py` - Updates the PathRAG .env file to use Ollama with alternative Qwen model
- `Qwen2.5-128k-64k-ctx.modelfile` - Modelfile for Qwen with 64K context length

## PathRAG Monitor Utilities

The `pathrag-monitor/` directory contains utilities for monitoring PathRAG performance:

- `phoenix_connector.py` - Connector for retrieving trace data from Arize Phoenix
- `requirements.txt` - Dependencies for the PathRAG monitor

## Usage

These utilities are meant to be run as needed during development and are not part of the regular application workflow. To use a utility, run it directly with Python:

```bash
python dev-utilities/ollama-setup/setup_qwen_model_properly.py
```

## Adding New Utilities

When creating new development utilities:

1. Place them in the appropriate subdirectory or create a new one if needed
2. Document the utility's purpose in this README
3. Include clear documentation within the utility itself
4. Follow the naming convention of existing utilities
