# Ollama Setup Utilities

This directory contains utilities for setting up and configuring Ollama as a local LLM provider for PathRAG. These scripts are intended for development and testing purposes to avoid costs associated with OpenAI API calls.

## Available Scripts

### `setup_qwen_model_properly.py`

Creates a custom Qwen model with extended context length (64K) based on the existing qwen2.5-128k model.

**Usage:**
```bash
python setup_qwen_model_properly.py
```

**What it does:**
1. Checks if the base model (qwen2.5-128k) is available
2. Creates a Modelfile with 64K context length
3. Creates a custom model in Ollama
4. Updates the PathRAG .env file to use this model

### `update_ollama_env.py`

Updates the PathRAG .env file to use Ollama with TinyLlama, a lightweight model for quick testing.

**Usage:**
```bash
python update_ollama_env.py
```

### `update_qwen_env.py`

Updates the PathRAG .env file to use Ollama with the Qwen2.5-128k model.

**Usage:**
```bash
python update_qwen_env.py
```

### `update_qwen_alt_env.py`

Updates the PathRAG .env file to use Ollama with an alternative Qwen model (mbenhamd/qwen2.5-14b-instruct-cline-128k-q8_0).

**Usage:**
```bash
python update_qwen_alt_env.py
```

### `Qwen2.5-128k-64k-ctx.modelfile`

Modelfile for creating a custom Qwen model with 64K context length. This file is used by the `setup_qwen_model_properly.py` script.

## Prerequisites

- Ollama must be installed and running
- The base models (qwen2.5-128k, tinyllama) should be available in Ollama

## Notes

- These scripts are one-time utilities for setting up the development environment
- The created models will be available for use in PathRAG via the .env configuration
- GPU resources are required for optimal performance
