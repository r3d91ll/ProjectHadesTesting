# Ollama Integration for PathRAG

This document describes the integration of Ollama with PathRAG for local LLM inference.

## Overview

PathRAG now supports using Ollama as a local inference provider, which allows for cost-effective development and testing without relying on OpenAI API calls. This integration enables:

- Local inference using Ollama models
- Configurable context length for handling large documents
- Seamless switching between Ollama and OpenAI
- Full telemetry logging to Arize Phoenix

## Configuration

The Ollama integration is configured through environment variables in the `.env` file:

```
# LLM Provider (openai or ollama)
LLM_PROVIDER=ollama

# Ollama API settings
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=qwen2.5-128k-64k-ctx
```

## Available Models

PathRAG has been tested with the following Ollama models:

- `qwen2.5-128k-64k-ctx`: A custom model based on Qwen 2.5 with 64K context length
- `tinyllama`: A lightweight model for quick testing
- Other models available in Ollama can also be used

## Setup Instructions

1. Install Ollama by following the instructions at [ollama.ai](https://ollama.ai)
2. Pull the desired model: `ollama pull qwen2.5-128k`
3. Run the setup script to create a custom model with extended context length:
   ```
   python dev-utilities/setup_qwen_model_properly.py
   ```
4. Update the `.env` file to use Ollama as the LLM provider

## Usage

Once configured, PathRAG will automatically use Ollama for inference:

```bash
python src/pathrag_runner.py --query "What are the key principles of RAG systems?" --session-id "test_ollama"
```

## Telemetry

All Ollama inference requests are logged to the Arize Phoenix project `pathrag-inference`. You can view the traces at:

```
http://localhost:8084/v1/traces
```

## Switching Between Providers

To switch between Ollama and OpenAI, simply update the `LLM_PROVIDER` environment variable in the `.env` file:

```
# For local development
LLM_PROVIDER=ollama

# For production
LLM_PROVIDER=openai
```

## Performance Considerations

- Ollama models run locally and utilize GPU resources
- Performance depends on your hardware capabilities
- The first query after starting Ollama may take longer as the model is loaded into memory
