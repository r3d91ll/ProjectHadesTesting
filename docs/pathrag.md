# PathRAG: Path Retrieval Augmented Generation

This document provides an overview of the PathRAG implementation in the HADES project, including its architecture, components, and various implementations.

## Overview

PathRAG (Path Retrieval Augmented Generation) is a graph-based RAG system that enhances traditional retrieval-augmented generation by finding meaningful paths between concepts to produce more coherent and accurate responses. The system integrates with Arize Phoenix for monitoring and performance evaluation.

## Directory Structure

The PathRAG implementation is organized across two main directories:

### 1. Main Framework (`/pathrag`)

```
pathrag/
├── config/               # Configuration files
│   ├── env.template      # Template for environment variables
│   └── pathrag_config.py # Configuration utilities
├── data/                 # Data storage for document embeddings
├── database/             # Vector database storage
├── logs/                 # Log files
├── src/                  # Source code (13 modules)
│   └── pathrag_runner.py # Main entry point
└── requirements.txt      # Python dependencies
```

This directory contains the core PathRAG framework and infrastructure, including the main implementation, configuration files, and supporting modules.

### 2. Model Implementations (`/implementations/pathrag`)

```
implementations/pathrag/
├── arize_integration/    # Arize Phoenix integration code
├── original/             # Original baseline implementation
├── qwen25/               # Implementation using Qwen 2.5 model
└── xnx/                  # XnX notation implementation
```

This directory contains specialized implementations of PathRAG for different models and approaches, allowing for experimentation with various LLMs while sharing the core infrastructure.

## Setup and Configuration

### Environment Setup

1. Create the environment file:
   ```bash
   cd pathrag
   cp config/env.template .env
   # Edit .env to add your API keys and configuration
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Database Configuration

PathRAG uses a vector database for storing embeddings. The database configuration is specified in the `.env` file:

- Default storage location: `/rag_databases/current/`
- Can be customized in the configuration

## Key Components

### 1. Path Finder

The PathRAG system uses graph traversal to find meaningful paths between concepts mentioned in the user query, leveraging:

- Knowledge graph structure
- Entity disambiguation
- Relevance scoring

### 2. Context Enrichment

Once paths are identified, the system:
- Gathers documents and facts along the path
- Ranks them by relevance
- Constructs a coherent context

### 3. Generation

The enriched context is passed to an LLM with appropriate prompting to generate accurate responses that explain the path of reasoning.

## Model Implementations

### Original Implementation

The baseline implementation uses OpenAI models and follows the approach described in the PathRAG paper.

### Qwen 2.5 Implementation

This variant uses the Qwen 2.5 model with local inference, providing:
- Lower latency
- Reduced API costs
- Custom reasoning capabilities

### XnX Implementation

An experimental implementation that incorporates XnX notation for improved explainability and traceability in reasoning paths.

### Arize Integration

All implementations include Arize Phoenix integration for:
- Tracing evaluations
- Performance monitoring
- Comparing model outputs
- Visualizing retrieval effectiveness

## Evaluation Methodology

PathRAG is evaluated using:

1. **Retrieval Metrics**
   - Precision@k (k=1,3,5)
   - Recall@k (k=1,3,5)
   - Mean Reciprocal Rank (MRR)

2. **Generation Metrics**
   - ROUGE-L
   - BLEU
   - BERTScore

3. **Path Quality Metrics**
   - Path relevance
   - Path coherence
   - Path factuality

## Usage

To run PathRAG:

```python
from pathrag.src.pathrag_runner import PathRAGRunner

# Initialize with configuration
runner = PathRAGRunner(config_path="path/to/config.yaml")

# Run inference
response = runner.run(query="What is the relationship between mitochondria and ATP production?")

# Access results
answer = response.answer
path = response.reasoning_path
evidence = response.supporting_evidence
```

## Monitoring and Visualization

Performance metrics and traces are available through the Arize Phoenix dashboard. To access:

1. Ensure the Arize service is running
2. Navigate to the Phoenix UI (typically http://localhost:6006)
3. Select the PathRAG project to view traces, metrics, and evaluation results

## References

- Original PathRAG paper
- Related work on graph-based retrieval
- Arize Phoenix documentation
