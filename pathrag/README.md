# PathRAG with Arize Phoenix Integration

This directory contains the implementation of PathRAG (Path Retrieval Augmented Generation) with Arize Phoenix integration for performance tracking and evaluation.

## Overview

PathRAG is a retrieval-augmented generation system that uses graph-based approaches to find paths between concepts for more coherent and accurate responses. This implementation integrates with Arize Phoenix for monitoring and performance tracking, allowing you to visualize and analyze the behavior of your RAG system.

## Directory Structure

```
pathrag/
├── config/               # Configuration files
│   ├── env.template      # Template for environment variables
│   └── pathrag_config.py # Configuration utilities
├── data/                 # Data storage for document embeddings
├── logs/                 # Log files
├── src/                  # Source code
│   └── pathrag_runner.py # Main entry point
└── requirements.txt      # Python dependencies
```

## Setup Instructions

1. **Environment Setup**

   Copy the environment template and add your OpenAI API key:

   ```bash
   cp config/env.template .env
   # Edit .env and add your OpenAI API key
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Arize Phoenix Connection**

   Make sure Arize Phoenix is running as configured in your Docker Compose setup:

   ```bash
   curl http://localhost:8080/health
   ```

## Usage

### Ingesting Documents

To ingest a document into PathRAG:

```bash
python src/pathrag_runner.py --ingest /path/to/document.txt
```

### Querying PathRAG

To query PathRAG:

```bash
python src/pathrag_runner.py --query "Your question here"
```

### Evaluating Responses

To evaluate a query against ground truth:

```bash
python src/pathrag_runner.py --query "Your question here" --evaluate --ground-truth "The expected answer"
```

### Tracking Sessions

You can specify user and session IDs for tracking:

```bash
python src/pathrag_runner.py --query "Your question" --user-id "user123" --session-id "session456"
```

## Monitoring with Arize Phoenix

The implementation automatically sends telemetry data to Arize Phoenix, including:

- Query/response pairs
- Latency metrics
- Path retrieval statistics
- Evaluation metrics (when using the `--evaluate` flag)

You can access the Arize Phoenix UI at http://localhost:8080 to:

1. View trace data for all PathRAG queries
2. Analyze performance metrics like latency and token usage
3. Inspect evaluation results for different queries
4. Track model performance over time

## Customization

You can customize the behavior of PathRAG by editing the configuration file at `config/pathrag_config.py` or by setting environment variables in your `.env` file.

## License

See the project root directory for license information.
