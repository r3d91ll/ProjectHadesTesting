# ProjectHadesTesting - XnX Notation Experimental Validation

## Project Overview

This repository contains a structured experimental framework for validating the efficacy of XnX notation in improving retrieval-augmented generation systems, specifically using PathRAG and GraphRAG as baseline implementations.

The project aims to provide scientific evidence for the impact of XnX notation through rigorous benchmarking and comparison against established methods.

## Experimental Design

The experiments will proceed in three distinct phases, designed to isolate and measure the impact of both model changes and XnX notation:

### Phase 1: Environment & Methodology Validation (Original Setup)

**Purpose**:  
Establish baseline accuracy and performance, validating that the environment setup is faithful to the original papers' methodology and results.

**Procedure**:  
- Use the **exact model** as described in each paper (GraphRAG & PathRAG).
- Use original datasets, embedding models, and indexing setups explicitly mentioned.
- Reproduce as closely as possible the original experimental setups.

**Expected Outcome**:  
- Validation of computational environment and methodology.
- Results should be very close (~within 1-5%) of reported benchmarks from papers.

### Phase 2: Model Validation (Qwen 2.5 Coder Integration)

**Purpose**:  
Validate that the Qwen2.5-Coder model integration maintains comparable performance to original models, ensuring the environment and model integration is reliable and effective.

**Procedure**:  
- **Replace the original model** from each paper with **Qwen 2.5 Coder**.
- All other factors remain identical to Phase 1:
  - Dataset
  - Vector and graph indexing setup
  - Embedding model
  - Retrieval methodologies

**Expected Outcome**:  
- Comparable, though not identical, results to original benchmarks.
- Slight deviations expected but overall similar performance should be demonstrated.
- Confirmation that the Qwen2.5-Coder model works effectively within the GraphRAG and PathRAG methodologies.

### Phase 3: XnX Notation Integration Testing

**Purpose**:  
Evaluate the impact and added value of **XnX notation** within GraphRAG and PathRAG contexts.

**Procedure**:  
- Integrate **XnX notation** into each system separately (GraphRAG + XnX and PathRAG + XnX).
- Test performance using:
  1. Original models (from papers)
  2. Qwen2.5-Coder model
- Metrics comparison directly against Phase 1 (original) and Phase 2 (Qwen) benchmarks.

**Expected Outcome**:  
- Quantifiable improvement or clear impact in terms of:
  - Improved retrieval accuracy.
  - Better interpretability or explainability of retrieval paths (qualitative improvement).
  - Enhanced confidence scoring of retrieval results.
- Documentation and thorough statistical comparison with prior phases.

## Repository Structure

```
hades-xnx-validation/
├── README.md                # This file
├── pyproject.toml           # Project configuration
├── requirements.txt         # Python dependencies
├── docs/                    # Documentation
│   ├── xnx/                 # XnX notation theory and documentation
│   ├── methodology.md       # Detailed experimental methodology
│   └── results/             # Experimental results and analysis
├── data/                    # Data storage
│   ├── datasets/            # Test datasets
│   └── results/             # Experimental results
├── common/                  # Shared code
│   ├── metrics/             # Performance metrics implementation
│   ├── utils/               # Utility functions
│   └── models/              # Model interfaces
├── implementations/         # System implementations
│   ├── pathrag/             # PathRAG implementations
│   │   ├── original/        # Original PathRAG implementation
│   │   ├── qwen25/          # PathRAG with Qwen2.5
│   │   └── xnx/             # PathRAG with XnX notation
│   └── graphrag/            # GraphRAG implementations
│       ├── original/        # Original GraphRAG implementation
│       ├── qwen25/          # GraphRAG with Qwen2.5
│       └── xnx/             # GraphRAG with XnX notation
└── experiments/             # Experiment runners
    ├── phase1/              # Phase 1 experiments
    ├── phase2/              # Phase 2 experiments
    └── phase3/              # Phase 3 experiments
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git
- Ollama (for local LLM inference)

All other dependencies (Python, Neo4j) are handled through Docker containers.

### LLM Inference Options

The project supports two LLM inference options:

1. **OpenAI API** - For production use with high-quality responses
2. **Ollama** - For local development and testing without API costs

See [Ollama Integration Documentation](pathrag/docs/ollama_integration.md) for details on setting up local inference.

### Installation

#### Option 1: Manual Setup

```bash
# Clone this repository
git clone https://github.com/yourusername/ProjectHadesTesting.git
cd ProjectHadesTesting

# Set up environment and install dependencies
./scripts/setup_environment.sh
```

#### Option 2: Docker Setup (Recommended)

```bash
# Clone this repository
git clone https://github.com/yourusername/ProjectHadesTesting.git
cd ProjectHadesTesting

# Build and start Docker containers
./scripts/docker_commands.sh build
./scripts/docker_commands.sh up

# Pull required Ollama models
./scripts/docker_commands.sh pull-models
```

## Running Experiments

### Using Docker (Recommended)

```bash
# Run Phase 1 experiments
./scripts/docker_commands.sh experiment phase1-pathrag
./scripts/docker_commands.sh experiment phase1-graphrag

# Run Phase 2 experiments
./scripts/docker_commands.sh experiment phase2-pathrag
./scripts/docker_commands.sh experiment phase2-graphrag

# Run Phase 3 experiments
./scripts/docker_commands.sh experiment phase3-pathrag
./scripts/docker_commands.sh experiment phase3-graphrag
```

### Manual Execution

If not using Docker, each phase has its own experiment runner:

```bash
# Phase 1: Original paper implementations
python -m experiments.phase1.run

# Phase 2: Qwen2.5 Coder integration
python -m experiments.phase2.run

# Phase 3: XnX notation integration
python -m experiments.phase3.run
```

## Metrics and Evaluation

The project uses a standardized evaluation framework to ensure consistent comparison across experiments. Key metrics include:

- Retrieval precision/recall
- Path relevance
- Answer accuracy
- Reasoning transparency

## Acknowledgements

This project builds upon the work in the following papers:
- PathRAG: [Citation]
- GraphRAG: [Citation]

## License

[Specify license]
