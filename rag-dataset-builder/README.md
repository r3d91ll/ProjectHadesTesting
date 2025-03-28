# RAG Dataset Builder

A flexible, memory-efficient tool for building datasets for Retrieval-Augmented Generation (RAG) systems.

## Features

- **Data Collection**: Gather documents from diverse sources (research papers, documentation, code)
- **Efficient Processing**: Incremental processing with minimal memory footprint
- **Local Embeddings**: Generate embeddings locally using GPU acceleration
- **Flexible Output Formats**: Support for multiple RAG implementations
- **Knowledge Graph Construction**: Optional path-based retrieval structure
- **Performance Monitoring**: Integration with monitoring tools like Arize Phoenix

## Supported Output Formats

- **PathRAG**: Knowledge graph structure for path-based retrieval
- **Vector Database**: Standard vector store format (FAISS, Chroma, etc.)
- **Hugging Face Dataset**: Format compatible with Hugging Face datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-dataset-builder.git
cd rag-dataset-builder

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python src/main.py --data-dir /path/to/data --output-dir /path/to/output --format pathrag

# Advanced usage
python src/main.py --data-dir /path/to/data --output-dir /path/to/output --format vector_db --embedding-model all-MiniLM-L6-v2 --chunk-size 300 --chunk-overlap 50
```

## Dataset Collection

The tool supports collecting data from various sources:

```bash
# Collect papers from arXiv
python src/collectors/arxiv_collector.py --search-term "actor network theory" --max-results 100 --output-dir ./data/arxiv

# Collect web documentation
python src/collectors/documentation_collector.py --source python --output-dir ./data/documentation

# Collect code samples
python src/collectors/code_collector.py --dataset codeparrot/codeparrot-clean --max-samples 1000 --output-dir ./data/code
```

## Configuration

See the `config/` directory for example configuration files and templates.

## Examples

Complete examples can be found in the `examples/` directory.

## License

MIT

## Citation

If you use this tool in your work, please cite:

```bibtex
@software{rag_dataset_builder,
  author = {Your Name},
  title = {RAG Dataset Builder},
  url = {https://github.com/yourusername/rag-dataset-builder},
  year = {2025},
}
```
