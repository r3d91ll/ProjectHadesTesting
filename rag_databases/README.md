# RAG Databases Directory

## Purpose
This directory serves as the default storage location for vector databases used in Retrieval-Augmented Generation (RAG) experiments within the HADES project. 

## Structure
- `/current/`: The active directory where generated databases are stored by default

## Usage
This directory is automatically used by:
1. The `rag-dataset-builder` module to store processed datasets and vector databases
2. The RAG evaluation workflows to retrieve information during experiments

## Configuration
The path to this directory is configured in:
- [/rag-dataset-builder/config/config.yaml](cci:7://file:///home/todd/ML-Lab/New-HADES/rag-dataset-builder/config/config.yaml:0:0-0:0): Sets this as the default output directory
- Other configuration files in the archived configs

## Maintenance
The contents of this directory are managed by:
- [/scripts/utility/cleanup.sh](cci:7://file:///home/todd/ML-Lab/New-HADES/scripts/utility/cleanup.sh:0:0-0:0): Periodically cleans the directory with `rm -rf rag_databases/current/*`

## Notes
- This directory may appear empty if no RAG datasets have been generated yet or after cleanup scripts have run
- Do not delete this directory structure as it's required by the RAG components of the system
- Large vector databases will be stored here when RAG experiments are active