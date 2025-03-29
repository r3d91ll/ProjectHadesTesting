# Development Utilities

This directory contains one-time repair scripts, temporary utilities, and other scripts that shouldn't be confused with core operational code. These utilities help with development, testing, and maintenance but are not part of the main application flow.

## Important Note

The key utilities have been migrated to their proper locations within the application:

- `ollama-setup/` → `/pathrag/utils/setup/` - Scripts for Ollama configuration
- `pathrag-monitor/` → `/pathrag/utils/monitor/` - PathRAG monitoring utilities
- Phoenix integration tools → `/pathrag/utils/phoenix/` - Arize Phoenix integration

This directory now only contains temporary scripts and one-time utilities that aren't part of the regular workflow.

## Remaining Utilities

This directory still contains some useful scripts that may be needed occasionally:

- `build_rag_dataset.py` - Script for building RAG datasets
- `semantic_scholar_dataset_explorer.py` - Tool for exploring Semantic Scholar datasets
- `run_dataset_builder.py` - Runner script for the dataset builder
- `debug_processor.py` - Debugging tool for document processors

## Usage

These utilities are meant to be run as needed during development and are not part of the regular application workflow. To use a utility, run it directly with Python:

```bash
python dev-utilities/build_rag_dataset.py
```

## Adding New Utilities

When creating new development utilities:

1. **Consider placement carefully**: 
   - For one-time scripts or temporary utilities, place them in this directory
   - For permanent utilities that will be used regularly, place them in the appropriate application directory:
     - PathRAG utilities → `/pathrag/utils/`
     - Dataset builder utilities → `/rag-dataset-builder/utils/`

2. Document the utility's purpose in the appropriate README
3. Include clear documentation within the utility itself
4. Follow the naming convention of existing utilities
