# Redis Test 1: Vector Database Implementation

## Test Date
March 31, 2025

## Test Objective
Replace the RAM disk storage approach with Redis for vector storage in the RAG Dataset Builder, providing improved scalability, performance, and persistence for vector embeddings and document metadata.

## Background
The previous experiments (CPU Test 1 and GPU Test) successfully implemented and validated the RAM disk pipeline for document collection, processing, and embedding generation. While the RAM disk approach provided good I/O performance, it has limitations in terms of scalability, persistence, and distributed capabilities. Redis offers a more robust solution for vector storage and retrieval, particularly for large-scale RAG applications.

## Proposed Changes

### 1. Redis Integration
- Implement a Redis client for vector storage and retrieval
- Configure Redis for vector similarity search operations
- Develop a migration utility to transfer existing vector databases to Redis

### 2. Vector Storage Architecture
- Use Redis Stack with RediSearch and RedisJSON modules for vector search capabilities
- Implement efficient indexing for vector similarity search
- Store document metadata alongside vectors for efficient retrieval

### 3. Configuration Updates
- Add Redis connection settings to the configuration files
- Implement fallback mechanisms for offline operation
- Configure Redis persistence options (RDB snapshots and/or AOF logs)

### 4. Performance Optimizations
- Implement batch operations for vector storage and retrieval
- Configure Redis memory settings for optimal performance
- Implement connection pooling for concurrent operations

## Implementation Plan

### Overview
The implementation will focus on creating a separate Redis vector store module that can be used by multiple components, not just the RAG Dataset Builder. This modular approach ensures that Redis functionality is encapsulated in a dedicated module, promoting code reuse and maintainability.

### Task 1: Create Redis Vector Store Module Structure
1. Create a new directory structure for the redis-vector-store module
2. Set up basic files including __init__.py, client.py, config.py, and utils.py
3. Create a proper package structure with setup.py for installation
4. Define interfaces and abstract classes for the vector store operations

**Directory Structure:**
```
redis-vector-store/
├── redis_vector_store/
│   ├── __init__.py
│   ├── client.py         # Main Redis client implementation
│   ├── config.py         # Configuration handling
│   ├── schema.py         # Data schemas and validation
│   ├── search.py         # Vector search operations
│   └── utils.py          # Utility functions
├── config/
│   ├── default.yaml      # Default configuration
│   └── production.yaml   # Production configuration template
├── tests/
│   ├── __init__.py
│   ├── test_client.py
│   └── test_search.py
├── setup.py              # Package installation
└── README.md             # Module documentation
```

### Task 2: Implement Redis Vector Store Core Functionality
1. Implement the core Redis client with connection pooling
2. Develop vector storage and retrieval operations
3. Implement vector similarity search functionality
4. Add comprehensive error handling and retry mechanisms
5. Create utility functions for data serialization/deserialization

**Key Classes:**
- `RedisVectorStore`: Main client class for interacting with Redis
- `VectorSearchEngine`: Handles vector similarity search operations
- `SchemaValidator`: Validates data against defined schemas
- `ConnectionManager`: Manages Redis connections and pooling

### Task 3: Create Redis Configuration Files
1. Define YAML configuration structure for Redis settings
2. Create default configuration with sensible values
3. Add documentation for each configuration option
4. Implement configuration validation and loading

**Configuration Example:**
```yaml
# Redis connection settings
redis:
  host: localhost
  port: 6379
  password: null  # Set to null for no password
  database: 0
  prefix: "pathrag:"
  timeout: 5.0
  pool_size: 10
  
  # Vector search configuration
  vector_search:
    index_name: "embeddings_idx"
    similarity_metric: "COSINE"
    initial_cap: 1000000
    
  # Persistence configuration
  persistence:
    enabled: true
    rdb_save_frequency: 900
    aof_enabled: true
```

### Task 4: Implement Python Startup Script for RAG Dataset Builder
1. Create a new Python-based startup script (run.py)
2. Implement command-line argument parsing
3. Add configuration loading and validation
4. Integrate with the Redis vector store module
5. Ensure backward compatibility with existing options

**Startup Script Structure:**
```python
#!/usr/bin/env python3
"""
RAG Dataset Builder - Main Entry Point
This script replaces the bash-based run_unified.sh with a cleaner Python implementation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Import the Redis module
from redis_vector_store import RedisVectorStore

# Import the RAG Dataset Builder
from rag_dataset_builder.core import Config, DatasetBuilder

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RAG Dataset Builder")
    
    # Processing mode
    parser.add_argument("--gpu", action="store_true", help="Use GPU for processing")
    
    # RAG implementation
    parser.add_argument("--pathrag", action="store_true", help="Use PathRAG implementation")
    parser.add_argument("--graphrag", action="store_true", help="Use GraphRAG implementation")
    parser.add_argument("--literag", action="store_true", help="Use LiteRAG implementation")
    
    # Other options
    parser.add_argument("--threads", type=int, default=24, help="Number of threads to use")
    parser.add_argument("--clean_db", action="store_true", help="Clean existing database before processing")
    parser.add_argument("--config_dir", type=str, default="config.d", help="Configuration directory")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("rag_dataset_builder")
    
    # Determine RAG implementation
    if args.pathrag:
        rag_impl = "pathrag"
    elif args.graphrag:
        rag_impl = "graphrag"
    elif args.literag:
        rag_impl = "literag"
    else:
        rag_impl = "pathrag"  # Default
    
    # Load configuration
    config = Config(config_dir=args.config_dir)
    
    # Initialize Redis vector store
    redis_store = RedisVectorStore.from_config(config.get("redis", {}))
    
    # Initialize dataset builder
    builder = DatasetBuilder(
        config=config,
        rag_impl=rag_impl,
        use_gpu=args.gpu,
        threads=args.threads,
        clean_db=args.clean_db,
        vector_store=redis_store
    )
    
    # Run the dataset builder
    builder.run()
    
    logger.info(f"RAG Dataset Builder ({rag_impl}) completed successfully.")

if __name__ == "__main__":
    main()
```

### Task 5: Develop Migration Utility
1. Create a utility script to migrate existing vector databases to Redis
2. Implement functions to read existing embeddings and metadata
3. Add batch processing for efficient migration
4. Include validation and error handling
5. Provide progress reporting and logging

### Task 6: Implement Integration Tests
1. Develop unit tests for the Redis vector store module
2. Create integration tests for the complete workflow
3. Implement performance benchmarks
4. Add test coverage reporting
5. Create CI/CD pipeline for automated testing

### Task 7: Update Documentation
1. Update project documentation with Redis integration details
2. Create installation and configuration guides
3. Add usage examples and best practices
4. Document migration process from RAM disk to Redis
5. Include performance tuning recommendations

## Expected Results
1. Improved scalability for large vector databases
2. Better performance for vector similarity search operations
3. Reliable persistence with automatic recovery
4. Support for distributed deployments if needed

## Test Execution
To run this test:
```bash
# Create a new branch for Redis implementation
git checkout -b feature/redis-vector-storage

# Run the Redis implementation test
cd /home/todd/ML-Lab/New-HADES/rag-dataset-builder
sudo ./scripts/run_redis_test.sh --pathrag
```

## Test Status
Planned for implementation starting March 31, 2025

## Implementation Details

### Redis Configuration
```yaml
# Redis configuration for vector storage
redis:
  host: localhost
  port: 6379
  password: null  # Set to null for no password
  database: 0
  prefix: "pathrag:"  # Prefix for all keys
  timeout: 5.0  # Connection timeout in seconds
  pool_size: 10  # Connection pool size
  vector_dimension: 768  # Dimension of vectors (matches nomic-embed-text)
  
  # Vector search configuration
  vector_search:
    index_name: "embeddings_idx"
    similarity_metric: "COSINE"  # Options: COSINE, IP, L2
    initial_cap: 1000000  # Initial capacity for vector index
    
  # Persistence configuration
  persistence:
    enabled: true
    rdb_save_frequency: 900  # Save RDB snapshot every 15 minutes
    aof_enabled: true  # Enable AOF log
```

### Vector Storage Schema
```json
{
  "document": {
    "id": "unique_document_id",
    "metadata": {
      "title": "Document Title",
      "source": "Document Source",
      "domain": "Knowledge Domain",
      "file_path": "Path to original document",
      "created_at": "ISO timestamp",
      "updated_at": "ISO timestamp"
    }
  },
  "chunks": [
    {
      "id": "unique_chunk_id",
      "document_id": "parent_document_id",
      "text": "Chunk text content",
      "metadata": {
        "position": 0,
        "length": 500,
        "section": "document section"
      },
      "vector": [0.1, 0.2, ...],  # Embedding vector
      "vector_model": "nomic-embed-text"
    }
  ]
}
```

## Performance Metrics to Collect
1. **Vector Storage Speed**: Time to store vectors in Redis vs. RAM disk
2. **Vector Retrieval Speed**: Time to retrieve vectors by ID
3. **Vector Search Speed**: Time to perform similarity search operations
4. **Memory Usage**: Redis memory consumption vs. RAM disk
5. **Scalability**: Performance with increasing vector database size
6. **Persistence Overhead**: Impact of persistence on performance

## Next Steps After Redis Implementation
1. Implement PathRAG inference using the Redis vector database
2. Build a chat interface for interactive testing
3. Evaluate end-to-end RAG performance with different configurations
4. Document best practices for production deployment

## References
1. Redis Vector Similarity Search: https://redis.io/docs/stack/search/reference/vectors/
2. Redis JSON: https://redis.io/docs/stack/json/
3. Redis Persistence: https://redis.io/topics/persistence
