# HADES Unified System Documentation

## Overview

The HADES Unified System provides a comprehensive interface for running the PathRAG system with Redis integration. It's designed to be modular and extensible, supporting both the current dataset creation/embedding process and future inference/retrieval components.

## Architecture

The system is built around a modular architecture with the following components:

1. **Dataset Creation Module**: Handles the process of loading documents into Redis, running the embedding process, and exporting the final database to disk.

2. **Inference Module** (Future): Will handle inference using a specified RAG database.

3. **Retrieval Module** (Future): Will handle retrieval using a specified RAG database and query.

4. **Redis Integration**: Provides high-performance in-memory caching for all components, significantly improving performance compared to disk-based operations.

5. **Database Selection**: Allows users to select which embedded database to use for inference and retrieval.

## Usage

### Dataset Creation

```bash
# Activate the virtual environment
source .venv/bin/activate

# Create a new dataset with GPU acceleration
python hades_unified.py create --source-dir /path/to/source --output-dir /path/to/output --gpu

# Create a new dataset with custom Redis configuration
python hades_unified.py create --source-dir /path/to/source --output-dir /path/to/output --redis-host localhost --redis-port 6379 --redis-db 0 --redis-prefix pathrag
```

### Inference (Future)

```bash
# Run inference using a specified database
python hades_unified.py infer --database /path/to/database --gpu
```

### Retrieval (Future)

```bash
# Run retrieval using a specified database and query
python hades_unified.py retrieve --database /path/to/database --query "your query" --gpu
```

## Command-Line Options

### Common Options

- `--redis-host`: Redis host (default: localhost)
- `--redis-port`: Redis port (default: 6379)
- `--redis-db`: Redis database (default: 0)
- `--redis-password`: Redis password (default: empty)
- `--redis-prefix`: Redis key prefix (default: pathrag)
- `--gpu`: Use GPU for acceleration

### Dataset Creation Options

- `--source-dir`: Directory containing source documents
- `--output-dir`: Directory to store the processed data
- `--threads`: Number of threads for parallel loading (default: 16)

### Inference Options (Future)

- `--database`: Path to the RAG database

### Retrieval Options (Future)

- `--database`: Path to the RAG database
- `--query`: Query for retrieval

## Workflow

### Dataset Creation Workflow

1. **Environment Setup**: Sets Redis environment variables and connects to Redis.
2. **Redis Preparation**: Clears the Redis database to ensure a clean start.
3. **Document Preloading**: Preloads all source documents into Redis for high-performance access.
4. **Embedding Process**: Runs the PathRAG embedding process with Redis integration.
5. **Data Export**: Exports the final data from Redis to disk for permanent storage.

### Inference Workflow (Future)

1. **Database Selection**: Allows users to select which embedded database to use.
2. **Redis Loading**: Loads the selected database into Redis for high-performance access.
3. **Inference Process**: Runs the inference process using the selected database.

### Retrieval Workflow (Future)

1. **Database Selection**: Allows users to select which embedded database to use.
2. **Redis Loading**: Loads the selected database into Redis for high-performance access.
3. **Query Processing**: Processes the user's query and retrieves relevant information.
4. **Result Presentation**: Presents the retrieval results to the user.

## Redis Integration Benefits

The Redis integration provides several key benefits:

1. **High-Performance Caching**: Redis provides in-memory caching, significantly improving performance compared to disk-based operations.
2. **Vector Search Capabilities**: Redis with RediSearch supports vector search operations, which are crucial for efficient retrieval.
3. **TTL Support**: Redis supports time-to-live (TTL) for cached items, allowing for automatic cleanup of old data.
4. **Better Memory Management**: Redis provides better memory management compared to the previous RAMDisk approach.

## Future Extensions

The unified system is designed to be easily extended with additional components:

1. **Web Interface**: A web interface for interacting with the system.
2. **API Integration**: Integration with external APIs for enhanced functionality.
3. **Multi-Database Support**: Support for multiple databases with different configurations.
4. **Distributed Processing**: Support for distributed processing across multiple machines.

## Troubleshooting

### Common Issues

1. **Redis Connection Issues**: Ensure Redis is running and accessible at the specified host and port.
2. **Memory Issues**: If Redis runs out of memory, consider increasing the available memory or using a smaller dataset.
3. **Permission Issues**: Ensure the user has the necessary permissions to read source documents and write to the output directory.

### Logs

The system logs detailed information about its operations, which can be useful for troubleshooting:

- **Info Logs**: Provide information about the system's normal operations.
- **Warning Logs**: Indicate potential issues that might require attention.
- **Error Logs**: Indicate errors that prevent the system from functioning correctly.

## Conclusion

The HADES Unified System provides a comprehensive interface for running the PathRAG system with Redis integration. Its modular architecture allows for easy extension with additional components, making it a flexible solution for a wide range of use cases.
