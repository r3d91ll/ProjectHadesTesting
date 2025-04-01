# Redis Integration for PathRAG

This document provides instructions for using Redis as a high-performance caching solution for PathRAG, replacing the previous RAMDisk implementation.

## Overview

Redis is used as an in-memory database to store vectors and metadata, providing similar performance benefits to RAMDisk while adding more features:

- **Vector Storage**: Efficiently store and retrieve embedding vectors
- **Vector Search**: Search for similar vectors using cosine similarity
- **Persistence**: Optional persistence to disk for durability
- **TTL Support**: Automatic expiration of cached items
- **Monitoring**: Built-in statistics and monitoring

## Requirements

- Redis server (version 6.0 or higher)
- RediSearch module (for vector search capabilities)
- Python 3.8 or higher
- Redis Python client (`pip install redis`)
- NumPy (`pip install numpy`)

## Setup

1. Install Redis and RediSearch:
   ```bash
   sudo apt-get update
   sudo apt-get install redis-server redis-tools redis-redisearch
   ```

2. Configure Redis for optimal performance:
   ```bash
   sudo cp /home/todd/ML-Lab/New-HADES/redis_test/redis-memory-optimized.conf /etc/redis/redis.conf
   sudo systemctl restart redis-server
   ```

3. Verify Redis is running with RediSearch:
   ```bash
   redis-cli ping
   redis-cli module list
   ```

## Usage

Instead of using the original `run_unified.sh` script, use the Redis-enabled version:

```bash
./scripts/run_unified_redis.sh [--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean_db]
```

This script will automatically use Redis for caching vectors and metadata, providing similar performance benefits to RAMDisk.

## Configuration

Redis configuration is stored in `scripts/redis_config.sh`. You can modify this file to change Redis connection settings, TTL, and other parameters.

## Performance Considerations

- Redis is most effective when it has enough memory to store all vectors without swapping
- For a system with 256GB of RAM, allocate up to 200GB to Redis for optimal performance
- Monitor Redis memory usage with `redis-cli info memory`
- If Redis memory usage exceeds available RAM, consider reducing the TTL or clearing the cache more frequently

## Troubleshooting

If you encounter issues with Redis:

1. Check if Redis is running:
   ```bash
   systemctl status redis-server
   ```

2. Check Redis logs:
   ```bash
   journalctl -u redis-server
   ```

3. Verify RediSearch module is loaded:
   ```bash
   redis-cli module list
   ```

4. Test Redis connectivity:
   ```bash
   redis-cli ping
   ```

5. Check Redis memory usage:
   ```bash
   redis-cli info memory
   ```

## Reverting to RAMDisk

If you need to revert to the original RAMDisk implementation:

```bash
./scripts/run_unified.sh [--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean_db]
```

This will use the original RAMDisk configuration without Redis.
