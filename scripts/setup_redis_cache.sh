#!/bin/bash
# Redis Cache Setup for PathRAG
# This script sets up Redis as a high-performance cache for PathRAG
# replacing the previous RAMDisk solution

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
REDIS_CONF_SRC="${PROJECT_ROOT}/redis_test/redis-memory-optimized.conf"
REDIS_CONF_DEST="/etc/redis/redis.conf"
REDIS_LOG_DIR="/var/log/redis"
REDIS_RUN_DIR="/var/run/redis"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "This script must be run as root (use sudo)"
    exit 1
fi

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    print_error "Redis is not installed. Please install Redis first."
    print_status "You can install Redis with: sudo apt-get install redis-server redis-tools"
    exit 1
fi

# Check if RediSearch module is installed
if [ ! -f "/usr/lib/redis/modules/redisearch.so" ]; then
    print_error "RediSearch module is not installed."
    print_status "You can install RediSearch with: sudo apt-get install redis-redisearch"
    exit 1
fi

# Create necessary directories
print_status "Creating Redis directories..."
mkdir -p ${REDIS_LOG_DIR}
mkdir -p ${REDIS_RUN_DIR}
chown redis:redis ${REDIS_LOG_DIR}
chown redis:redis ${REDIS_RUN_DIR}
chmod 755 ${REDIS_LOG_DIR}
chmod 755 ${REDIS_RUN_DIR}

# Backup existing Redis configuration
if [ -f "${REDIS_CONF_DEST}" ]; then
    BACKUP_FILE="${REDIS_CONF_DEST}.bak.$(date +%Y%m%d%H%M%S)"
    print_status "Backing up existing Redis configuration to ${BACKUP_FILE}"
    cp ${REDIS_CONF_DEST} ${BACKUP_FILE}
fi

# Copy optimized configuration
print_status "Installing optimized Redis configuration..."
cp ${REDIS_CONF_SRC} ${REDIS_CONF_DEST}
chown redis:redis ${REDIS_CONF_DEST}
chmod 640 ${REDIS_CONF_DEST}

# Adjust memory settings based on available RAM
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$((${TOTAL_MEM_KB} / 1024 / 1024))
REDIS_MEM_GB=$((${TOTAL_MEM_GB} * 80 / 100)) # Use 80% of available RAM

print_status "System has ${TOTAL_MEM_GB}GB of RAM, allocating ${REDIS_MEM_GB}GB to Redis"

# Update the maxmemory setting in the Redis configuration
sed -i "s/maxmemory [0-9]*gb/maxmemory ${REDIS_MEM_GB}gb/" ${REDIS_CONF_DEST}

# Restart Redis service
print_status "Restarting Redis service..."
systemctl restart redis-server

# Wait for Redis to start
sleep 2

# Check if Redis is running
if systemctl is-active --quiet redis-server; then
    print_status "Redis server is running with optimized configuration"
else
    print_error "Failed to start Redis server"
    print_status "Check logs with: journalctl -u redis-server"
    exit 1
fi

# Verify RediSearch module is loaded
if redis-cli module list | grep -q "ft"; then
    print_status "RediSearch module is loaded successfully"
else
    print_error "RediSearch module is not loaded"
    print_status "Check Redis logs for errors"
    exit 1
fi

# Create a symbolic link to the Redis cache module in the src directory
if [ -f "${PROJECT_ROOT}/src/redis_cache.py" ]; then
    print_status "Redis cache module is already installed"
else
    print_error "Redis cache module not found at ${PROJECT_ROOT}/src/redis_cache.py"
    exit 1
fi

print_status "Testing Redis vector operations..."
python3 ${PROJECT_ROOT}/redis_test/test_vector_search.py --num-vectors 10 --num-queries 2

print_status "============================================="
print_status "Redis cache setup complete!"
print_status "Redis is now configured to use up to ${REDIS_MEM_GB}GB of RAM"
print_status "To use Redis cache in your PathRAG application:"
print_status "1. Import the RedisCache class from src.redis_cache"
print_status "2. Initialize the cache with appropriate parameters"
print_status "3. Use the cache methods for vector storage and retrieval"
print_status "============================================="
