#!/bin/bash
# Cleanup Utility for PathRAG Dataset Builder
# Cleans up temporary files and RAM disks created during dataset building

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "PathRAG Dataset Builder Cleanup Utility"
echo "======================================"

# Parse command line arguments
CLEAN_RAMDISKS=true
CLEAN_TEMP=true
CLEAN_LOGS=false
CLEAN_DATABASES=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-ramdisks)
      CLEAN_RAMDISKS=false
      shift
      ;;
    --no-temp)
      CLEAN_TEMP=false
      shift
      ;;
    --logs)
      CLEAN_LOGS=true
      shift
      ;;
    --databases)
      CLEAN_DATABASES=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-ramdisks] [--no-temp] [--logs] [--databases]"
      exit 1
      ;;
  esac
done

# Clean up RAM disks
if [ "$CLEAN_RAMDISKS" = true ]; then
    echo "Cleaning up RAM disks..."
    
    # Stop any running lsyncd processes
    echo "Stopping lsyncd processes..."
    sudo pkill -f "lsyncd.*ramdisk" || true
    
    # Unmount RAM disks
    echo "Unmounting RAM disks..."
    sudo umount /tmp/ramdisk_source_documents 2>/dev/null || true
    sudo umount /tmp/ramdisk_rag_databases 2>/dev/null || true
    
    echo "RAM disk cleanup complete."
fi

# Clean up temporary files
if [ "$CLEAN_TEMP" = true ]; then
    echo "Cleaning up temporary files..."
    rm -f /tmp/lsyncd_*.conf /tmp/lsyncd_*.log /tmp/lsyncd_*.status 2>/dev/null || true
    echo "Temporary file cleanup complete."
fi

# Clean up log files
if [ "$CLEAN_LOGS" = true ]; then
    echo "Cleaning up log files..."
    read -p "Are you sure you want to delete all log files? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${PROJECT_ROOT}/../logs"/*
        echo "Log file cleanup complete."
    else
        echo "Log file cleanup skipped."
    fi
fi

# Clean up database files
if [ "$CLEAN_DATABASES" = true ]; then
    echo "Cleaning up database files..."
    read -p "Are you sure you want to delete all database files? This cannot be undone! (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Please type 'DELETE' to confirm: " confirm
        if [ "$confirm" = "DELETE" ]; then
            rm -rf "${PROJECT_ROOT}/../rag_databases"/*
            echo "Database cleanup complete."
        else
            echo "Database cleanup skipped."
        fi
    else
        echo "Database cleanup skipped."
    fi
fi

echo "Cleanup complete."
