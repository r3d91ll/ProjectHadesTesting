#!/bin/bash
# RAM Disk Embedder for PathRAG
# This script creates RAM disks for both source documents and output database,
# syncs them with disk storage using lsyncd, and runs the dataset builder with
# configurable CPU/GPU-based embedding for maximum performance

set -e

# Default processing mode
PROCESSING_MODE="cpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      PROCESSING_MODE="gpu"
      shift
      ;;
    --cpu)
      PROCESSING_MODE="cpu"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--cpu|--gpu]"
      exit 1
      ;;
  esac
done

# Configuration paths
CONFIG_D_DIR="/home/todd/ML-Lab/New-HADES/rag-dataset-builder/config.d"
NUM_THREADS=24  # Threadripper has 24 cores/48 threads

# Load RAM disk settings from config file
RAMDISK_CONFIG="${CONFIG_D_DIR}/20-ramdisk.yaml"

# Default values in case config file doesn't exist
SOURCE_RAMDISK_SIZE=20G
DB_RAMDISK_SIZE=30G  # Increased to 30GB by default
SOURCE_RAMDISK_PATH="/tmp/ramdisk_source_documents"
DB_RAMDISK_PATH="/tmp/ramdisk_rag_databases"
SOURCE_DIR="/home/todd/ML-Lab/New-HADES/source_documents"
DB_DIR="/home/todd/ML-Lab/New-HADES/rag_databases"

# Parse YAML config file if it exists
if [ -f "${RAMDISK_CONFIG}" ]; then
    echo "Loading RAM disk configuration from ${RAMDISK_CONFIG}..."
    
    # Extract values using grep and sed
    if grep -q "source_size:" "${RAMDISK_CONFIG}"; then
        SOURCE_RAMDISK_SIZE=$(grep "source_size:" "${RAMDISK_CONFIG}" | sed 's/.*source_size: *"\(.*\)".*/\1/')
    fi
    
    if grep -q "output_size:" "${RAMDISK_CONFIG}"; then
        DB_RAMDISK_SIZE=$(grep "output_size:" "${RAMDISK_CONFIG}" | sed 's/.*output_size: *"\(.*\)".*/\1/')
    fi
    
    if grep -q "source_path:" "${RAMDISK_CONFIG}"; then
        SOURCE_RAMDISK_PATH=$(grep "source_path:" "${RAMDISK_CONFIG}" | sed 's/.*source_path: *"\(.*\)".*/\1/')
    fi
    
    if grep -q "output_path:" "${RAMDISK_CONFIG}"; then
        DB_RAMDISK_PATH=$(grep "output_path:" "${RAMDISK_CONFIG}" | sed 's/.*output_path: *"\(.*\)".*/\1/')
    fi
    
    if grep -q "source_disk_path:" "${RAMDISK_CONFIG}"; then
        SOURCE_DISK_PATH=$(grep "source_disk_path:" "${RAMDISK_CONFIG}" | sed 's/.*source_disk_path: *"\(.*\)".*/\1/')
        # Convert relative path to absolute if needed
        if [[ "${SOURCE_DISK_PATH}" == "../"* ]]; then
            SOURCE_DIR="/home/todd/ML-Lab/New-HADES/$(echo ${SOURCE_DISK_PATH} | sed 's/^\.\.\/\(.*\)/\1/')"
        elif [[ "${SOURCE_DISK_PATH}" == "./"* ]]; then
            SOURCE_DIR="/home/todd/ML-Lab/New-HADES/rag-dataset-builder/$(echo ${SOURCE_DISK_PATH} | sed 's/^\.\//\(.*\)/\1/')"
        elif [[ "${SOURCE_DISK_PATH}" == /* ]]; then
            SOURCE_DIR="${SOURCE_DISK_PATH}"
        fi
    fi
    
    if grep -q "output_disk_path:" "${RAMDISK_CONFIG}"; then
        OUTPUT_DISK_PATH=$(grep "output_disk_path:" "${RAMDISK_CONFIG}" | sed 's/.*output_disk_path: *"\(.*\)".*/\1/')
        # Convert relative path to absolute if needed
        if [[ "${OUTPUT_DISK_PATH}" == "../"* ]]; then
            DB_DIR="/home/todd/ML-Lab/New-HADES/$(echo ${OUTPUT_DISK_PATH} | sed 's/^\.\.\/\(.*\)/\1/')"
        elif [[ "${OUTPUT_DISK_PATH}" == "./"* ]]; then
            DB_DIR="/home/todd/ML-Lab/New-HADES/rag-dataset-builder/$(echo ${OUTPUT_DISK_PATH} | sed 's/^\.\//\(.*\)/\1/')"
        elif [[ "${OUTPUT_DISK_PATH}" == /* ]]; then
            DB_DIR="${OUTPUT_DISK_PATH}"
        fi
    fi
fi

echo "Using RAM disk settings:"
echo "Source RAM disk: ${SOURCE_RAMDISK_SIZE} at ${SOURCE_RAMDISK_PATH}"
echo "Output RAM disk: ${DB_RAMDISK_SIZE} at ${DB_RAMDISK_PATH}"
echo "Source directory: ${SOURCE_DIR}"
echo "Output directory: ${DB_DIR}"

# Set up timestamp and logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set output subdirectory based on processing mode
if [ "$PROCESSING_MODE" = "cpu" ]; then
    OUTPUT_SUBDIR="pathRAG_CPU"
    CONFIG_SUFFIX="35-cpu-embedder.yaml"
    echo "Using CPU-based embedding with ${NUM_THREADS} threads"
else
    OUTPUT_SUBDIR="pathRAG_GPU"
    CONFIG_SUFFIX="30-embedders.yaml"  # Default GPU config
    echo "Using GPU-based embedding"
fi

LOG_FILE="/home/todd/ML-Lab/New-HADES/logs/ramdisk_embedding_${PROCESSING_MODE}_${TIMESTAMP}.log"
mkdir -p $(dirname "${LOG_FILE}")
touch "${LOG_FILE}"

# Create source documents RAM disk
echo "Creating ${SOURCE_RAMDISK_SIZE} RAM disk for source documents at ${SOURCE_RAMDISK_PATH}..." | tee -a ${LOG_FILE}
sudo mkdir -p ${SOURCE_RAMDISK_PATH}
sudo mount -t tmpfs -o size=${SOURCE_RAMDISK_SIZE} tmpfs ${SOURCE_RAMDISK_PATH}
sudo chmod 777 ${SOURCE_RAMDISK_PATH}

# Create database RAM disk
echo "Creating ${DB_RAMDISK_SIZE} RAM disk for database at ${DB_RAMDISK_PATH}..." | tee -a ${LOG_FILE}
sudo mkdir -p ${DB_RAMDISK_PATH}
sudo mount -t tmpfs -o size=${DB_RAMDISK_SIZE} tmpfs ${DB_RAMDISK_PATH}
sudo chmod 777 ${DB_RAMDISK_PATH}

# First, copy source documents to RAM disk (faster initial copy than lsyncd)
echo "Copying source documents to RAM disk..." | tee -a ${LOG_FILE}
echo "Start time: $(date)" | tee -a ${LOG_FILE}
rsync -av --progress ${SOURCE_DIR}/ ${SOURCE_RAMDISK_PATH}/ 2>&1 | tee -a ${LOG_FILE}
echo "Copy completed at: $(date)" | tee -a ${LOG_FILE}

# Create lsyncd configuration for source documents (RAM disk → disk)
SOURCE_LSYNCD_CONFIG="/tmp/lsyncd_source_ramdisk.conf"
cat > ${SOURCE_LSYNCD_CONFIG} << EOF
settings {
   logfile = "/tmp/lsyncd_source_ramdisk.log",
   statusFile = "/tmp/lsyncd_source_ramdisk.status",
   statusInterval = 10
}

sync {
   default.rsync,
   source = "${SOURCE_RAMDISK_PATH}",
   target = "${SOURCE_DIR}",
   rsync = {
      archive = true,
      compress = false,
      whole_file = true
   }
}
EOF

# Create lsyncd configuration for database (RAM disk → disk)
DB_LSYNCD_CONFIG="/tmp/lsyncd_db_ramdisk.conf"
cat > ${DB_LSYNCD_CONFIG} << EOF
settings {
   logfile = "/tmp/lsyncd_db_ramdisk.log",
   statusFile = "/tmp/lsyncd_db_ramdisk.status",
   statusInterval = 10
}

sync {
   default.rsync,
   source = "${DB_RAMDISK_PATH}",
   target = "${DB_DIR}",
   rsync = {
      archive = true,
      compress = false,
      whole_file = true
   }
}
EOF

# Start lsyncd for both RAM disks
echo "Starting lsyncd to sync RAM disks with disk storage..." | tee -a ${LOG_FILE}
sudo lsyncd ${SOURCE_LSYNCD_CONFIG}
sudo lsyncd ${DB_LSYNCD_CONFIG}
echo "lsyncd started at: $(date)" | tee -a ${LOG_FILE}

# Create a RAM disk paths configuration file
echo "Creating RAM disk paths configuration file..." | tee -a ${LOG_FILE}
RAM_PATHS_CONFIG="${CONFIG_D_DIR}/90-ramdisk-paths.yaml"

# Create the config file with RAM disk paths
cat > ${RAM_PATHS_CONFIG} << EOF
# RAM disk paths configuration
# Generated by ramdisk_embedder.sh at $(date)

# Override source and output directories to use RAM disks
source_documents_dir: "${SOURCE_RAMDISK_PATH}"
output_dir: "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}"

# Disable collection to prevent downloading documents again
collection:
  enabled: false
EOF

# Create output directory in RAM disk
echo "Creating output directory in RAM disk: ${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}" | tee -a ${LOG_FILE}
mkdir -p ${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}

# Create a processing mode configuration file if using CPU
if [ "$PROCESSING_MODE" = "cpu" ]; then
    echo "Setting up CPU-specific environment variables for PyTorch..." | tee -a ${LOG_FILE}
    # These will be exported later before running the command
    echo "OMP_NUM_THREADS=${NUM_THREADS}" | tee -a ${LOG_FILE}
    echo "MKL_NUM_THREADS=${NUM_THREADS}" | tee -a ${LOG_FILE}
    echo "NUMEXPR_NUM_THREADS=${NUM_THREADS}" | tee -a ${LOG_FILE}
    echo "OPENBLAS_NUM_THREADS=${NUM_THREADS}" | tee -a ${LOG_FILE}
    echo "VECLIB_MAXIMUM_THREADS=${NUM_THREADS}" | tee -a ${LOG_FILE}
fi

# Print stats before starting
echo "RAM disk stats before processing:" | tee -a ${LOG_FILE}
df -h ${SOURCE_RAMDISK_PATH} ${DB_RAMDISK_PATH} | tee -a ${LOG_FILE}
echo "Source document count:" | tee -a ${LOG_FILE}
DOC_COUNT=$(find ${SOURCE_RAMDISK_PATH} -type f | wc -l)
echo "${DOC_COUNT} documents found" | tee -a ${LOG_FILE}

# Start time measurement
START_TIME=$(date +%s)
echo "Starting dataset builder at $(date)" | tee -a ${LOG_FILE}

# Run the dataset builder with the modified config
echo "Running dataset builder with RAM disk and CPU-based multithreading..." | tee -a ${LOG_FILE}
cd /home/todd/ML-Lab/New-HADES/rag-dataset-builder

# Always use the Python from the virtual environment
PYTHON_PATH="/home/todd/ML-Lab/New-HADES/.venv/bin/python"

# Verify that the Python executable exists
if [ ! -x "${PYTHON_PATH}" ]; then
    echo "ERROR: Python executable not found at ${PYTHON_PATH}. Please ensure the virtual environment is properly set up." | tee -a ${LOG_FILE}
    exit 1
fi

echo "Using Python from virtual environment at: ${PYTHON_PATH}" | tee -a ${LOG_FILE}

# Create output directory on disk as well
mkdir -p ${DB_DIR}/${OUTPUT_SUBDIR}

# Set PyTorch environment variables for multithreading
export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export NUMEXPR_NUM_THREADS=${NUM_THREADS}
export OPENBLAS_NUM_THREADS=${NUM_THREADS}
export VECLIB_MAXIMUM_THREADS=${NUM_THREADS}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run the dataset builder with the config.d directory approach
echo "Command: ${PYTHON_PATH} -m src.main --config_dir ${CONFIG_D_DIR} --threads ${NUM_THREADS}" | tee -a ${LOG_FILE}

if [ "$PROCESSING_MODE" = "cpu" ]; then
    echo "Using PyTorch with ${NUM_THREADS} threads on CPU" | tee -a ${LOG_FILE}
else
    echo "Using GPU acceleration with PyTorch" | tee -a ${LOG_FILE}
fi

# Make sure we're in the correct directory
cd /home/todd/ML-Lab/New-HADES/rag-dataset-builder

# Run the command with proper error handling
${PYTHON_PATH} -m src.main --config_dir ${CONFIG_D_DIR} --threads ${NUM_THREADS} 2>&1 | tee -a ${LOG_FILE}

# Check if the command succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Dataset builder failed. Check the log for details." | tee -a ${LOG_FILE}
    # Continue with cleanup even if the command failed
fi

# End time measurement
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Processing completed in ${ELAPSED_TIME} seconds ($(echo "scale=2; ${ELAPSED_TIME}/60" | bc) minutes)" | tee -a ${LOG_FILE}
echo "Finished at $(date)" | tee -a ${LOG_FILE}

# Print stats after processing
echo "RAM disk stats after processing:" | tee -a ${LOG_FILE}
df -h ${SOURCE_RAMDISK_PATH} ${DB_RAMDISK_PATH} | tee -a ${LOG_FILE}

# Stop lsyncd and perform final sync with rsync
echo "Stopping lsyncd and performing final sync..." | tee -a ${LOG_FILE}
sudo pkill lsyncd
echo "lsyncd stopped at $(date)" | tee -a ${LOG_FILE}

# Final sync from RAM disks to disk
echo "Final sync from RAM disks to disk storage..." | tee -a ${LOG_FILE}
echo "Starting final sync at $(date)" | tee -a ${LOG_FILE}
rsync -av --delete ${SOURCE_RAMDISK_PATH}/ ${SOURCE_DIR}/ 2>&1 | tee -a ${LOG_FILE}
rsync -av --delete ${DB_RAMDISK_PATH}/ ${DB_DIR}/ 2>&1 | tee -a ${LOG_FILE}
echo "Final sync completed at $(date)" | tee -a ${LOG_FILE}

# Copy the final output directory to a timestamped location for comparison
FINAL_OUTPUT_DIR="${DB_DIR}/${OUTPUT_SUBDIR}"
echo "Copying final output to ${FINAL_OUTPUT_DIR}..." | tee -a ${LOG_FILE}
mkdir -p ${FINAL_OUTPUT_DIR}

# Check if there are files to copy
if [ "$(ls -A ${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/)" ]; then
    cp -r ${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/* ${FINAL_OUTPUT_DIR}/
    echo "Output files copied successfully" | tee -a ${LOG_FILE}
else
    echo "Warning: No output files found in ${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/" | tee -a ${LOG_FILE}
fi

# Record final statistics
echo "\nFinal Statistics:" | tee -a ${LOG_FILE}
echo "Timestamp: ${TIMESTAMP}" | tee -a ${LOG_FILE}

# Check if output directory exists and has files
if [ -d "${FINAL_OUTPUT_DIR}" ] && [ "$(ls -A ${FINAL_OUTPUT_DIR})" ]; then
    DOCS_PROCESSED=$(find ${FINAL_OUTPUT_DIR} -name "*.json" 2>/dev/null | wc -l)
    echo "Total documents processed: ${DOCS_PROCESSED}" | tee -a ${LOG_FILE}
    
    # Use a safer approach to count chunks
    if [ ${DOCS_PROCESSED} -gt 0 ]; then
        TOTAL_CHUNKS=$(grep -o '"text":' ${FINAL_OUTPUT_DIR}/*.json 2>/dev/null | wc -l)
        echo "Total chunks: ${TOTAL_CHUNKS}" | tee -a ${LOG_FILE}
    else
        TOTAL_CHUNKS=0
        echo "Total chunks: 0 (no JSON files found)" | tee -a ${LOG_FILE}
    fi
else
    DOCS_PROCESSED=0
    TOTAL_CHUNKS=0
    echo "Total documents processed: 0 (no output directory or empty directory)" | tee -a ${LOG_FILE}
    echo "Total chunks: 0" | tee -a ${LOG_FILE}
fi

echo "Completed at: $(date)" | tee -a ${LOG_FILE}

# Compare with previous run
echo "\nComparison with previous run:" | tee -a ${LOG_FILE}
echo "Previous GPU run: 1,852 documents, 2,252,816 chunks, ~5 hours processing time" | tee -a ${LOG_FILE}
echo "This CPU run with RAM disk: ${DOCS_PROCESSED} documents, ${TOTAL_CHUNKS} chunks, $((ELAPSED_TIME/60)) minutes processing time" | tee -a ${LOG_FILE}

# Only calculate speedup if we actually processed documents and the elapsed time is greater than zero
if [ ${DOCS_PROCESSED} -gt 0 ] && [ ${ELAPSED_TIME} -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; 300/${ELAPSED_TIME}*60" | bc)
    echo "Speedup factor: approximately ${SPEEDUP}x faster with CPU+RAM disk vs GPU" | tee -a ${LOG_FILE}
else
    echo "Speedup factor: Could not calculate (no documents processed or processing failed)" | tee -a ${LOG_FILE}
fi
echo "Output directories for comparison:" | tee -a ${LOG_FILE}
echo "- GPU version: ${DB_DIR}/current" | tee -a ${LOG_FILE}
echo "- CPU version: ${DB_DIR}/${OUTPUT_SUBDIR}" | tee -a ${LOG_FILE}

# Unmount RAM disks and clean up
echo "Unmounting RAM disks and cleaning up..." | tee -a ${LOG_FILE}
sudo umount ${SOURCE_RAMDISK_PATH}
sudo umount ${DB_RAMDISK_PATH}
sudo rmdir ${SOURCE_RAMDISK_PATH}
sudo rmdir ${DB_RAMDISK_PATH}
echo "RAM disks unmounted at $(date)" | tee -a ${LOG_FILE}

echo "RAM disk embedding completed in ${ELAPSED_TIME} seconds ($(echo "scale=2; ${ELAPSED_TIME}/60" | bc) minutes)" | tee -a ${LOG_FILE}
echo "All data has been synced back to disk storage and RAM has been freed" | tee -a ${LOG_FILE}
echo "Log file saved to: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "Output saved to: ${FINAL_OUTPUT_DIR}" | tee -a ${LOG_FILE}
