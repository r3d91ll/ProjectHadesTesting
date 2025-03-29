#!/bin/bash
# RAM Disk Embedder for PathRAG
# This script creates RAM disks for both source documents and output database,
# syncs them with disk storage using lsyncd, and runs the dataset builder with
# multithreaded CPU-based embedding for maximum performance

set -e

# Configuration
SOURCE_RAMDISK_SIZE=20G
DB_RAMDISK_SIZE=10G
SOURCE_RAMDISK_PATH="/tmp/ramdisk_source_documents"
DB_RAMDISK_PATH="/tmp/ramdisk_rag_databases"
SOURCE_DIR="/home/todd/ML-Lab/New-HADES/source_documents"
DB_DIR="/home/todd/ML-Lab/New-HADES/rag_databases"
CONFIG_PATH="/home/todd/ML-Lab/New-HADES/rag-dataset-builder/config/config.yaml"
NUM_THREADS=24  # Threadripper has 24 cores/48 threads

# Create source documents RAM disk
echo "Creating ${SOURCE_RAMDISK_SIZE} RAM disk for source documents at ${SOURCE_RAMDISK_PATH}..."
sudo mkdir -p ${SOURCE_RAMDISK_PATH}
sudo mount -t tmpfs -o size=${SOURCE_RAMDISK_SIZE} tmpfs ${SOURCE_RAMDISK_PATH}
sudo chmod 777 ${SOURCE_RAMDISK_PATH}

# Create database RAM disk
echo "Creating ${DB_RAMDISK_SIZE} RAM disk for database at ${DB_RAMDISK_PATH}..."
sudo mkdir -p ${DB_RAMDISK_PATH}
sudo mount -t tmpfs -o size=${DB_RAMDISK_SIZE} tmpfs ${DB_RAMDISK_PATH}
sudo chmod 777 ${DB_RAMDISK_PATH}

# First, copy source documents to RAM disk (faster initial copy than lsyncd)
echo "Copying source documents to RAM disk..."
rsync -av --progress ${SOURCE_DIR}/ ${SOURCE_RAMDISK_PATH}/

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
echo "Starting lsyncd to sync RAM disks with disk storage..."
sudo lsyncd ${SOURCE_LSYNCD_CONFIG}
sudo lsyncd ${DB_LSYNCD_CONFIG}

# Create a modified config file that points to the RAM disks
MODIFIED_CONFIG="/tmp/ramdisk_config.yaml"
cp ${CONFIG_PATH} ${MODIFIED_CONFIG}
sed -i "s|source_documents_dir: .*|source_documents_dir: \"${SOURCE_RAMDISK_PATH}\"|g" ${MODIFIED_CONFIG}
sed -i "s|output_dir: .*|output_dir: \"${DB_RAMDISK_PATH}/current\"|g" ${MODIFIED_CONFIG}

# Modify the config to use CPU-based embedding with more threads
sed -i "s|use_gpu: .*|use_gpu: false|g" ${MODIFIED_CONFIG}
sed -i "s|batch_size: .*|batch_size: 64|g" ${MODIFIED_CONFIG}

# Print stats before starting
echo "RAM disk stats before processing:"
df -h ${SOURCE_RAMDISK_PATH} ${DB_RAMDISK_PATH}
echo "Source document count:"
find ${SOURCE_RAMDISK_PATH} -type f | wc -l

# Start time measurement
START_TIME=$(date +%s)

# Run the dataset builder with the modified config
echo "Running dataset builder with RAM disk and CPU-based multithreading..."
cd /home/todd/ML-Lab/New-HADES/rag-dataset-builder
python -m src.main --config ${MODIFIED_CONFIG} --threads ${NUM_THREADS}

# End time measurement
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Processing completed in ${ELAPSED_TIME} seconds"

# Print stats after processing
echo "RAM disk stats after processing:"
df -h ${SOURCE_RAMDISK_PATH} ${DB_RAMDISK_PATH}

# Stop lsyncd and perform final sync with rsync
echo "Stopping lsyncd and performing final sync..."
sudo pkill lsyncd

# Final sync from RAM disks to disk
echo "Final sync from RAM disks to disk storage..."
rsync -av --delete ${SOURCE_RAMDISK_PATH}/ ${SOURCE_DIR}/
rsync -av --delete ${DB_RAMDISK_PATH}/ ${DB_DIR}/

# Unmount RAM disks and clean up
echo "Unmounting RAM disks and cleaning up..."
sudo umount ${SOURCE_RAMDISK_PATH}
sudo umount ${DB_RAMDISK_PATH}
sudo rmdir ${SOURCE_RAMDISK_PATH}
sudo rmdir ${DB_RAMDISK_PATH}

echo "RAM disk embedding completed in ${ELAPSED_TIME} seconds"
echo "All data has been synced back to disk storage and RAM has been freed"
