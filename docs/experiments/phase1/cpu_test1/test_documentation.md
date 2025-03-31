# CPU Test 1: Document Collection Configuration Test

## Test Date
March 30, 2025

## Test Objective
Verify that the RAG Dataset Builder correctly downloads and processes academic papers according to the specified configuration settings, with proper organization into domain-specific directories and without nested output directories.

## Configuration Settings
- **Test Name**: `pathrag_cpu_test1`
- **Output Directory**: `../rag_databases/pathrag_cpu_test1`
- **Custom Output Directory**: `true` (prevents appending RAG implementation name)
- **Collection Settings**:
  - `max_papers_per_term`: 10
  - `max_documents_per_category`: 100

## Domains Enabled
- `rag_implementation`
- `llm_optimization`
- `vector_databases`
- `anthropology_of_value`
- `science_technology_studies`

## Changes Made

### 1. Fixed Output Directory Nesting
- **Issue**: The output directory was being set in multiple places, causing nested directories like `pathrag/pathrag/`.
- **Changes**:
  - Modified `main.py` to respect the `CUSTOM_OUTPUT_DIR` environment variable
  - Updated `run_unified.sh` to export the `CUSTOM_OUTPUT_DIR` environment variable
  - Added logging to show the actual output directory being used

```python
# In main.py - Added custom output directory handling
if "output_dir" in config:
    # Check if we should use a custom output directory without appending RAG implementation
    custom_output_dir = os.environ.get("CUSTOM_OUTPUT_DIR", "false").lower() == "true"
    if "custom_output_dir" in config:
        custom_output_dir = config["custom_output_dir"]
        
    if custom_output_dir:
        # Use the output directory as is without appending RAG implementation
        base_dir = config["output_dir"]
        logger.info(f"Using custom output directory: {base_dir}")
```

### 2. Fixed Document Collection Configuration
- **Issue**: The document collection process was not respecting the `enabled: false` setting in the configuration.
- **Changes**:
  - Updated `main.py` to include the `enabled` flag in the temporary configuration file for the AcademicCollector
  - Added a check in the `collect_arxiv_papers` method to respect the `collection.enabled` flag
  - Added logging to indicate whether document collection is enabled or disabled

### 3. Fixed Output Directory Comment Issue
- **Issue**: The output directory was being created with an unintended comment in the name (`pathrag_cpu_test1  # Base directory for processed data`).
- **Changes**:
  - Updated `config.yaml` to move the comment to a separate line, preventing it from being included in the directory name

### 4. Improved RAM Disk Cleanup Process
- **Issue**: The RAM disk cleanup process was encountering rsync warnings and "target is busy" errors during unmounting.
- **Changes**:
  - Enhanced the cleanup function in `run_unified.sh` to properly terminate all processes before cleanup
  - Added retry mechanism for rsync operations with appropriate delays
  - Implemented a more robust unmounting sequence with multiple attempts
  - Added better synchronization and error handling throughout the process

### 5. Improved Command Line Options
- **Issue**: The `--clean` flag was cleaning source documents instead of just the output database.
- **Changes**:
  - Renamed `--clean` to `--clean_db` for clarity (with backward compatibility)
  - Modified the behavior to only clean the output database directory
  - Updated documentation and log messages to reflect the change

### 6. Clarified Directory Configuration
- **Issue**: Confusion between directory settings in `config.yaml` and `05-directories.yaml`.
- **Changes**:
  - Updated `05-directories.yaml` with clear comments explaining the RAM disk vs. persistent storage distinction
  - Documented that RAM disk paths override persistent paths when RAM disk is enabled
```

```bash
# In run_unified.sh - Export custom output directory setting
CUSTOM_OUTPUT_DIR=true

# Export the CUSTOM_OUTPUT_DIR environment variable for the Python process
export CUSTOM_OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
```

### 2. Improved Search Term Handling
- **Issue**: The academic collector was using hardcoded default search terms instead of the ones from the configuration file.
- **Changes**:
  - Enhanced the academic collector to properly handle search terms from the configuration
  - Improved error messages when no search terms are found
  - Added detailed logging of search terms for each domain
  - Fixed the temporary configuration file creation to include all necessary settings

```python
# In academic_collector.py - Improved search term handling
if not self.categories:
    # Only use hardcoded terms if explicitly enabled
    use_default_terms = False
    if use_default_terms:
        logger.info("Using default search terms for testing")
    else:
        logger.warning("No search terms found and default terms are disabled. Please check your config file.")
```

```python
# In main.py - Enhanced domain logging and configuration
# Log collection settings for debugging
logger.info(f"Collection settings: max_papers_per_term={collection_config.get('max_papers_per_term', 0)}, "
           f"max_documents_per_category={collection_config.get('max_documents_per_category', 0)}")

# Add search terms from domains with proper structure
for domain_name, domain_config in domains_config.items():
    if domain_config.get("enabled", True):
        search_terms = domain_config.get("search_terms", [])
        if search_terms:  # Only add domains that have search terms
            arxiv_config["domains"][domain_name] = {
                "enabled": True,
                "search_terms": search_terms
            }
            enabled_domains.append(domain_name)
            # Log the search terms for debugging
            logger.info(f"Domain '{domain_name}' has {len(search_terms)} search terms: {search_terms[:3]}...")
```

## Expected Results
1. The system should use the correct output directory without nesting additional `pathrag` directories
2. The academic collector should properly find and use search terms from the domains configuration
3. Documents should be downloaded according to the specified limits:
   - Up to 10 papers per search term
   - Up to 100 documents per category
4. Documents should be organized into the correct domain-specific directories
5. The system should log detailed information about the search terms being used

## Test Execution
To run this test:
```bash
cd /home/todd/ML-Lab/New-HADES/rag-dataset-builder
sudo ./scripts/run_unified.sh --pathrag --clean
```

## Test Status
Completed successfully on March 30, 2025

## Test Results

### Test Run Summary
- **Date**: March 30, 2025
- **Duration**: Approximately 25-30 minutes
- **Documents Processed**: Several hundred academic papers
- **Total Database Size**: 2.07 GB (2,070,163,945 bytes)
- **CPU Utilization**: 70-95% across 24 threads
- **Peak I/O Performance**: ~35 MB/s during final sync phase

### Issues Fixed
1. **Plugin System Loading**: Fixed the issue with loading collector modules by improving the import mechanism in the plugin system.
2. **Domain Configuration Loading**: Enhanced the domain loading logic to properly extract search terms from the configuration files.
3. **Output Directory Structure**: Fixed the output directory structure to prevent nested directories.
4. **Collection Configuration Hierarchy**: Discovered and documented the three-level configuration hierarchy for document collection:
   - Master switch in 18-collection.yaml (`collection.enabled`)
   - Source-specific settings (`sources.arxiv.enabled`, etc.)
   - Domain-specific settings in 19-domains.yaml
5. **--clean_db Flag Issue**: Fixed an issue where the `--clean_db` flag was inadvertently deleting source documents during RAM disk cleanup by removing the `--delete` flag from the rsync command.

### Observations
- The academic collector successfully downloaded papers for all enabled domains.
- The documents were properly organized into domain-specific directories.
- The system used the correct output directory structure without unnecessary nesting.
- The enhanced logging provided clear visibility into the collection process.
- The RAM disk approach significantly improved I/O performance compared to direct disk access.
- The system made efficient use of the 24 CPU threads, maintaining high utilization throughout the process.
- The bidirectional sync between RAM disk and persistent storage worked correctly, preserving all downloaded papers.
- Multiple lsyncd processes accumulated over time, suggesting a need for better process cleanup between runs.

### Performance Metrics

#### CPU Performance
- Sustained high CPU utilization (70-90%) throughout the processing phase
- Efficient parallelization across 24 threads
- System processes used approximately 10-20% of CPU resources

#### I/O Performance
- Steady disk operations during processing (30-50 KB/s)
- Significant spike (200-300 KB/s) during final sync phase
- Peak I/O reached approximately 35 MB/s during final sync

#### Memory Usage
- RAM disk usage remained within allocated limits (20G for source documents, 30G for output database)
- No memory-related issues observed during processing

## Next Steps

### GPU Test Preparation
1. Configure the system to use GPU acceleration for embedding generation
2. Compare performance metrics between CPU and GPU processing
3. Validate that the same document collection and processing logic works correctly with GPU acceleration
4. Document any GPU-specific optimizations or issues encountered

Once both CPU and GPU tests are successfully completed, we can confirm that our embedding and dataset-creation pipeline is working correctly across different hardware configurations.
