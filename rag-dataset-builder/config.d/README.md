# RAG Dataset Builder Configuration System

This directory contains the configuration files for the RAG Dataset Builder framework. The configuration follows a Linux-style approach with numeric prefixes to control the loading order.

## Configuration Loading Order

Files are loaded in numeric order based on the prefix, with lower numbers loaded first. This means that settings in higher-numbered files will override those in lower-numbered files.

1. `00-default.yaml` - Core system settings and defaults
2. `10-processors.yaml` - Document processor configurations
3. `20-chunkers.yaml` - Text chunking strategies
4. `30-embedders.yaml` - Embedding models
5. `40-storage.yaml` - Storage backends
6. `50-retrieval.yaml` - Retrieval systems 
7. `60-monitoring.yaml` - Performance tracking and monitoring

## Custom Configurations

To create custom configurations, you have two options:

1. **Add a new file to this directory** with a higher numeric prefix (e.g., `90-custom.yaml`)
2. **Create a user configuration file** elsewhere and pass it to the system at runtime

## Template Configurations

Implementation-specific template configurations can be found in their respective directories:

- PathRAG: `src/implementations/pathrag/config-example.yaml`
- Neo4j: `src/storage/neo4j/config-template.yaml`

These templates can be copied to the `config.d/` directory with a high numeric prefix to override the defaults.

## Configuration References

The configuration system supports:

1. **Environment variable references** using `${ENV_VAR}` syntax
2. **Cross-references** to other configuration values using `${path.to.value}` syntax

Example:
```yaml
directories:
  logs: "./logs"

logging:
  log_file: "${directories.logs}/app.log"
```

## Required Configurations

At minimum, a valid configuration must include:

- `directories` section with `input` and `output` settings
- `processing` section with basic processing parameters
- `logging` section for log configuration

## More Information

For more details on the configuration system, see the documentation in `src/core/config_loader.py`.
