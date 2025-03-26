FROM python:3.10-slim

WORKDIR /app

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and setup files
COPY requirements.txt pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install monitoring libraries
RUN pip install --no-cache-dir \
    prometheus-client \
    psutil \
    arize-phoenix \
    grafana-pandas \
    mlflow
    
# Copy the source code and project files
COPY . .

# Install NetworkX for PathRAG and other requirements
RUN pip install --no-cache-dir \
    networkx \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    tqdm

# Create directories for data and results
RUN mkdir -p data/datasets/kilt data/results

# Download KILT datasets if not already downloaded during environment setup
RUN if [ ! -f "data/datasets/kilt/nq-test.jsonl" ]; then \
    wget -O data/datasets/kilt/nq-test.jsonl https://dl.fbaipublicfiles.com/KILT/nq-test.jsonl; \
    fi

RUN if [ ! -f "data/datasets/kilt/hotpotqa-test.jsonl" ]; then \
    wget -O data/datasets/kilt/hotpotqa-test.jsonl https://dl.fbaipublicfiles.com/KILT/hotpotqa-test.jsonl; \
    fi

# Add monitoring scripts
COPY monitoring/scripts/metrics_exporter.py /app/monitoring/scripts/
COPY monitoring/scripts/start_monitoring.sh /app/monitoring/scripts/
RUN chmod +x /app/monitoring/scripts/start_monitoring.sh

# Create a non-root user and switch to it
RUN groupadd -r hades && useradd -r -g hades hades
RUN chown -R hades:hades /app
USER hades

# Set the entrypoint to use our monitoring wrapper script
ENTRYPOINT ["/app/monitoring/scripts/start_monitoring.sh"]
