FROM python:3.10-slim

WORKDIR /app

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and setup files
COPY requirements.txt pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy the source code and project files
COPY . .

# Install NetworkX for PathRAG
RUN pip install --no-cache-dir networkx

# Create directories for data and results
RUN mkdir -p data/datasets/kilt data/results

# Download KILT datasets if not already downloaded during environment setup
RUN if [ ! -f "data/datasets/kilt/nq-test.jsonl" ]; then \
    wget -O data/datasets/kilt/nq-test.jsonl https://dl.fbaipublicfiles.com/KILT/nq-test.jsonl; \
    fi

RUN if [ ! -f "data/datasets/kilt/hotpotqa-test.jsonl" ]; then \
    wget -O data/datasets/kilt/hotpotqa-test.jsonl https://dl.fbaipublicfiles.com/KILT/hotpotqa-test.jsonl; \
    fi

# Create a non-root user and switch to it
RUN groupadd -r hades && useradd -r -g hades hades
RUN chown -R hades:hades /app
USER hades

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]
