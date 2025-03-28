#!/usr/bin/env python3
"""
Setup script for RAG Dataset Builder
"""

from setuptools import setup, find_packages

setup(
    name="rag-dataset-builder",
    version="0.1.0",
    description="A flexible, memory-efficient tool for building datasets for Retrieval-Augmented Generation (RAG) systems",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "networkx>=2.7.0",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
        "nltk>=3.6.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.10.0",
        "requests>=2.26.0",
        "beautifulsoup4>=4.10.0",
        "arxiv>=1.4.2",
    ],
    extras_require={
        "pdf": ["PyPDF2>=2.0.0"],
        "openai": ["openai>=1.0.0"],
        "vectordb": ["faiss-cpu>=1.7.0", "chromadb>=0.3.0"],
        "huggingface": ["datasets>=2.0.0"],
        "arize": ["arize-phoenix>=1.0.0"],
        "all": [
            "PyPDF2>=2.0.0",
            "openai>=1.0.0",
            "faiss-cpu>=1.7.0",
            "chromadb>=0.3.0",
            "datasets>=2.0.0",
            "arize-phoenix>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-builder=rag_dataset_builder.src.main:main",
            "rag-collector=rag_dataset_builder.src.collectors.academic_collector:main",
        ],
    },
    python_requires=">=3.8",
)
