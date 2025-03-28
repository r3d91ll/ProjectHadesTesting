"""
Academic Paper Collectors

This module provides collectors for academic papers from various sources.
"""

from .base_academic_collector import BaseAcademicCollector
from .semantic_scholar_collector import SemanticScholarCollector
from .pubmed_collector import PubMedCollector
from .socarxiv_collector import SocArXivCollector

__all__ = [
    'BaseAcademicCollector',
    'SemanticScholarCollector',
    'PubMedCollector',
    'SocArXivCollector'
]
