"""
Document Deduplication Module for RAG Dataset Builder

This module provides functionality for detecting and handling duplicate documents
across multiple sources and collection runs.
"""

from .base import BaseDeduplicator
from .academic_dedup import AcademicPaperDeduplicator
from .registry import DocumentRegistry

__all__ = ['BaseDeduplicator', 'AcademicPaperDeduplicator', 'DocumentRegistry']
