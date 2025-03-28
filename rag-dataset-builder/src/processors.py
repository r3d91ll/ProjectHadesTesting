#!/usr/bin/env python3
"""
RAG Dataset Builder - Document Processors

This module contains implementations of document processors for the RAG Dataset Builder.
"""

import os
import logging
import json
import hashlib
import re
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from builder import BaseProcessor

# Configure logging
logger = logging.getLogger("rag_dataset_builder.processors")


class SimpleTextProcessor(BaseProcessor):
    """Simple processor for plain text documents."""
    
    def process_document(self, doc_path: str) -> Dict[str, Any]:
        """
        Process a plain text document.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Processed document data
        """
        try:
            # Read text file
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Extract basic metadata
            filename = os.path.basename(doc_path)
            file_size = os.path.getsize(doc_path)
            
            # Determine category based on path
            category = self._determine_category(doc_path)
            
            # Try to extract title from text
            title = filename
            title_match = re.search(r'^(?:#|Title:)\s*(.+?)$', text, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "text": text,
                "metadata": {
                    "filename": filename,
                    "title": title,
                    "path": doc_path,
                    "file_size": file_size,
                    "category": category,
                    "extension": os.path.splitext(filename)[1],
                    "created_at": datetime.fromtimestamp(os.path.getctime(doc_path)).isoformat(),
                    "modified_at": datetime.fromtimestamp(os.path.getmtime(doc_path)).isoformat(),
                    "character_count": len(text)
                }
            }
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "text": "",
                "metadata": {
                    "filename": os.path.basename(doc_path),
                    "path": doc_path,
                    "error": str(e)
                }
            }
    
    def _determine_category(self, doc_path: str) -> str:
        """
        Determine document category based on path.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Document category
        """
        path_parts = doc_path.lower().split(os.sep)
        
        # Check for specific research areas in the path
        for category in [
            "actor_network_theory", 
            "sts_digital_sociology",
            "value_studies",
            "anthropology_value",
            "science_technology_studies",
            "knowledge_graphs_retrieval",
            "computational_linguistics",
            "ethics_bias_ai",
            "graph_reasoning_ml",
            "semiotics_linguistic_anthropology"
        ]:
            if any(category.replace("_", "") in part.replace("_", "") for part in path_parts):
                return category
        
        # Check for document type categories
        for part in path_parts:
            if "papers" in part:
                return "research_paper"
            elif "documentation" in part:
                return "documentation"
            elif "code" in part or "samples" in part:
                return "code"
        
        return "unknown"


class PDFProcessor(BaseProcessor):
    """Processor for PDF documents."""
    
    def process_document(self, doc_path: str) -> Dict[str, Any]:
        """
        Process a PDF document.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Processed document data
        """
        if not doc_path.lower().endswith('.pdf'):
            logger.warning(f"Not a PDF file: {doc_path}")
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "text": "",
                "metadata": {
                    "filename": os.path.basename(doc_path),
                    "path": doc_path,
                    "error": "Not a PDF file"
                }
            }
        
        try:
            # Extract text using pdftotext (if available)
            try:
                text = self._extract_text_with_pdftotext(doc_path)
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning(f"pdftotext failed, falling back to alternative method for {doc_path}")
                text = self._extract_text_with_fallback(doc_path)
            
            if not text:
                logger.warning(f"Failed to extract text from PDF: {doc_path}")
                return {
                    "id": hashlib.md5(doc_path.encode()).hexdigest(),
                    "text": "",
                    "metadata": {
                        "filename": os.path.basename(doc_path),
                        "path": doc_path,
                        "error": "Failed to extract text"
                    }
                }
            
            # Extract metadata
            filename = os.path.basename(doc_path)
            file_size = os.path.getsize(doc_path)
            category = self._determine_category(doc_path)
            
            # Try to extract title from text
            title = filename
            title_match = re.search(r'^(?:#|Title:)\s*(.+?)$', text, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                # Try extracting from the first few lines
                lines = text.split('\n')[:10]
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 10 and len(line) < 100:
                        title = line
                        break
            
            # Try to extract authors
            authors = []
            author_match = re.search(r'(?:Authors?|By):\s*(.+?)$', text, re.MULTILINE)
            if author_match:
                authors_text = author_match.group(1).strip()
                authors = [a.strip() for a in authors_text.split(',')]
            
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "text": text,
                "metadata": {
                    "filename": filename,
                    "title": title,
                    "authors": authors,
                    "path": doc_path,
                    "file_size": file_size,
                    "category": category,
                    "extension": ".pdf",
                    "created_at": datetime.fromtimestamp(os.path.getctime(doc_path)).isoformat(),
                    "modified_at": datetime.fromtimestamp(os.path.getmtime(doc_path)).isoformat(),
                    "character_count": len(text)
                }
            }
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "text": "",
                "metadata": {
                    "filename": os.path.basename(doc_path),
                    "path": doc_path,
                    "error": str(e)
                }
            }
    
    def _extract_text_with_pdftotext(self, doc_path: str) -> str:
        """
        Extract text from PDF using pdftotext.
        
        Args:
            doc_path: Path to the PDF
            
        Returns:
            Extracted text
        """
        result = subprocess.run(
            ['pdftotext', '-layout', doc_path, '-'],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    
    def _extract_text_with_fallback(self, doc_path: str) -> str:
        """
        Extract text from PDF using PyPDF2 as fallback.
        
        Args:
            doc_path: Path to the PDF
            
        Returns:
            Extracted text
        """
        try:
            import PyPDF2
            
            text = ""
            with open(doc_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            
            return text
        except ImportError:
            logger.error("PyPDF2 not installed. Unable to extract text from PDF.")
            return ""
    
    def _determine_category(self, doc_path: str) -> str:
        """
        Determine document category based on path.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Document category
        """
        # Reuse the same category determination logic
        processor = SimpleTextProcessor()
        return processor._determine_category(doc_path)


class CodeProcessor(BaseProcessor):
    """Processor for code files."""
    
    def __init__(self):
        """Initialize the code processor."""
        # Map file extensions to languages
        self.extension_to_language = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "c++",
            ".h": "c/c++ header",
            ".hpp": "c++ header",
            ".cs": "c#",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "shell",
            ".bash": "bash",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".r": "r",
            ".jl": "julia"
        }
    
    def process_document(self, doc_path: str) -> Dict[str, Any]:
        """
        Process a code file.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Processed document data
        """
        try:
            # Read code file
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Extract basic metadata
            filename = os.path.basename(doc_path)
            file_extension = os.path.splitext(filename)[1].lower()
            file_size = os.path.getsize(doc_path)
            
            # Determine language based on extension
            language = self.extension_to_language.get(file_extension, "unknown")
            
            # Get category from path
            category = self._determine_category(doc_path)
            
            # Try to extract module/class/function names
            module_name = filename
            class_names = self._extract_class_names(text, language)
            function_names = self._extract_function_names(text, language)
            
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "text": text,
                "metadata": {
                    "filename": filename,
                    "path": doc_path,
                    "file_size": file_size,
                    "category": category,
                    "extension": file_extension,
                    "language": language,
                    "module_name": module_name,
                    "class_names": class_names,
                    "function_names": function_names,
                    "created_at": datetime.fromtimestamp(os.path.getctime(doc_path)).isoformat(),
                    "modified_at": datetime.fromtimestamp(os.path.getmtime(doc_path)).isoformat(),
                    "line_count": len(text.splitlines()),
                    "character_count": len(text)
                }
            }
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return {
                "id": hashlib.md5(doc_path.encode()).hexdigest(),
                "text": "",
                "metadata": {
                    "filename": os.path.basename(doc_path),
                    "path": doc_path,
                    "error": str(e)
                }
            }
    
    def _extract_class_names(self, text: str, language: str) -> List[str]:
        """
        Extract class names from code.
        
        Args:
            text: Code text
            language: Programming language
            
        Returns:
            List of class names
        """
        class_names = []
        
        if language == "python":
            # Match Python class definitions
            matches = re.finditer(r'class\s+([A-Za-z0-9_]+)\s*(?:\(.*\))?:', text)
            class_names = [match.group(1) for match in matches]
        elif language in ["java", "c#", "kotlin"]:
            # Match Java/C#/Kotlin class definitions
            matches = re.finditer(r'(?:public|private|protected)?\s*class\s+([A-Za-z0-9_]+)', text)
            class_names = [match.group(1) for match in matches]
        elif language in ["javascript", "typescript"]:
            # Match JavaScript/TypeScript class definitions
            matches = re.finditer(r'class\s+([A-Za-z0-9_]+)', text)
            class_names = [match.group(1) for match in matches]
        
        return class_names
    
    def _extract_function_names(self, text: str, language: str) -> List[str]:
        """
        Extract function names from code.
        
        Args:
            text: Code text
            language: Programming language
            
        Returns:
            List of function names
        """
        function_names = []
        
        if language == "python":
            # Match Python function definitions
            matches = re.finditer(r'def\s+([A-Za-z0-9_]+)\s*\(', text)
            function_names = [match.group(1) for match in matches]
        elif language in ["javascript", "typescript"]:
            # Match JavaScript/TypeScript function definitions
            matches = re.finditer(r'function\s+([A-Za-z0-9_]+)\s*\(', text)
            function_names = [match.group(1) for match in matches]
            
            # Also match arrow functions and methods
            methods = re.finditer(r'([A-Za-z0-9_]+)\s*[=:]\s*function|\([^)]*\)\s*=>', text)
            function_names.extend([match.group(1) for match in methods if match.group(1)])
        elif language in ["java", "c#", "kotlin"]:
            # Match Java/C#/Kotlin method definitions
            matches = re.finditer(r'(?:public|private|protected)?\s*(?:static)?\s*(?:[A-Za-z0-9_<>]+)\s+([A-Za-z0-9_]+)\s*\(', text)
            function_names = [match.group(1) for match in matches]
        
        return function_names
    
    def _determine_category(self, doc_path: str) -> str:
        """
        Determine document category based on path.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Document category
        """
        path_parts = doc_path.lower().split(os.sep)
        
        # Check if it's part of a specific project or library
        if "examples" in path_parts:
            return "code_example"
        elif "tests" in path_parts:
            return "test_code"
        elif "docs" in path_parts or "documentation" in path_parts:
            return "documentation_code"
        
        # Default to code samples
        return "code_sample"
