#!/usr/bin/env python3
"""
Extended PathRAG Dataset Collector

This script extends our dataset collection with additional sources including:
1. Documentation for tools used in the project
2. Linux documentation (Ubuntu, LFS, BLFS)
3. Programming language documentation
4. Technical specifications relevant to the project

It processes these sources into a format suitable for PathRAG while maintaining
low memory usage.
"""

import os
import time
import requests
import argparse
import subprocess
import re
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extended_dataset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("extended_dataset")

# Define sources for documentation
DOCUMENTATION_SOURCES = {
    "ubuntu": {
        "name": "Ubuntu Documentation",
        "url": "https://help.ubuntu.com/",
        "type": "web"
    },
    "lfs": {
        "name": "Linux From Scratch",
        "url": "https://www.linuxfromscratch.org/lfs/view/stable/",
        "type": "web"
    },
    "blfs": {
        "name": "Beyond Linux From Scratch",
        "url": "https://www.linuxfromscratch.org/blfs/view/stable/",
        "type": "web"
    },
    "python": {
        "name": "Python Documentation",
        "url": "https://docs.python.org/3/",
        "type": "web"
    },
    "python_library": {
        "name": "Python Standard Library",
        "url": "https://docs.python.org/3/library/",
        "type": "web"
    },
    "python_tutorial": {
        "name": "Python Tutorial",
        "url": "https://docs.python.org/3/tutorial/",
        "type": "web"
    },
    "mojo": {
        "name": "Mojo Documentation",
        "url": "https://docs.modular.com/mojo/",
        "type": "web"
    },
    "mojo_manual": {
        "name": "Mojo Programming Manual",
        "url": "https://docs.modular.com/mojo/programming-manual.html",
        "type": "web"
    },
    "modular": {
        "name": "Modular Platform Documentation",
        "url": "https://docs.modular.com/",
        "type": "web"
    },
    "openai": {
        "name": "OpenAI API Documentation",
        "url": "https://platform.openai.com/docs/",
        "type": "web"
    },
    "graphml": {
        "name": "Graph ML Documentation",
        "url": "https://pytorch-geometric.readthedocs.io/en/latest/",
        "type": "web"
    },
    "arize": {
        "name": "Arize Documentation",
        "url": "https://docs.arize.com/",
        "type": "web"
    },
    "langchain": {
        "name": "LangChain Documentation",
        "url": "https://python.langchain.com/docs/",
        "type": "web"
    },
    "git": {
        "name": "Git Documentation",
        "url": "https://git-scm.com/doc",
        "type": "web"
    }
}

# Define coding datasets
CODING_DATASETS = [
    "codeparrot/github-code",
    "codeparrot/codeparrot-clean",
    "bigcode/the-stack",
    "bigcode/the-stack-smol",
    "huggingface/code_search_net"
]

class DocumentationCollector:
    """
    Collects documentation from various sources and formats it for PathRAG.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the DocumentationCollector.
        
        Args:
            output_dir: Directory to save collected documentation
        """
        self.output_dir = output_dir
        self.docs_dir = os.path.join(output_dir, "documentation")
        os.makedirs(self.docs_dir, exist_ok=True)
    
    def collect_web_documentation(self, name: str, url: str) -> List[str]:
        """
        Collect documentation from a website using wget.
        
        Args:
            name: Name of the documentation
            url: URL of the documentation
            
        Returns:
            List of paths to downloaded HTML files
        """
        # Create directory for this documentation
        safe_name = name.replace(' ', '_').lower()
        doc_dir = os.path.join(self.docs_dir, safe_name)
        os.makedirs(doc_dir, exist_ok=True)
        
        logger.info(f"Collecting documentation: {name} from {url}")
        
        # Use wget to download the documentation
        # --recursive: recursively download
        # --level=3: go 3 levels deep
        # --no-clobber: don't overwrite existing files
        # --page-requisites: get all assets to display the page
        # --html-extension: save as .html
        # --convert-links: convert links to work locally
        # --restrict-file-names=windows: escape special characters
        # --domains: stay on these domains
        # --no-parent: don't go to parent directory
        command = [
            "wget",
            "--recursive",
            "--level=3",
            "--no-clobber",
            "--page-requisites",
            "--html-extension",
            "--convert-links",
            "--restrict-file-names=windows",
            "--domains=" + url.split('/')[2],
            "--no-parent",
            "--timeout=10",
            "--tries=3",
            "-P", doc_dir,
            url
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True)
            logger.info(f"Successfully downloaded documentation: {name}")
            
            # Find all HTML files
            html_files = []
            for root, dirs, files in os.walk(doc_dir):
                for file in files:
                    if file.endswith('.html') or file.endswith('.htm'):
                        html_files.append(os.path.join(root, file))
            
            logger.info(f"Found {len(html_files)} HTML files for {name}")
            return html_files
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading documentation: {e}")
            return []
    
    def convert_html_to_text(self, html_path: str) -> Optional[str]:
        """
        Convert an HTML file to plain text.
        
        Args:
            html_path: Path to the HTML file
            
        Returns:
            Path to the text file or None if conversion failed
        """
        # Extract filename without extension
        filename = os.path.basename(html_path).rsplit('.', 1)[0]
        parent_dir = os.path.dirname(html_path)
        txt_dir = os.path.join(parent_dir, 'text')
        os.makedirs(txt_dir, exist_ok=True)
        output_path = os.path.join(txt_dir, f"{filename}.txt")
        
        # Check if file already exists
        if os.path.exists(output_path):
            logger.info(f"Text file already exists: {output_path}")
            return output_path
        
        logger.info(f"Converting HTML to text: {html_path}")
        try:
            # Use html2text (install with pip if needed)
            try:
                import html2text
                with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.body_width = 0  # No wrapping
                text = h.handle(html_content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                logger.info(f"Converted HTML to text: {output_path}")
                return output_path
            except ImportError:
                # Fallback to lynx if html2text is not installed
                result = subprocess.run(
                    ['lynx', '-dump', '-nolist', html_path],
                    capture_output=True, text=True, check=True
                )
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                
                logger.info(f"Converted HTML to text: {output_path}")
                return output_path
        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            return None
    
    def collect_all_documentation(self) -> Dict[str, List[str]]:
        """
        Collect documentation from all sources.
        
        Returns:
            Dictionary mapping source names to lists of text file paths
        """
        results = {}
        
        for source_id, source_info in DOCUMENTATION_SOURCES.items():
            logger.info(f"Processing documentation source: {source_info['name']}")
            
            if source_info['type'] == 'web':
                html_files = self.collect_web_documentation(
                    source_info['name'], source_info['url']
                )
                
                # Convert HTML to text
                text_files = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for text_path in executor.map(self.convert_html_to_text, html_files):
                        if text_path:
                            text_files.append(text_path)
                
                results[source_id] = text_files
                logger.info(f"Processed {len(text_files)} files for {source_info['name']}")
        
        return results


class CodingDatasetCollector:
    """
    Collects code samples from Hugging Face datasets for PathRAG testing.
    """
    
    def __init__(self, output_dir: str, max_samples_per_dataset: int = 1000):
        """
        Initialize the CodingDatasetCollector.
        
        Args:
            output_dir: Directory to save collected code samples
            max_samples_per_dataset: Maximum number of samples to collect per dataset
        """
        self.output_dir = output_dir
        self.code_dir = os.path.join(output_dir, "code_samples")
        self.max_samples = max_samples_per_dataset
        os.makedirs(self.code_dir, exist_ok=True)
    
    def collect_code_dataset(self, dataset_name: str) -> List[str]:
        """
        Collect code samples from a Hugging Face dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of paths to code files
        """
        # Import here to avoid loading these libraries unless needed
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            return []
        
        logger.info(f"Collecting code samples from dataset: {dataset_name}")
        
        # Create directory for this dataset
        safe_name = dataset_name.replace('/', '_')
        dataset_dir = os.path.join(self.code_dir, safe_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        file_paths = []
        
        try:
            # Load dataset with streaming to avoid memory issues
            dataset = load_dataset(dataset_name, streaming=True)
            
            count = 0
            for split in dataset.keys():
                split_dir = os.path.join(dataset_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                
                for i, example in enumerate(dataset[split].take(self.max_samples)):
                    # Extract code from the example - field names vary by dataset
                    code = self._extract_code_from_example(example)
                    if code:
                        file_extension = self._guess_file_extension(example)
                        file_path = os.path.join(split_dir, f"sample_{i}{file_extension}")
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(code)
                        
                        file_paths.append(file_path)
                        count += 1
                    
                    if count >= self.max_samples:
                        break
            
            logger.info(f"Collected {count} code samples from {dataset_name}")
            return file_paths
        except Exception as e:
            logger.error(f"Error collecting code samples from {dataset_name}: {e}")
            return []
    
    def _extract_code_from_example(self, example: Dict[str, Any]) -> Optional[str]:
        """
        Extract code from a dataset example.
        
        Args:
            example: Dataset example
            
        Returns:
            Extracted code or None if no code could be extracted
        """
        # Check for common field names in code datasets
        for field in ['code', 'content', 'func', 'function', 'source_code', 'source']:
            if field in example and isinstance(example[field], str):
                return example[field]
        
        # For the-stack dataset
        if 'content' in example and isinstance(example['content'], bytes):
            return example['content'].decode('utf-8', errors='ignore')
        
        return None
    
    def _guess_file_extension(self, example: Dict[str, Any]) -> str:
        """
        Guess the file extension based on the example.
        
        Args:
            example: Dataset example
            
        Returns:
            File extension (e.g., '.py', '.js')
        """
        # Check if language is specified
        if 'language' in example:
            lang = example['language'].lower()
            if lang == 'python':
                return '.py'
            elif lang == 'javascript':
                return '.js'
            elif lang == 'java':
                return '.java'
            elif lang == 'c#':
                return '.cs'
            elif lang == 'c++':
                return '.cpp'
            elif lang == 'c':
                return '.c'
            elif lang == 'go':
                return '.go'
            elif lang == 'ruby':
                return '.rb'
            elif lang == 'rust':
                return '.rs'
            elif lang == 'typescript':
                return '.ts'
            elif lang == 'php':
                return '.php'
        
        # Default to .txt
        return '.txt'
    
    def collect_all_datasets(self) -> Dict[str, List[str]]:
        """
        Collect code samples from all datasets.
        
        Returns:
            Dictionary mapping dataset names to lists of code file paths
        """
        results = {}
        
        for dataset_name in CODING_DATASETS:
            logger.info(f"Processing dataset: {dataset_name}")
            file_paths = self.collect_code_dataset(dataset_name)
            results[dataset_name] = file_paths
            logger.info(f"Processed {len(file_paths)} files for {dataset_name}")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect extended datasets for PathRAG testing")
    parser.add_argument("--output-dir", type=str, default="./pathrag_data/extended",
                        help="Directory to save collected data")
    parser.add_argument("--docs-only", action="store_true",
                        help="Only collect documentation")
    parser.add_argument("--code-only", action="store_true",
                        help="Only collect code samples")
    parser.add_argument("--max-code-samples", type=int, default=1000,
                        help="Maximum number of code samples to collect per dataset")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect documentation if requested
    if not args.code_only:
        logger.info("Collecting documentation")
        doc_collector = DocumentationCollector(args.output_dir)
        doc_results = doc_collector.collect_all_documentation()
        
        # Save results
        with open(os.path.join(args.output_dir, "documentation_results.json"), 'w') as f:
            json.dump({k: len(v) for k, v in doc_results.items()}, f, indent=2)
    
    # Collect code samples if requested
    if not args.docs_only:
        logger.info("Collecting code samples")
        code_collector = CodingDatasetCollector(
            args.output_dir, max_samples_per_dataset=args.max_code_samples
        )
        code_results = code_collector.collect_all_datasets()
        
        # Save results
        with open(os.path.join(args.output_dir, "code_results.json"), 'w') as f:
            json.dump({k: len(v) for k, v in code_results.items()}, f, indent=2)
    
    logger.info("Extended dataset collection complete")


if __name__ == "__main__":
    main()
