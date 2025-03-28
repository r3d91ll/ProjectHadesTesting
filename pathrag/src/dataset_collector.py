#!/usr/bin/env python3
"""
PathRAG Dataset Collector

A tool for gathering research papers from arXiv and other sources to create a 
test corpus for PathRAG that aligns with the XnX framework research interests.

This script handles downloading, converting, and organizing papers while maintaining
low memory usage by processing files incrementally.
"""

import os
import time
import requests
import argparse
import xml.etree.ElementTree as ET
import urllib.parse
import concurrent.futures
import shutil
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataset_collector")

# Define categories of interest based on the reading list
CATEGORIES = {
    "actor_network_theory": [
        "Actor-Network Theory", 
        "ANT", 
        "Bruno Latour", 
        "John Law", 
        "Michel Callon"
    ],
    "sts_digital_sociology": [
        "Digital Sociology", 
        "Science and Technology Studies", 
        "STS", 
        "Laboratory Life"
    ],
    "knowledge_graphs_retrieval": [
        "Knowledge Graph", 
        "PathRAG", 
        "GraphRAG", 
        "Retrieval-Augmented Generation",
        "RAG",
        "Multi-hop Reasoning",
        "Knowledge Graphs"
    ],
    "computational_linguistics": [
        "Computational Linguistics", 
        "Natural Language Processing", 
        "NLP", 
        "Semantics"
    ],
    "ethics_bias_ai": [
        "AI Ethics", 
        "Machine Learning Bias", 
        "Responsible AI", 
        "AI Alignment"
    ],
    "graph_reasoning_ml": [
        "Graph Neural Networks", 
        "GNN", 
        "Graph Representation Learning", 
        "Graph Machine Learning"
    ],
    "semiotics_linguistic_anthropology": [
        "Semiotics", 
        "Linguistic Anthropology", 
        "Sociolinguistics"
    ]
}

class ArxivPaperCollector:
    """
    Handles collection of papers from arXiv based on search terms.
    Uses the arXiv API to search for papers and download PDFs.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    DELAY_BETWEEN_REQUESTS = 3  # seconds, to respect arXiv's rate limits
    
    def __init__(self, output_dir: str, max_results_per_category: int = 10):
        """
        Initialize the ArxivPaperCollector.
        
        Args:
            output_dir: Directory to save downloaded papers
            max_results_per_category: Maximum number of papers to download per category
        """
        self.output_dir = output_dir
        self.max_results_per_category = max_results_per_category
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create category subdirectories
        for category in CATEGORIES.keys():
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    def search_arxiv(self, search_term: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching the search term.
        
        Args:
            search_term: The search term to use
            max_results: Maximum number of results to return
            
        Returns:
            List of papers with metadata
        """
        params = {
            'search_query': f'all:{urllib.parse.quote(search_term)}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        logger.info(f"Searching arXiv with query: {search_term}")
        
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Error searching arXiv: {response.status_code}")
            return []
        
        # Parse the XML response
        root = ET.fromstring(response.text)
        
        # Define namespace
        namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('arxiv:entry', namespace):
            paper = {
                'title': entry.find('arxiv:title', namespace).text.strip(),
                'authors': [author.find('arxiv:name', namespace).text for author in entry.findall('arxiv:author', namespace)],
                'summary': entry.find('arxiv:summary', namespace).text.strip(),
                'published': entry.find('arxiv:published', namespace).text,
                'updated': entry.find('arxiv:updated', namespace).text,
                'id': entry.find('arxiv:id', namespace).text,
                'pdf_url': None,
                'categories': []
            }
            
            # Extract PDF URL
            for link in entry.findall('arxiv:link', namespace):
                if link.get('title') == 'pdf':
                    paper['pdf_url'] = link.get('href')
            
            # Extract categories
            for category in entry.findall('.//arxiv:category', namespace):
                paper['categories'].append(category.get('term'))
            
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers for search term: {search_term}")
        return papers
    
    def download_pdf(self, paper: Dict[str, Any], category_dir: str) -> Optional[str]:
        """
        Download a PDF from arXiv.
        
        Args:
            paper: Paper metadata including PDF URL
            category_dir: Directory to save the PDF in
            
        Returns:
            Path to the downloaded PDF or None if download failed
        """
        if not paper['pdf_url']:
            logger.warning(f"No PDF URL for paper: {paper['title']}")
            return None
        
        # Extract paper ID from the arXiv ID
        arxiv_id = paper['id'].split('/')[-1]
        safe_title = re.sub(r'[^\w\s-]', '', paper['title']).strip().replace(' ', '_')
        filename = f"{arxiv_id}_{safe_title[:50]}.pdf"
        output_path = os.path.join(category_dir, filename)
        
        # Check if file already exists
        if os.path.exists(output_path):
            logger.info(f"PDF already exists: {output_path}")
            return output_path
        
        # Download the PDF
        logger.info(f"Downloading PDF: {paper['pdf_url']}")
        try:
            response = requests.get(paper['pdf_url'], stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded PDF to: {output_path}")
                return output_path
            else:
                logger.error(f"Error downloading PDF: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Exception while downloading PDF: {e}")
            return None
    
    def collect_papers_for_category(self, category: str, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Collect papers for a specific category.
        
        Args:
            category: Category name
            search_terms: List of search terms for the category
            
        Returns:
            List of papers with metadata
        """
        category_dir = os.path.join(self.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        all_papers = []
        
        for term in search_terms:
            # Search arXiv for the term
            papers = self.search_arxiv(term, max_results=self.max_results_per_category)
            
            # Download PDFs
            for paper in papers:
                # Check if we already have this paper
                if any(p['id'] == paper['id'] for p in all_papers):
                    continue
                    
                # Add category to paper metadata
                paper['hades_category'] = category
                
                # Download the PDF
                pdf_path = self.download_pdf(paper, category_dir)
                if pdf_path:
                    paper['local_pdf_path'] = pdf_path
                    all_papers.append(paper)
                
                # Respect arXiv's rate limits
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                
                # If we have enough papers, stop
                if len(all_papers) >= self.max_results_per_category:
                    break
            
            # If we have enough papers, stop
            if len(all_papers) >= self.max_results_per_category:
                break
        
        # Save metadata for all papers in this category
        metadata_path = os.path.join(category_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(all_papers, f, indent=2)
        
        logger.info(f"Collected {len(all_papers)} papers for category: {category}")
        return all_papers
    
    def collect_all_papers(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect papers for all categories.
        
        Returns:
            Dictionary mapping categories to lists of papers
        """
        all_papers_by_category = {}
        
        for category, search_terms in CATEGORIES.items():
            logger.info(f"Collecting papers for category: {category}")
            papers = self.collect_papers_for_category(category, search_terms)
            all_papers_by_category[category] = papers
        
        # Save overall metadata
        metadata_path = os.path.join(self.output_dir, "all_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(all_papers_by_category, f, indent=2)
        
        return all_papers_by_category


class PDFTextConverter:
    """
    Converts PDFs to plain text and processes them for use with PathRAG.
    Designed to handle memory constraints by processing files incrementally.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the PDFTextConverter.
        
        Args:
            output_dir: Directory to save processed text files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def convert_pdf_to_text(self, pdf_path: str) -> Optional[str]:
        """
        Convert a PDF to plain text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the text file or None if conversion failed
        """
        # Extract filename without extension
        filename = os.path.basename(pdf_path).rsplit('.', 1)[0]
        output_path = os.path.join(self.output_dir, f"{filename}.txt")
        
        # Check if file already exists
        if os.path.exists(output_path):
            logger.info(f"Text file already exists: {output_path}")
            return output_path
        
        # Use pdftotext tool (needs to be installed)
        logger.info(f"Converting PDF to text: {pdf_path}")
        try:
            import subprocess
            result = subprocess.run(['pdftotext', '-layout', pdf_path, output_path], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error converting PDF to text: {result.stderr}")
                return None
                
            logger.info(f"Converted PDF to text: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Exception while converting PDF to text: {e}")
            return None
    
    def process_text_for_pathrag(self, text_path: str) -> Optional[str]:
        """
        Process a text file for use with PathRAG.
        This includes cleaning and formatting the text.
        
        Args:
            text_path: Path to the text file
            
        Returns:
            Path to the processed text file or None if processing failed
        """
        # Extract filename without extension
        filename = os.path.basename(text_path).rsplit('.', 1)[0]
        output_path = os.path.join(self.output_dir, f"{filename}_processed.txt")
        
        # Check if file already exists
        if os.path.exists(output_path):
            logger.info(f"Processed text file already exists: {output_path}")
            return output_path
        
        logger.info(f"Processing text for PathRAG: {text_path}")
        try:
            with open(text_path, 'r', errors='ignore') as f:
                text = f.read()
            
            # Clean the text (remove multiple spaces, newlines, etc.)
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove any potential control characters
            cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_text)
            
            # Split into chunks to avoid memory issues
            chunk_size = 100000  # 100K characters per chunk
            text_chunks = [cleaned_text[i:i+chunk_size] 
                          for i in range(0, len(cleaned_text), chunk_size)]
            
            # Write processed text
            with open(output_path, 'w') as f:
                for chunk in text_chunks:
                    f.write(chunk)
            
            logger.info(f"Processed text for PathRAG: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Exception while processing text: {e}")
            return None
    
    def convert_all_pdfs(self, pdf_paths: List[str]) -> List[str]:
        """
        Convert all PDFs to text and process them for PathRAG.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of paths to processed text files
        """
        processed_files = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Convert PDFs to text
            text_paths = list(executor.map(self.convert_pdf_to_text, pdf_paths))
            
            # Process text files for PathRAG
            for text_path in text_paths:
                if text_path:
                    processed_path = self.process_text_for_pathrag(text_path)
                    if processed_path:
                        processed_files.append(processed_path)
        
        return processed_files


class HuggingFaceDatasetCollector:
    """
    Collects relevant datasets from Hugging Face to supplement the paper collection.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the HuggingFaceDatasetCollector.
        
        Args:
            output_dir: Directory to save downloaded datasets
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def download_dataset(self, dataset_name: str, subset: Optional[str] = None) -> bool:
        """
        Download a dataset from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset
            subset: Optional subset name
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"Downloading dataset: {dataset_name}" + 
                       (f", subset: {subset}" if subset else ""))
            
            dataset = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
            
            # Create directory for this dataset
            safe_name = dataset_name.replace('/', '_')
            if subset:
                safe_name += f"_{subset}"
            dataset_dir = os.path.join(self.output_dir, safe_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save dataset to text files
            for split in dataset.keys():
                split_dir = os.path.join(dataset_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                
                for i, item in enumerate(dataset[split]):
                    # Extract text from the item - adapt this based on the dataset structure
                    text = self._extract_text_from_item(item)
                    if text:
                        text_path = os.path.join(split_dir, f"item_{i}.txt")
                        with open(text_path, 'w') as f:
                            f.write(text)
            
            logger.info(f"Downloaded and saved dataset: {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Exception while downloading dataset: {e}")
            return False
    
    def _extract_text_from_item(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Extract text from a dataset item.
        
        Args:
            item: Dataset item
            
        Returns:
            Extracted text or None if no text could be extracted
        """
        # Try common field names for text
        for field in ['text', 'content', 'document', 'article', 'context']:
            if field in item and isinstance(item[field], str):
                return item[field]
        
        # If we have a question and answer, combine them
        if 'question' in item and 'answer' in item:
            return f"Question: {item['question']}\nAnswer: {item['answer']}"
        
        # If all else fails, convert the whole item to JSON
        return json.dumps(item)
    
    def collect_relevant_datasets(self) -> List[str]:
        """
        Collect relevant datasets from Hugging Face.
        
        Returns:
            List of downloaded dataset names
        """
        # Define relevant datasets for each category
        relevant_datasets = {
            "knowledge_graphs_retrieval": [
                ("chenyn/GraphRAG", None),  # Example dataset name
                ("yayay222/GraphRAG", None),
                ("openai/selfrag", None)
            ],
            "computational_linguistics": [
                ("cnn_dailymail", "3.0.0"),
                ("squad", "plain_text")
            ],
            "graph_reasoning_ml": [
                ("tigg/ogb-arxiv", None),
            ]
            # Add more datasets as needed
        }
        
        downloaded_datasets = []
        
        for category, datasets in relevant_datasets.items():
            category_dir = os.path.join(self.output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            for dataset_name, subset in datasets:
                success = self.download_dataset(dataset_name, subset)
                if success:
                    downloaded_datasets.append(dataset_name + (f"/{subset}" if subset else ""))
        
        return downloaded_datasets


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect papers from arXiv and datasets from Hugging Face for PathRAG testing")
    parser.add_argument("--output-dir", type=str, default="./pathrag_data",
                        help="Directory to save collected papers and datasets")
    parser.add_argument("--max-papers", type=int, default=20,
                        help="Maximum number of papers to collect per category")
    parser.add_argument("--include-huggingface", action="store_true",
                        help="Include datasets from Hugging Face")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories
    papers_dir = os.path.join(args.output_dir, "papers")
    text_dir = os.path.join(args.output_dir, "text")
    datasets_dir = os.path.join(args.output_dir, "datasets")
    
    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Collect papers from arXiv
    logger.info("Collecting papers from arXiv")
    collector = ArxivPaperCollector(papers_dir, max_results_per_category=args.max_papers)
    papers_by_category = collector.collect_all_papers()
    
    # Convert PDFs to text
    logger.info("Converting PDFs to text")
    converter = PDFTextConverter(text_dir)
    
    all_pdf_paths = []
    for category, papers in papers_by_category.items():
        for paper in papers:
            if 'local_pdf_path' in paper:
                all_pdf_paths.append(paper['local_pdf_path'])
    
    processed_text_files = converter.convert_all_pdfs(all_pdf_paths)
    logger.info(f"Processed {len(processed_text_files)} text files")
    
    # Collect datasets from Hugging Face if requested
    if args.include_huggingface:
        logger.info("Collecting datasets from Hugging Face")
        hf_collector = HuggingFaceDatasetCollector(datasets_dir)
        downloaded_datasets = hf_collector.collect_relevant_datasets()
        logger.info(f"Downloaded {len(downloaded_datasets)} datasets from Hugging Face")
    
    logger.info("Dataset collection complete")


if __name__ == "__main__":
    main()
