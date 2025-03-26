"""
Tests for the Neo4j GraphRAG adapter implementation.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add implementations directory to path for importing
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the adapter to test
from implementations.graphrag.original.adapter import GraphRAGAdapter


@pytest.fixture
def mock_simplepipeline():
    """Mock the Neo4j GraphRAG SimpleKGPipeline for testing."""
    with patch('implementations.graphrag.original.adapter.SimpleKGPipeline') as mock_pipeline:
        # Configure the mock
        instance = mock_pipeline.return_value
        instance.execute.return_value = {
            "answer": "Test answer from Neo4j GraphRAG",
            "context": "Test context from Neo4j",
            "paths": [{"id": "1", "text": "Neo4j path 1"}, {"id": "2", "text": "Neo4j path 2"}]
        }
        
        # Mock the retriever
        instance.retriever = MagicMock()
        instance.retriever.retrieve.return_value = [
            {"id": "1", "text": "Neo4j path 1"}, 
            {"id": "2", "text": "Neo4j path 2"}
        ]
        
        yield mock_pipeline


@pytest.fixture
def mock_openai_llm():
    """Mock the OpenAI LLM for Neo4j GraphRAG."""
    with patch('implementations.graphrag.original.adapter.OpenAILLM') as mock_llm:
        mock_llm.return_value = MagicMock()
        yield mock_llm


def test_adapter_initialization(mock_simplepipeline, mock_openai_llm):
    """Test that the Neo4j GraphRAG adapter initializes correctly."""
    config = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "test123",
        "neo4j_database": "neo4j",
        "openai_api_key": "test-key"
    }
    adapter = GraphRAGAdapter(config)
    
    # Initialization should not create the pipeline yet
    mock_simplepipeline.assert_not_called()
    mock_openai_llm.assert_not_called()
    
    # Initialize the adapter
    adapter.initialize()
    
    # Now it should have created the pipeline with correct parameters
    mock_openai_llm.assert_called_once_with(api_key="test-key")
    mock_simplepipeline.assert_called_once()
    
    # Verify the connection parameters
    call_kwargs = mock_simplepipeline.call_args.kwargs
    assert call_kwargs["uri"] == "bolt://localhost:7687"
    assert call_kwargs["username"] == "neo4j"
    assert call_kwargs["password"] == "test123"
    assert call_kwargs["database"] == "neo4j"


def test_adapter_query(mock_simplepipeline, mock_openai_llm):
    """Test that the adapter's query method works correctly."""
    config = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "test123",
        "neo4j_database": "neo4j",
        "openai_api_key": "test-key"
    }
    adapter = GraphRAGAdapter(config)
    adapter.initialize()
    
    # Reset the mock to clear the initialization call
    mock_simplepipeline.reset_mock()
    
    # Get the mock instance
    instance = mock_simplepipeline.return_value
    
    # Test querying
    result = adapter.query("Test GraphRAG query", top_k=5)
    
    # Verify the pipeline was called correctly
    instance.execute.assert_called_once_with(query="Test GraphRAG query", top_k=5)
    
    # Verify the result format
    assert result["answer"] == "Test answer from Neo4j GraphRAG"
    assert len(result["paths"]) == 2
    assert result["context"] == "Test context from Neo4j"
    assert "raw_result" in result


def test_adapter_get_paths(mock_simplepipeline, mock_openai_llm):
    """Test that the adapter's get_paths method works correctly."""
    config = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "test123",
        "neo4j_database": "neo4j",
        "openai_api_key": "test-key"
    }
    adapter = GraphRAGAdapter(config)
    adapter.initialize()
    
    # Reset the mock to clear the initialization call
    mock_simplepipeline.reset_mock()
    
    # Get the mock instance
    instance = mock_simplepipeline.return_value
    
    # Test getting paths
    paths = adapter.get_paths("Test GraphRAG query", top_k=3)
    
    # Verify the retriever was called correctly
    instance.retriever.retrieve.assert_called_once_with(query="Test GraphRAG query", top_k=3)
    
    # Verify the paths format
    assert len(paths) == 2
    assert paths[0]["id"] == "1"
    assert paths[1]["text"] == "Neo4j path 2"
