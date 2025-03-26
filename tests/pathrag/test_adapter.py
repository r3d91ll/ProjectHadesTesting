"""
Tests for the PathRAG adapter implementation.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add implementations directory to path for importing
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the adapter to test
from implementations.pathrag.original.adapter import PathRAGAdapter


@pytest.fixture
def mock_pathrag_implementation():
    """Mock the original PathRAG implementation for testing."""
    with patch('implementations.pathrag.original.adapter.OriginalPathRAG') as mock_pathrag:
        # Configure the mock
        instance = mock_pathrag.return_value
        instance.answer.return_value = {
            "answer": "Test answer",
            "paths": [{"id": "1", "text": "Test path 1"}, {"id": "2", "text": "Test path 2"}],
            "context": "Test context"
        }
        instance.retrieve_paths.return_value = [
            {"id": "1", "text": "Test path 1"}, 
            {"id": "2", "text": "Test path 2"}
        ]
        yield mock_pathrag


@pytest.fixture
def mock_init_model():
    """Mock the init_model function."""
    with patch('implementations.pathrag.original.adapter.init_model') as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_init_tokenizer():
    """Mock the init_tokenizer function."""
    with patch('implementations.pathrag.original.adapter.init_tokenizer') as mock:
        mock.return_value = MagicMock()
        yield mock


def test_adapter_initialization(mock_pathrag_implementation, mock_init_model, mock_init_tokenizer):
    """Test that the adapter initializes correctly."""
    config = {"model_name": "gpt2"}
    adapter = PathRAGAdapter(config)
    
    # Initialization should not call the original implementation yet
    mock_pathrag_implementation.assert_not_called()
    
    # Initialize the adapter
    adapter.initialize()
    
    # Now it should have called the init functions and created the implementation
    mock_init_model.assert_called_once_with("gpt2")
    mock_init_tokenizer.assert_called_once_with("gpt2")
    mock_pathrag_implementation.assert_called_once()


def test_adapter_query(mock_pathrag_implementation, mock_init_model, mock_init_tokenizer):
    """Test that the adapter's query method works correctly."""
    config = {"model_name": "gpt2"}
    adapter = PathRAGAdapter(config)
    adapter.initialize()
    
    # Reset the mock to clear the initialization call
    mock_pathrag_implementation.reset_mock()
    
    # Get the mock instance
    instance = mock_pathrag_implementation.return_value
    
    # Test querying
    result = adapter.query("Test query")
    
    # Verify the original implementation was called correctly
    instance.answer.assert_called_once_with("Test query")
    
    # Verify the result format
    assert result["answer"] == "Test answer"
    assert len(result["paths"]) == 2
    assert result["context"] == "Test context"
    assert "raw_result" in result


def test_adapter_get_paths(mock_pathrag_implementation, mock_init_model, mock_init_tokenizer):
    """Test that the adapter's get_paths method works correctly."""
    config = {"model_name": "gpt2"}
    adapter = PathRAGAdapter(config)
    adapter.initialize()
    
    # Reset the mock to clear the initialization call
    mock_pathrag_implementation.reset_mock()
    
    # Get the mock instance
    instance = mock_pathrag_implementation.return_value
    
    # Test getting paths
    paths = adapter.get_paths("Test query", top_k=3)
    
    # Verify the original implementation was called correctly
    instance.retrieve_paths.assert_called_once_with("Test query", top_k=3)
    
    # Verify the paths format
    assert len(paths) == 2
    assert paths[0]["id"] == "1"
    assert paths[1]["text"] == "Test path 2"
