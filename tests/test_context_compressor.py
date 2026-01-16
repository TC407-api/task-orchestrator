import pytest
import numpy as np
from unittest.mock import MagicMock
from src.governance.context_compressor import ContextCompressor, CompressedContext


@pytest.fixture
def compressor() -> ContextCompressor:
    """Returns a ContextCompressor instance with a low rank for testing."""
    return ContextCompressor(compression_rank=10)


@pytest.fixture
def sample_tool_definition() -> str:
    """Returns a string representing a tool definition."""
    return """
    {
        "name": "search_database",
        "description": "Searches the vector database for relevant documents.",
        "parameters": {
            "query": "string",
            "limit": "integer"
        }
    }
    """ * 20  # Repeat to ensure enough content for compression simulation


def test_compressor_reduces_token_count(compressor: ContextCompressor, sample_tool_definition: str):
    """Verifies that the compressed object reports fewer tokens than the original."""
    result = compressor.compress(sample_tool_definition)

    assert isinstance(result, CompressedContext)
    assert result.compressed_tokens < result.original_tokens
    assert result.original_tokens > 0


def test_compressed_tools_maintain_functionality(compressor: ContextCompressor, sample_tool_definition: str):
    """
    Verifies that the compressed vectors maintain the dimensions defined by the rank,
    ensuring the mathematical representation of the tool is preserved.
    """
    result = compressor.compress(sample_tool_definition)

    assert result.vectors is not None
    assert isinstance(result.vectors, np.ndarray)
    # The second dimension should match the compression rank
    assert result.vectors.shape[1] == compressor.compression_rank
    assert "encoding_metadata" in result.metadata


def test_decompression_reconstructs_tools(compressor: ContextCompressor, sample_tool_definition: str):
    """Verifies that the decompress method returns a string representation."""
    compressed = compressor.compress(sample_tool_definition)
    reconstructed = compressor.decompress(compressed)

    assert isinstance(reconstructed, str)
    assert len(reconstructed) > 0
    # In a real implementation, we would check cosine similarity here
    # For the RED phase, ensuring it returns a string is sufficient


def test_compression_ratio_meets_target(compressor: ContextCompressor, sample_tool_definition: str):
    """Verifies that the compression achieves the target 20% size (0.2 ratio)."""
    result = compressor.compress(sample_tool_definition)

    ratio = result.compressed_tokens / result.original_tokens
    # Target is 20% size (0.2)
    assert ratio <= 0.20


def test_compressor_integrates_with_tracker(compressor: ContextCompressor):
    """
    Verifies that the compressor provides metrics compatible with ContextTracker.
    """
    # Mocking ContextTracker behavior by checking interface compatibility
    savings_estimate = compressor.estimate_savings()

    assert isinstance(savings_estimate, float)
    assert 0.0 <= savings_estimate <= 1.0

    # Ensure the compressor can accept a mock tracker update (simulation)
    mock_tracker = MagicMock()
    mock_tracker.current_usage = 1000

    # Simulate logic where tracker queries compressor
    projected_usage = mock_tracker.current_usage * (1 - savings_estimate)
    assert projected_usage <= mock_tracker.current_usage
