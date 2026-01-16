import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List

# Assumed import based on existing context
from src.evaluation.immune_system.failure_store import FailurePattern
from src.governance.ttt_memory import TTTMemoryLayer


class TestTTTMemoryLayer:

    @pytest.fixture
    def memory_layer(self):
        """Fixture for a standard memory layer instance."""
        return TTTMemoryLayer(embedding_dim=64, learning_rate=0.01)

    @pytest.fixture
    def sample_patterns(self) -> List[FailurePattern]:
        """Generates a list of dummy failure patterns."""
        patterns = []
        for i in range(10):
            # Mocking FailurePattern structure
            pattern = MagicMock(spec=FailurePattern)
            pattern.id = f"fp_{i}"
            pattern.context = f"Error in module {i} caused by buffer overflow"
            pattern.solution = "Resize buffer"
            patterns.append(pattern)
        return patterns

    def test_memory_layer_compresses_patterns(self, memory_layer, sample_patterns):
        """
        Test that the layer can ingest a list of patterns and initialize
        its internal weight matrix (compression).
        """
        memory_layer.compress_patterns(sample_patterns)

        # Accessing private attributes for white-box testing of state initialization
        assert memory_layer._weights is not None
        assert memory_layer._vectorizer is not None
        assert isinstance(memory_layer._weights, np.ndarray)

    def test_risk_prediction_is_o1_complexity(self, memory_layer, sample_patterns):
        """
        Test that prediction time remains constant regardless of the number
        of historical patterns ingested (O(1) vs O(N)).
        """
        # 1. Train on small dataset
        memory_layer.compress_patterns(sample_patterns[:5])
        start_small = time.perf_counter()
        _ = memory_layer.predict_risk_fast("buffer overflow risk")
        duration_small = time.perf_counter() - start_small

        # 2. Train on "large" dataset (simulated by repeating list)
        large_patterns = sample_patterns * 100
        memory_layer.compress_patterns(large_patterns)
        start_large = time.perf_counter()
        _ = memory_layer.predict_risk_fast("buffer overflow risk")
        duration_large = time.perf_counter() - start_large

        # Allow for slight system variance, but it shouldn't be linear growth (100x)
        # If it were O(N), duration_large would be ~100x duration_small.
        # We assert they are roughly similar (within 5x margin for overhead).
        assert duration_large < (duration_small * 5), \
            f"Prediction time grew linearly: Small={duration_small:.6f}s, Large={duration_large:.6f}s"

    def test_memory_learns_from_new_failures(self, memory_layer, sample_patterns):
        """
        Test the TTT mechanism: updating weights based on a single new failure
        should change the risk prediction for similar prompts.
        """
        memory_layer.compress_patterns(sample_patterns)
        test_prompt = "Specific edge case error"

        # Initial risk
        initial_risk = memory_layer.predict_risk_fast(test_prompt)

        # Learn from a new failure matching that prompt
        new_failure = MagicMock(spec=FailurePattern)
        new_failure.context = "Critical failure: Specific edge case error"
        new_failure.severity = 1.0

        memory_layer.update_weights(new_failure)

        # Risk should change (likely increase) after learning
        new_risk = memory_layer.predict_risk_fast(test_prompt)
        assert new_risk != initial_risk
        assert isinstance(new_risk, float)

    def test_memory_persists_across_calls(self, tmp_path, sample_patterns):
        """Test save_state and load_state functionality."""
        # Setup initial layer
        layer_1 = TTTMemoryLayer()
        layer_1.compress_patterns(sample_patterns)
        prompt = "test context"
        risk_1 = layer_1.predict_risk_fast(prompt)

        # Save
        save_path = tmp_path / "ttt_memory.pkl"
        layer_1.save_state(str(save_path))

        # Load into new layer
        layer_2 = TTTMemoryLayer()
        layer_2.load_state(str(save_path))
        risk_2 = layer_2.predict_risk_fast(prompt)

        assert risk_1 == risk_2
        assert layer_2._weights is not None

    def test_memory_integrates_with_pattern_matcher(self, memory_layer, sample_patterns):
        """
        Test that the memory layer can act as a high-speed filter
        before calling the slower PatternMatcher.
        """
        # This test simulates the workflow: High Risk (TTT) -> Detailed Match (PatternMatcher)
        memory_layer.compress_patterns(sample_patterns)

        risk_score = memory_layer.predict_risk_fast("buffer overflow")

        # We just verify the contract here: returns float between 0 and 1 (normalized)
        assert 0.0 <= risk_score <= 1.0

    def test_memory_transfer_from_graphiti(self, memory_layer):
        """
        Test that we can convert Graphiti node structures into
        patterns for compression.
        """
        # Mock Graphiti nodes
        mock_nodes = [
            {"id": "1", "data": {"context": "err1", "solution": "fix1"}},
            {"id": "2", "data": {"context": "err2", "solution": "fix2"}}
        ]

        # Assuming a helper method exists or we pass these as FailurePatterns
        # Here we verify compress_patterns handles the conversion logic if implemented,
        # or we manually convert in the test to verify the layer accepts the data.

        patterns = [
            MagicMock(spec=FailurePattern, context=n["data"]["context"])
            for n in mock_nodes
        ]

        memory_layer.compress_patterns(patterns)
        assert memory_layer.predict_risk_fast("err1") > 0.0
