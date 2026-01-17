"""Tests for ContextTracker - TDD RED phase."""
from unittest.mock import Mock
from src.mcp.context_tracker import ContextTracker


def test_tracker_initializes_with_defaults():
    """Test default initialization values."""
    tracker = ContextTracker()
    assert tracker.max_context == 200000
    assert tracker.threshold_pct == 0.1
    assert tracker.current_usage == 0


def test_update_usage_tracks_tokens():
    """Test that update_usage accumulates tokens correctly."""
    tracker = ContextTracker(max_context=100000)
    tracker.update_usage(5000)
    assert tracker.current_usage == 5000

    tracker.update_usage(2000)
    assert tracker.current_usage == 7000


def test_below_threshold_returns_true_at_10_percent():
    """Test threshold detection at 10% remaining context."""
    # Max 1000, threshold 10% (100 tokens remaining, so usage > 900)
    tracker = ContextTracker(max_context=1000, threshold_pct=0.1)

    # Usage 800 (80%) -> 200 remaining (20%) -> Not below threshold
    tracker.update_usage(800)
    assert tracker.below_threshold() is False

    # Usage 901 (90.1%) -> 99 remaining (9.9%) -> Below threshold
    tracker.update_usage(101)  # Total 901
    assert tracker.below_threshold() is True


def test_estimate_tool_tokens_caches_results():
    """Test that token estimation caches results for performance."""
    mock_estimator = Mock(return_value=100)
    tracker = ContextTracker(token_estimator=mock_estimator)

    tool_def = {"name": "test_tool", "description": "a test tool"}

    # First call should use estimator
    tokens1 = tracker.estimate_tool_tokens(tool_def)
    assert tokens1 == 100
    mock_estimator.assert_called_once()

    # Second call with same tool should use cache
    tokens2 = tracker.estimate_tool_tokens(tool_def)
    assert tokens2 == 100
    mock_estimator.assert_called_once()  # Call count shouldn't increase


def test_get_stats_returns_complete_info():
    """Test that get_stats returns all required information."""
    tracker = ContextTracker(max_context=1000)
    tracker.update_usage(500)

    stats = tracker.get_stats()
    assert stats["max_context"] == 1000
    assert stats["current_usage"] == 500
    assert stats["remaining_tokens"] == 500
    assert stats["usage_pct"] == 0.5
    assert stats["threshold_warning"] is False
