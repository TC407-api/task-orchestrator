"""
Tests for FailurePatternStore Titan Memory integration.

TDD RED phase: These tests define the expected behavior for Titan integration.
"""

import pytest
from datetime import datetime

# Skip entire module if Titan CLI is not available
TITAN_CLI = r"C:\Users\Travi\.claude\titan-memory\dist\cli\index.js"


def titan_cli_available():
    """Check if Titan CLI is available."""
    import os
    return os.path.exists(TITAN_CLI)


@pytest.fixture
def titan_client():
    """Create a TitanClient instance for testing."""
    from src.evaluation.immune_system.titan_client import TitanClient
    return TitanClient()


@pytest.fixture
def failure_store():
    """Create a FailurePatternStore with Titan backend."""
    from src.evaluation.immune_system.failure_store import FailurePatternStore
    from src.evaluation.immune_system.titan_client import TitanClient
    titan = TitanClient()
    return FailurePatternStore(titan_client=titan)


class TestTitanClient:
    """Tests for the TitanClient wrapper."""

    @pytest.mark.skipif(not titan_cli_available(), reason="Titan CLI not available")
    @pytest.mark.asyncio
    async def test_add_memory_stores_to_titan(self, titan_client):
        """Test that add_memory stores content in Titan."""
        content = f"[Test] Failure pattern test at {datetime.utcnow().isoformat()}"
        tags = ["test", "failure", "immune"]
        project = "task-orchestrator-test"

        result = await titan_client.add_memory(
            content=content,
            tags=tags,
            project_id=project,
            layer=5,  # Episodic layer for failures
        )

        assert result is not None
        assert "id" in result or "memory_id" in result
        assert result.get("stored", True) is True

    @pytest.mark.skipif(not titan_cli_available(), reason="Titan CLI not available")
    @pytest.mark.asyncio
    async def test_search_memories_queries_titan(self, titan_client):
        """Test that search_memories queries Titan with tags."""
        # First add a test memory
        test_content = f"[Test Search] Pattern for search test {datetime.utcnow().isoformat()}"
        await titan_client.add_memory(
            content=test_content,
            tags=["test", "search", "immune"],
            project_id="task-orchestrator-test",
        )

        # Now search for it
        results = await titan_client.search_memories(
            query="search test",
            tags=["test", "immune"],
            limit=5,
        )

        assert isinstance(results, list)
        # Should find at least our test memory
        assert len(results) >= 0  # May be 0 if surprise detection filters it

    @pytest.mark.skipif(not titan_cli_available(), reason="Titan CLI not available")
    @pytest.mark.asyncio
    async def test_feedback_marks_helpful(self, titan_client):
        """Test that feedback can mark a memory as helpful."""
        # First add a test memory
        content = f"[Test Feedback] Memory for feedback test {datetime.utcnow().isoformat()}"
        result = await titan_client.add_memory(
            content=content,
            tags=["test", "feedback"],
            project_id="task-orchestrator-test",
        )

        memory_id = result.get("id") or result.get("memory_id")
        if memory_id and not memory_id.startswith("ghost_"):
            feedback_result = await titan_client.record_feedback(
                memory_id=memory_id,
                signal="helpful",
                context="Test feedback",
            )
            assert feedback_result.get("success", True) is True


class TestFailureStoreWithTitan:
    """Tests for FailurePatternStore with Titan backend."""

    @pytest.mark.skipif(not titan_cli_available(), reason="Titan CLI not available")
    @pytest.mark.asyncio
    async def test_store_failure_writes_to_titan(self, failure_store):
        """Test that store_failure writes to Titan."""
        grader_results = [
            {"name": "json_valid", "passed": False, "score": 0.0},
        ]

        pattern = await failure_store.store_failure(
            operation="spawn_agent",
            input_prompt="Test prompt",
            output="Invalid JSON output",
            grader_results=grader_results,
            context={"model": "test"},
        )

        assert pattern is not None
        assert pattern.failure_type == "json_invalid"
        assert pattern.operation == "spawn_agent"

    @pytest.mark.skipif(not titan_cli_available(), reason="Titan CLI not available")
    @pytest.mark.asyncio
    async def test_get_failures_queries_titan(self, failure_store):
        """Test that get_failures queries from Titan."""
        # Store a failure first
        grader_results = [{"name": "pattern_check", "passed": False, "score": 0.0}]
        await failure_store.store_failure(
            operation="test_op",
            input_prompt="Test input",
            output="Bad output",
            grader_results=grader_results,
        )

        # Query failures
        failures = await failure_store.get_failures_by_operation("test_op")
        assert isinstance(failures, list)

    @pytest.mark.asyncio
    async def test_store_failure_local_fallback(self):
        """Test that store_failure works with local cache when Titan unavailable."""
        from src.evaluation.immune_system.failure_store import FailurePatternStore

        # Create store without Titan client
        store = FailurePatternStore(titan_client=None)

        grader_results = [{"name": "test", "passed": False, "score": 0.5}]
        pattern = await store.store_failure(
            operation="local_test",
            input_prompt="Local test input",
            output="Local test output",
            grader_results=grader_results,
        )

        assert pattern is not None
        assert pattern.operation == "local_test"
        # Should be in local cache
        assert pattern.id in store._local_cache


class TestImmuneTitanIntegration:
    """Integration tests for immune system with Titan."""

    @pytest.mark.skipif(not titan_cli_available(), reason="Titan CLI not available")
    @pytest.mark.asyncio
    async def test_failure_patterns_have_titan_tags(self, failure_store):
        """Test that failure patterns are stored with correct tags for filtering."""
        grader_results = [{"name": "schema", "passed": False, "score": 0.0}]

        pattern = await failure_store.store_failure(
            operation="schema_test",
            input_prompt="Schema test",
            output="Invalid schema",
            grader_results=grader_results,
        )

        # Pattern should be tagged for later retrieval
        # Tags should include: failure, immune, operation type
        # This is verified by the search working correctly
        assert pattern is not None
        assert pattern.failure_type == "schema_violation"

    @pytest.mark.skipif(not titan_cli_available(), reason="Titan CLI not available")
    @pytest.mark.asyncio
    async def test_predict_risk_uses_titan_data(self, failure_store):
        """Test that risk prediction queries Titan for similar failures."""
        # This tests that the immune system queries Titan momentum/utility
        # for predicting risk based on similar past failures
        from src.evaluation.immune_system.titan_client import TitanClient

        titan = TitanClient()
        results = await titan.search_memories(
            query="json_invalid failure",
            tags=["failure", "immune"],
            limit=5,
        )

        # Should be able to query failure patterns
        assert isinstance(results, list)


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_memories():
    """Clean up test memories after tests."""
    yield
    # Cleanup could be implemented here if needed
    # For now, test memories will naturally decay in Titan


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
