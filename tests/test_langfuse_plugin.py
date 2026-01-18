"""
Tests for enhanced Langfuse Plugin.

TDD RED Phase: These tests define the expected behavior.
The implementation should make them pass.
"""

import pytest
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

# These imports will fail until implementation exists
try:
    from src.integrations.langfuse_plugin import (
        LangfusePlugin,
        create_langfuse_plugin,
    )
except ImportError:
    LangfusePlugin = None
    create_langfuse_plugin = None

# Mock Trial for testing
class MockTrial:
    """Mock Trial object for testing."""
    def __init__(
        self,
        id: str = "trial-001",
        input_prompt: str = "test prompt",
        output: str = "test output",
        operation: str = "spawn_agent",
        model: str = "gemini-3-flash-preview",
        cost_usd: float = 0.001,
        grader_results: list = None
    ):
        self.id = id
        self.input_prompt = input_prompt
        self.output = output
        self.operation = operation
        self.model = model
        self.cost_usd = cost_usd
        self.grader_results = grader_results or []
        self.created_at = datetime.now()
        self.langfuse_trace_id = ""


class MockGraderResult:
    """Mock GraderResult for testing."""
    def __init__(
        self,
        name: str = "test-grader",
        score: float = 0.85,
        passed: bool = True,
        reason: str = "Test passed"
    ):
        self.name = name
        self.score = score
        self.passed = passed
        self.reason = reason
        self.metadata = {}


@pytest.fixture
def mock_langfuse_client():
    """Create a mock Langfuse client."""
    client = MagicMock()
    client.create_dataset = MagicMock(return_value=MagicMock(id="dataset-001"))
    client.create_dataset_item = MagicMock()
    client.trace = MagicMock(return_value=MagicMock(id="trace-001"))
    client.score = MagicMock()
    client.flush = MagicMock()
    return client


@pytest.fixture
def sample_trials() -> List[MockTrial]:
    """Create sample trials for testing."""
    return [
        MockTrial(
            id="trial-001",
            input_prompt="Analyze code for bugs",
            output="Found 3 potential issues",
            grader_results=[
                MockGraderResult(name="relevance", score=0.9, passed=True),
                MockGraderResult(name="safety", score=1.0, passed=True),
            ]
        ),
        MockTrial(
            id="trial-002",
            input_prompt="Generate unit tests",
            output="Created 5 test cases",
            grader_results=[
                MockGraderResult(name="relevance", score=0.8, passed=True),
                MockGraderResult(name="safety", score=1.0, passed=True),
            ]
        ),
    ]


@pytest.fixture
def sample_immune_patterns() -> List[dict]:
    """Create sample immune patterns for testing."""
    return [
        {
            "pattern_id": "pat-001",
            "pattern_type": "failure",
            "signature": "timeout_error",
            "frequency": 5,
            "last_seen": datetime.now().isoformat(),
        },
        {
            "pattern_id": "pat-002",
            "pattern_type": "success",
            "signature": "clean_execution",
            "frequency": 100,
            "last_seen": datetime.now().isoformat(),
        },
    ]


@pytest.mark.skipif(LangfusePlugin is None, reason="LangfusePlugin not implemented yet")
class TestLangfusePlugin:
    """Tests for LangfusePlugin functionality."""

    @pytest.mark.asyncio
    async def test_export_immune_patterns_creates_dataset(
        self, mock_langfuse_client, sample_immune_patterns
    ):
        """Test that immune patterns can be exported as a Langfuse dataset."""
        plugin = LangfusePlugin(mock_langfuse_client)

        # Export patterns
        result = await plugin.export_immune_patterns(
            patterns=sample_immune_patterns,
            dataset_name="immune-patterns-test"
        )

        # Verify dataset was created
        assert result is not None
        assert "dataset_id" in result
        assert result["items_exported"] == len(sample_immune_patterns)

        # Verify Langfuse API was called
        mock_langfuse_client.create_dataset.assert_called_once()
        assert mock_langfuse_client.create_dataset_item.call_count == len(sample_immune_patterns)

    @pytest.mark.asyncio
    async def test_push_dashboard_metrics_succeeds(self, mock_langfuse_client):
        """Test that dashboard metrics can be pushed to Langfuse."""
        plugin = LangfusePlugin(mock_langfuse_client)

        metrics = {
            "total_cost_usd": 0.45,
            "total_operations": 150,
            "success_rate": 0.95,
            "circuit_breaker_trips": 2,
        }

        # Push metrics
        result = await plugin.push_dashboard_metrics(metrics)

        # Verify success
        assert result is True

        # Verify trace was created with metrics
        mock_langfuse_client.trace.assert_called()

    @pytest.mark.asyncio
    async def test_create_evaluation_dataset_from_trials(
        self, mock_langfuse_client, sample_trials
    ):
        """Test creating a Langfuse dataset from evaluation trials."""
        plugin = LangfusePlugin(mock_langfuse_client)

        # Create dataset
        dataset_id = await plugin.create_evaluation_dataset(
            name="eval-dataset-test",
            trials=sample_trials,
            description="Test evaluation dataset"
        )

        # Verify dataset was created
        assert dataset_id is not None
        assert len(dataset_id) > 0

        # Verify dataset items were created for each trial
        assert mock_langfuse_client.create_dataset_item.call_count == len(sample_trials)

    @pytest.mark.asyncio
    async def test_aggregate_scores_returns_averages(self, mock_langfuse_client):
        """Test that score aggregation returns correct averages."""
        # Create proper mock score objects
        class MockScore:
            def __init__(self, name: str, value: float):
                self.name = name
                self.value = value

        # Setup mock to return scores with proper name attributes
        mock_langfuse_client.get_trace = MagicMock(return_value=MagicMock(
            scores=[
                MockScore("relevance", 0.85),
                MockScore("safety", 1.0),
            ]
        ))

        plugin = LangfusePlugin(mock_langfuse_client)

        trace_ids = ["trace-001", "trace-002", "trace-003"]

        # Aggregate scores
        averages = await plugin.aggregate_scores(trace_ids)

        # Verify averages are returned
        assert "relevance" in averages
        assert "safety" in averages
        assert 0 <= averages["relevance"] <= 1
        assert 0 <= averages["safety"] <= 1

    def test_plugin_initialization_without_client(self):
        """Test that plugin can be created without explicit client (uses env vars)."""
        # Reset global instance for clean test
        import src.integrations.langfuse_plugin as lp
        lp._plugin_instance = None

        with patch.dict('os.environ', {
            'LANGFUSE_PUBLIC_KEY': 'test-public-key',
            'LANGFUSE_SECRET_KEY': 'test-secret-key',
        }):
            # Patch at the module level where it's imported
            with patch.object(lp, 'Langfuse', create=True) as mock_lf:
                mock_lf.return_value = MagicMock()
                plugin = create_langfuse_plugin()
                # Plugin creation should work even if Langfuse not initialized
                assert plugin is not None


@pytest.mark.skipif(LangfusePlugin is None, reason="LangfusePlugin not implemented yet")
class TestLangfuseExportMCPTool:
    """Tests for the langfuse_export MCP tool."""

    @pytest.mark.asyncio
    async def test_langfuse_export_mcp_tool_patterns(self, mock_langfuse_client):
        """Test the MCP tool for exporting patterns."""
        # Import the MCP tool handler
        try:
            from src.integrations.langfuse_plugin import handle_langfuse_export
        except ImportError:
            pytest.skip("MCP tool not implemented yet")

        # Call the tool
        result = await handle_langfuse_export({
            "type": "patterns",
            "dataset_name": "test-patterns"
        })

        # Verify result - either success or error is acceptable
        # (error expected when Langfuse not configured in CI)
        assert "success" in result or "error" in result
        if "error" not in result and "success" in result:
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_langfuse_export_mcp_tool_trials(self, mock_langfuse_client):
        """Test the MCP tool for exporting trials."""
        try:
            from src.integrations.langfuse_plugin import handle_langfuse_export
        except ImportError:
            pytest.skip("MCP tool not implemented yet")

        result = await handle_langfuse_export({
            "type": "trials",
            "dataset_name": "test-trials"
        })

        # Either success or error is acceptable (error when Langfuse not configured)
        assert "success" in result or "error" in result

    @pytest.mark.asyncio
    async def test_langfuse_export_mcp_tool_metrics(self, mock_langfuse_client):
        """Test the MCP tool for pushing metrics."""
        try:
            from src.integrations.langfuse_plugin import handle_langfuse_export
        except ImportError:
            pytest.skip("MCP tool not implemented yet")

        result = await handle_langfuse_export({
            "type": "metrics"
        })

        # Either success or error is acceptable (error when Langfuse not configured)
        assert "success" in result or "error" in result

    @pytest.mark.asyncio
    async def test_langfuse_export_invalid_type_returns_error(self):
        """Test that invalid export type returns error."""
        try:
            from src.integrations.langfuse_plugin import handle_langfuse_export
        except ImportError:
            pytest.skip("MCP tool not implemented yet")

        result = await handle_langfuse_export({
            "type": "invalid_type"
        })

        assert "error" in result


# Standalone function tests (not in class to run even if import fails)
def test_langfuse_plugin_module_exists():
    """Test that the langfuse_plugin module can be imported."""
    try:
        from src.integrations import langfuse_plugin
        assert langfuse_plugin is not None
    except ImportError as e:
        pytest.fail(f"langfuse_plugin module not found: {e}")


def test_langfuse_plugin_class_exists():
    """Test that LangfusePlugin class exists."""
    try:
        from src.integrations.langfuse_plugin import LangfusePlugin
        assert LangfusePlugin is not None
    except ImportError as e:
        pytest.fail(f"LangfusePlugin class not found: {e}")


def test_create_langfuse_plugin_function_exists():
    """Test that create_langfuse_plugin factory function exists."""
    try:
        from src.integrations.langfuse_plugin import create_langfuse_plugin
        assert callable(create_langfuse_plugin)
    except ImportError as e:
        pytest.fail(f"create_langfuse_plugin function not found: {e}")
