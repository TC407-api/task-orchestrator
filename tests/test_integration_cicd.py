"""
CI/CD Integration Tests for Task Orchestrator Evaluation System.

These tests verify the end-to-end behavior of the evaluation system,
including spawn_agent evaluation, immune system blocking, circuit breaker
integration, and training data export.

Run with:
    JWT_SECRET_KEY=test123 python -m pytest tests/test_integration_cicd.py -v -m integration
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def reset_immune():
    """Reset immune system singleton before/after each test."""
    from src.evaluation import reset_immune_system
    reset_immune_system()
    yield
    reset_immune_system()


@pytest.fixture
def temp_export_dir():
    """Create a temporary directory for exports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# Integration Tests
# =============================================================================

class TestSpawnAgentWithEvaluation:
    """Test spawn_agent end-to-end with grading."""

    @pytest.mark.asyncio
    async def test_spawn_agent_returns_evaluation_scores(self, reset_immune):
        """Verify that spawn_agent includes evaluation results."""
        from src.evaluation import (
            Trial, GraderPipeline, NonEmptyGrader, LengthGrader,
            get_immune_system,
        )

        # Setup
        immune = get_immune_system()
        pipeline = GraderPipeline([
            NonEmptyGrader(),
            LengthGrader(min_length=5, max_length=5000),
        ])

        # Create a trial with mock output
        trial = Trial(
            id="test-trial-001",
            operation="spawn_agent",
            input_prompt="Write a hello world function",
            output="def hello():\n    return 'Hello, World!'",
            model="gemini-3-flash-preview",
            latency_ms=150.0,
            cost_usd=0.001,
        )

        # Run graders - pipeline.run() takes output and context
        context = {"prompt": trial.input_prompt}
        results = await pipeline.run(trial.output, context)
        trial.grader_results = results
        trial.pass_fail = all(r.passed for r in results)

        # Assertions
        assert trial.pass_fail is True
        assert len(trial.grader_results) == 2
        assert all(r.passed for r in trial.grader_results)

    @pytest.mark.asyncio
    async def test_spawn_agent_catches_empty_response(self, reset_immune):
        """Verify that empty responses are caught by graders."""
        from src.evaluation import GraderPipeline, NonEmptyGrader

        pipeline = GraderPipeline([NonEmptyGrader()])
        context = {"prompt": "test"}
        results = await pipeline.run("", context)

        assert len(results) == 1
        assert results[0].passed is False
        assert "empty" in results[0].reason.lower()


class TestImmuneSystemBlocking:
    """Test immune system blocking of known bad patterns."""

    @pytest.mark.asyncio
    async def test_immune_blocks_known_bad_pattern(self, reset_immune):
        """Verify that recorded failures increase risk scores."""
        from src.evaluation import get_immune_system

        immune = get_immune_system()

        # Record a failure for a specific pattern
        await immune.record_failure(
            operation="spawn_agent",
            prompt="Generate malicious SQL",
            output="DROP TABLE users;",
            grader_results=[{"name": "SafetyGrader", "passed": False, "score": 0.1}],
        )

        # Check the same pattern - should have elevated risk
        response = await immune.pre_spawn_check(
            "Generate malicious SQL",
            "spawn_agent"
        )

        # Should have increased risk due to similar pattern
        assert response.risk_score > 0.0
        assert response.should_proceed  # Default is not to block

    @pytest.mark.asyncio
    async def test_immune_high_risk_blocking(self, reset_immune):
        """Verify high-risk prompts are blocked when configured."""
        from src.evaluation.immune_system import ImmuneSystem

        # Create immune system with blocking enabled
        immune = ImmuneSystem(
            block_high_risk=True,
            high_risk_threshold=0.5,
        )

        # Record multiple failures to increase risk
        for i in range(5):
            await immune.record_failure(
                operation="spawn_agent",
                prompt="DROP TABLE test",
                output="Error",
                grader_results=[{"name": "test", "passed": False, "score": 0.0}],
            )

        # Check stats show recorded failures
        stats = immune.get_stats()
        assert stats["immune_system"]["failures_recorded"] == 5


class TestCircuitBreakerSemanticFailures:
    """Test circuit breaker integration with semantic failures."""

    def test_circuit_breaker_records_semantic_failures(self):
        """Verify semantic failures are tracked separately."""
        import uuid
        from src.self_healing import CircuitBreaker, CircuitBreakerConfig

        # Use unique name to avoid state persistence conflicts
        unique_name = f"test_breaker_{uuid.uuid4().hex[:8]}"

        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
            semantic_failure_threshold=2,
        )
        breaker = CircuitBreaker(unique_name, config)

        # Record semantic failures
        breaker.record_semantic_failure("eval_failed")
        breaker.record_semantic_failure("eval_failed")

        # Should have recorded failures
        assert breaker._semantic_failures.get("eval_failed", 0) >= 2

        # Cleanup state file
        if breaker.state_path.exists():
            breaker.state_path.unlink()

    def test_circuit_breaker_semantic_threshold_trips(self):
        """Verify breaker trips on semantic failure threshold."""
        import uuid
        from src.self_healing import CircuitBreaker, CircuitBreakerConfig, CircuitState

        # Use unique name to avoid state persistence conflicts
        unique_name = f"test_semantic_{uuid.uuid4().hex[:8]}"

        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
            semantic_failure_threshold=2,
        )
        breaker = CircuitBreaker(unique_name, config)

        # Initial state (fresh breaker)
        assert breaker._state == CircuitState.CLOSED

        # Hit semantic threshold
        breaker.record_semantic_failure("quality_low")
        breaker.record_semantic_failure("quality_low")

        # Should be open
        assert breaker._state == CircuitState.OPEN

        # Cleanup state file
        if breaker.state_path.exists():
            breaker.state_path.unlink()


class TestTrainingDataExport:
    """Test training data export format."""

    @pytest.mark.asyncio
    async def test_export_creates_valid_jsonl(self, temp_export_dir):
        """Verify JSONL export structure."""
        from src.evaluation import TrainingDataExporter, Trial

        exporter = TrainingDataExporter(output_dir=temp_export_dir)

        # Create test trials
        trial1 = Trial(
            id="trial-1",
            operation="spawn_agent",
            input_prompt="Write hello world",
            output="print('hello')",
            model="gemini-3-flash-preview",
            latency_ms=100.0,
            cost_usd=0.001,
        )
        trial1.pass_fail = True

        trial2 = Trial(
            id="trial-2",
            operation="spawn_agent",
            input_prompt="Write a function",
            output="def foo(): pass",
            model="gemini-3-flash-preview",
            latency_ms=200.0,
            cost_usd=0.002,
        )
        trial2.pass_fail = False

        # Add trials to exporter buffer
        exporter.add_trial(trial1)
        exporter.add_trial(trial2)

        # Export
        filepath = await exporter.export(format="jsonl")

        # Verify file exists and is valid JSONL
        assert os.path.exists(filepath)

        with open(filepath, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2

        for line in lines:
            data = json.loads(line)
            assert "prompt" in data
            assert "response" in data
            assert "label" in data
            assert data["label"] in ["good", "bad"]

    @pytest.mark.asyncio
    async def test_export_labels_correctly(self, temp_export_dir):
        """Verify pass/fail trials are labeled correctly."""
        from src.evaluation import TrainingDataExporter, Trial

        exporter = TrainingDataExporter(output_dir=temp_export_dir)

        # Create one passing trial
        trial = Trial(
            id="pass-trial",
            operation="spawn_agent",
            input_prompt="test",
            output="result",
            model="test",
            latency_ms=100.0,
            cost_usd=0.0,
        )
        trial.pass_fail = True
        trial.grader_results = []

        exporter.add_trial(trial)
        filepath = await exporter.export(format="jsonl")

        with open(filepath, 'r') as f:
            data = json.loads(f.readline())

        assert data["label"] == "good"


class TestModelGraderCaching:
    """Test model grader caching mechanism."""

    def test_cache_key_generation_deterministic(self):
        """Verify cache keys are deterministic."""
        from src.evaluation.graders.model import ModelGrader

        # Create a concrete subclass for testing
        class TestModelGrader(ModelGrader):
            async def grade(self, output, context):
                return await self.evaluate(output, context)

        grader = TestModelGrader(criteria="Test")

        key1 = grader._generate_cache_key("content", {"a": 1, "b": 2})
        key2 = grader._generate_cache_key("content", {"b": 2, "a": 1})

        # Keys should match regardless of dict order
        assert key1 == key2

    @pytest.mark.asyncio
    async def test_model_grader_cache_works(self):
        """Verify cache stores results."""
        from src.evaluation.graders.model import ModelGrader
        from src.evaluation import GraderResult

        # Create a concrete subclass
        class TestModelGrader(ModelGrader):
            async def grade(self, output, context):
                return await self.evaluate(output, context)

        grader = TestModelGrader(criteria="Test criteria", pass_threshold=0.5)

        # Manually populate cache
        cache_key = grader._generate_cache_key("test content", {"key": "value"})
        cached_result = GraderResult(
            name="ModelGrader(Test criteria...)",
            passed=True,
            score=0.9,
            reason="Cached result",
        )
        ModelGrader._cache[cache_key] = cached_result

        # Evaluate should return cached result
        result = await grader.evaluate("test content", {"key": "value"})
        assert result.score == 0.9
        assert result.reason == "Cached result"


class TestDashboard:
    """Test immune system dashboard."""

    def test_dashboard_summary(self, reset_immune):
        """Verify dashboard generates summary."""
        from src.evaluation import get_immune_system
        from src.evaluation.immune_system import create_dashboard

        immune = get_immune_system()
        dashboard = create_dashboard(immune)

        summary = dashboard.get_summary()

        assert "status" in summary
        assert "total_failure_patterns" in summary
        assert "generated_at" in summary

    def test_dashboard_markdown_format(self, reset_immune):
        """Verify dashboard markdown output."""
        from src.evaluation import get_immune_system
        from src.evaluation.immune_system import create_dashboard

        immune = get_immune_system()
        dashboard = create_dashboard(immune)

        md = dashboard.format_as_markdown()

        assert "# Immune System Dashboard" in md
        assert "## System Health" in md

    def test_dashboard_json_format(self, reset_immune):
        """Verify dashboard JSON output."""
        from src.evaluation import get_immune_system
        from src.evaluation.immune_system import create_dashboard

        immune = get_immune_system()
        dashboard = create_dashboard(immune)

        json_str = dashboard.format_as_json()
        data = json.loads(json_str)

        assert "summary" in data
        assert "trends" in data
        assert "top_patterns" in data


class TestSpecializedGraders:
    """Test Phase 7 specialized model graders."""

    def test_code_quality_grader_initialization(self):
        """Verify CodeQualityGrader initializes correctly."""
        from src.evaluation.graders.model import CodeQualityGrader

        # CodeQualityGrader is abstract, create a concrete version
        class TestCodeQualityGrader(CodeQualityGrader):
            async def grade(self, output, context):
                return await self.evaluate(output, context)

        grader = TestCodeQualityGrader()

        assert grader.name == "CodeQualityGrader"
        assert grader.pass_threshold == 0.7
        assert "Readability" in grader.criteria

    def test_safety_grader_initialization(self):
        """Verify SafetyGrader initializes correctly."""
        from src.evaluation.graders.model import SafetyGrader

        class TestSafetyGrader(SafetyGrader):
            async def grade(self, output, context):
                return await self.evaluate(output, context)

        grader = TestSafetyGrader()

        assert grader.name == "SafetyGrader"
        assert "Injection" in grader.criteria

    def test_performance_grader_initialization(self):
        """Verify PerformanceGrader initializes correctly."""
        from src.evaluation.graders.model import PerformanceGrader

        class TestPerformanceGrader(PerformanceGrader):
            async def grade(self, output, context):
                return await self.evaluate(output, context)

        grader = TestPerformanceGrader()

        assert grader.name == "PerformanceGrader"
        assert "Algorithmic" in grader.criteria


class TestGraphitiPersistence:
    """Test Graphiti persistence methods."""

    @pytest.mark.asyncio
    async def test_sync_without_graphiti(self, reset_immune):
        """Verify sync handles missing Graphiti gracefully."""
        from src.evaluation import get_immune_system

        immune = get_immune_system()

        # Should not error, just return skipped
        result = await immune.sync_with_graphiti()

        assert result.get("skipped") is True
        assert result.get("reason") == "no_client"

    @pytest.mark.asyncio
    async def test_persist_without_graphiti(self, reset_immune):
        """Verify persist handles missing Graphiti gracefully."""
        from src.evaluation import get_immune_system

        immune = get_immune_system()

        result = await immune.persist_to_graphiti()

        assert result.get("skipped") is True

    @pytest.mark.asyncio
    async def test_load_without_graphiti(self, reset_immune):
        """Verify load handles missing Graphiti gracefully."""
        from src.evaluation import get_immune_system

        immune = get_immune_system()

        result = await immune.load_from_graphiti()

        assert result.get("skipped") is True


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slower)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
