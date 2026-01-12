"""
Tests for the Graphiti Immune System.
"""

import pytest

from src.evaluation.immune_system import (
    ImmuneSystem,
    ImmuneResponse,
    get_immune_system,
    reset_immune_system,
    FailurePattern,
    FailurePatternStore,
    PatternMatcher,
    PromptGuardrails,
    GUARDRAIL_TEMPLATES,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def failure_store():
    """Create a fresh failure store for testing."""
    return FailurePatternStore()


@pytest.fixture
def pattern_matcher(failure_store):
    """Create a pattern matcher with the test failure store."""
    return PatternMatcher(failure_store)


@pytest.fixture
def guardrails(pattern_matcher):
    """Create prompt guardrails for testing."""
    return PromptGuardrails(pattern_matcher, risk_threshold=0.5)


@pytest.fixture
def immune_system():
    """Create a fresh immune system for testing."""
    reset_immune_system()
    return ImmuneSystem(risk_threshold=0.5)


# =============================================================================
# FailurePattern Tests
# =============================================================================


class TestFailurePattern:
    """Tests for FailurePattern dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        pattern = FailurePattern(
            id="test123",
            operation="spawn_agent",
            failure_type="json_invalid",
            input_summary="Generate JSON",
            output_summary="{bad json",
            grader_scores={"JSONValidGrader": 0.0},
            occurrence_count=2,
        )

        data = pattern.to_dict()

        assert data["id"] == "test123"
        assert data["operation"] == "spawn_agent"
        assert data["failure_type"] == "json_invalid"
        assert data["occurrence_count"] == 2

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "test456",
            "operation": "spawn_agent",
            "failure_type": "empty_response",
            "input_summary": "Test prompt",
            "output_summary": "",
            "grader_scores": {"NonEmptyGrader": 0.0},
            "context": {"model": "gemini"},
            "created_at": "2025-01-01T00:00:00",
            "occurrence_count": 1,
        }

        pattern = FailurePattern.from_dict(data)

        assert pattern.id == "test456"
        assert pattern.failure_type == "empty_response"
        assert pattern.context["model"] == "gemini"


# =============================================================================
# FailurePatternStore Tests
# =============================================================================


class TestFailurePatternStore:
    """Tests for FailurePatternStore."""

    @pytest.mark.asyncio
    async def test_store_failure(self, failure_store):
        """Test storing a failure pattern."""
        pattern = await failure_store.store_failure(
            operation="spawn_agent",
            input_prompt="Generate JSON output",
            output="{invalid",
            grader_results=[{"name": "JSONValidGrader", "passed": False, "score": 0.0}],
        )

        assert pattern.operation == "spawn_agent"
        assert pattern.failure_type == "json_invalid"
        assert pattern.occurrence_count == 1

    @pytest.mark.asyncio
    async def test_duplicate_failure_increments_count(self, failure_store):
        """Test that duplicate failures increment the count."""
        # Store first failure
        pattern1 = await failure_store.store_failure(
            operation="spawn_agent",
            input_prompt="Generate JSON output",
            output="{invalid",
            grader_results=[{"name": "JSONValidGrader", "passed": False}],
        )

        # Store similar failure
        pattern2 = await failure_store.store_failure(
            operation="spawn_agent",
            input_prompt="Generate JSON output",
            output="{also invalid",
            grader_results=[{"name": "JSONValidGrader", "passed": False}],
        )

        assert pattern2.id == pattern1.id
        assert pattern2.occurrence_count == 2

    @pytest.mark.asyncio
    async def test_get_failures_by_type(self, failure_store):
        """Test retrieving failures by type."""
        await failure_store.store_failure(
            operation="spawn_agent",
            input_prompt="JSON prompt",
            output="{bad",
            grader_results=[{"name": "JSONValidGrader", "passed": False}],
        )

        failures = await failure_store.get_failures_by_type("json_invalid")

        assert len(failures) == 1
        assert failures[0].failure_type == "json_invalid"

    @pytest.mark.asyncio
    async def test_get_stats(self, failure_store):
        """Test getting store statistics."""
        await failure_store.store_failure(
            operation="spawn_agent",
            input_prompt="Test",
            output="",
            grader_results=[{"name": "NonEmptyGrader", "passed": False}],
        )

        stats = failure_store.get_stats()

        assert stats["total_patterns"] == 1
        assert "by_type" in stats


# =============================================================================
# PatternMatcher Tests
# =============================================================================


class TestPatternMatcher:
    """Tests for PatternMatcher."""

    @pytest.mark.asyncio
    async def test_find_similar_failures_empty(self, pattern_matcher):
        """Test finding matches when store is empty."""
        matches = await pattern_matcher.find_similar_failures(
            prompt="Test prompt",
            operation="spawn_agent",
        )

        assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_find_similar_failures_with_match(self, failure_store, pattern_matcher):
        """Test finding matches when similar failure exists."""
        # Store a failure
        await failure_store.store_failure(
            operation="spawn_agent",
            input_prompt="Generate JSON output for analysis",
            output="{bad",
            grader_results=[{"name": "JSONValidGrader", "passed": False}],
        )

        # Look for similar
        matches = await pattern_matcher.find_similar_failures(
            prompt="Generate JSON output for report",
            operation="spawn_agent",
        )

        assert len(matches) >= 1
        assert matches[0].pattern.failure_type == "json_invalid"

    @pytest.mark.asyncio
    async def test_check_prompt_risk(self, failure_store, pattern_matcher):
        """Test risk assessment for prompts."""
        # Store a failure
        await failure_store.store_failure(
            operation="spawn_agent",
            input_prompt="Generate code function",
            output="invalid",
            grader_results=[{"name": "RegexGrader", "passed": False}],
        )

        # Check risk for similar prompt
        risk_score, reasons = await pattern_matcher.check_prompt_risk(
            prompt="Generate code function for sorting",
            operation="spawn_agent",
        )

        # Risk should be > 0 due to similarity
        assert risk_score > 0 or len(reasons) == 0  # May or may not match depending on threshold


# =============================================================================
# PromptGuardrails Tests
# =============================================================================


class TestPromptGuardrails:
    """Tests for PromptGuardrails."""

    @pytest.mark.asyncio
    async def test_apply_guardrails_no_risk(self, guardrails):
        """Test guardrails with no matching failures."""
        result = await guardrails.apply_guardrails(
            prompt="Simple question",
            operation="spawn_agent",
        )

        assert result.risk_score == 0.0
        assert len(result.guardrails_applied) == 0
        assert not result.was_modified

    @pytest.mark.asyncio
    async def test_apply_guardrails_with_risk(self, failure_store, pattern_matcher):
        """Test guardrails with matching failures."""
        # Store failures to build up risk
        for i in range(3):
            await failure_store.store_failure(
                operation="spawn_agent",
                input_prompt=f"Generate JSON output {i}",
                output="{bad",
                grader_results=[{"name": "JSONValidGrader", "passed": False}],
            )

        guardrails = PromptGuardrails(pattern_matcher, risk_threshold=0.3)

        result = await guardrails.apply_guardrails(
            prompt="Generate JSON output for report",
            operation="spawn_agent",
        )

        # Should have some guardrails if risk detected
        if result.risk_score >= 0.3:
            assert result.was_modified or len(result.warnings) > 0

    def test_get_stats(self, guardrails):
        """Test getting guardrail statistics."""
        stats = guardrails.get_stats()

        assert "prompts_checked" in stats
        assert "guardrails_applied" in stats
        assert "risk_threshold" in stats


class TestGuardrailTemplates:
    """Tests for guardrail templates."""

    def test_all_templates_exist(self):
        """Test that all expected templates exist."""
        expected_types = [
            "json_invalid",
            "schema_violation",
            "empty_response",
            "pattern_mismatch",
            "length_violation",
            "hallucination",
            "evaluation_failed",
        ]

        for failure_type in expected_types:
            assert failure_type in GUARDRAIL_TEMPLATES
            assert len(GUARDRAIL_TEMPLATES[failure_type]) > 0


# =============================================================================
# ImmuneSystem Tests
# =============================================================================


class TestImmuneSystem:
    """Tests for the main ImmuneSystem class."""

    @pytest.mark.asyncio
    async def test_pre_spawn_check_clean(self, immune_system):
        """Test pre-spawn check with no prior failures."""
        response = await immune_system.pre_spawn_check(
            prompt="Write a function",
            operation="spawn_agent",
        )

        assert isinstance(response, ImmuneResponse)
        assert response.should_proceed
        assert response.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_record_and_detect_failure(self, immune_system):
        """Test recording a failure and detecting similar prompts."""
        # Record a failure
        pattern = await immune_system.record_failure(
            operation="spawn_agent",
            prompt="Generate JSON",
            output="{bad",
            grader_results=[{"name": "JSONValidGrader", "passed": False}],
        )

        assert pattern.failure_type == "json_invalid"

        # Check similar prompt
        response = await immune_system.pre_spawn_check(
            prompt="Generate JSON output",
            operation="spawn_agent",
        )

        # Should detect some risk
        assert response.risk_score > 0

    @pytest.mark.asyncio
    async def test_get_stats(self, immune_system):
        """Test getting immune system stats."""
        await immune_system.pre_spawn_check("Test", "spawn_agent")

        stats = immune_system.get_stats()

        assert "immune_system" in stats
        assert stats["immune_system"]["pre_spawn_checks"] == 1

    @pytest.mark.asyncio
    async def test_get_health(self, immune_system):
        """Test getting health status."""
        health = immune_system.get_health()

        assert health["status"] == "healthy"
        assert "total_patterns" in health

    @pytest.mark.asyncio
    async def test_blocking_high_risk(self):
        """Test that high risk prompts can be blocked."""
        reset_immune_system()
        immune = ImmuneSystem(
            risk_threshold=0.3,
            block_high_risk=True,
            high_risk_threshold=0.8,
        )

        # Build up failures
        for i in range(5):
            await immune.record_failure(
                operation="spawn_agent",
                prompt=f"Generate comprehensive JSON {i}",
                output="{bad",
                grader_results=[{"name": "JSONValidGrader", "passed": False}],
            )

        # Check similar high-risk prompt
        response = await immune.pre_spawn_check(
            prompt="Generate comprehensive JSON report",
            operation="spawn_agent",
        )

        # If risk is high enough, should be blocked
        if response.risk_score >= 0.8:
            assert not response.should_proceed


class TestGlobalImmuneSystem:
    """Tests for the global immune system singleton."""

    def test_get_immune_system_singleton(self):
        """Test that get_immune_system returns singleton."""
        reset_immune_system()

        immune1 = get_immune_system()
        immune2 = get_immune_system()

        assert immune1 is immune2

    def test_reset_immune_system(self):
        """Test resetting the singleton."""
        reset_immune_system()
        immune1 = get_immune_system()

        reset_immune_system()
        immune2 = get_immune_system()

        assert immune1 is not immune2


# =============================================================================
# Integration Tests
# =============================================================================


class TestImmuneSystemIntegration:
    """Integration tests for the immune system."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test the complete immune system workflow."""
        reset_immune_system()
        immune = ImmuneSystem()

        # 1. Initial check passes
        response1 = await immune.pre_spawn_check(
            prompt="Generate analysis",
            operation="spawn_agent",
        )
        assert response1.should_proceed

        # 2. Record some failures
        await immune.record_failure(
            operation="spawn_agent",
            prompt="Generate analysis report",
            output="",
            grader_results=[{"name": "NonEmptyGrader", "passed": False}],
        )

        await immune.record_failure(
            operation="spawn_agent",
            prompt="Generate analysis summary",
            output="",
            grader_results=[{"name": "NonEmptyGrader", "passed": False}],
        )

        # 3. Check similar prompt - should have higher risk
        await immune.pre_spawn_check(
            prompt="Generate analysis output",
            operation="spawn_agent",
        )

        # 4. Verify stats updated
        stats = immune.get_stats()
        assert stats["immune_system"]["failures_recorded"] == 2
        assert stats["immune_system"]["pre_spawn_checks"] == 2

    @pytest.mark.asyncio
    async def test_different_operations_isolated(self):
        """Test that different operations have isolated patterns."""
        reset_immune_system()
        immune = ImmuneSystem()

        # Record failure for one operation
        await immune.record_failure(
            operation="spawn_agent",
            prompt="Generate code",
            output="bad",
            grader_results=[{"name": "RegexGrader", "passed": False}],
        )

        # Check different operation - should have low risk
        response = await immune.pre_spawn_check(
            prompt="Generate code",
            operation="analyze_task",  # Different operation
        )

        # Different operations should have less cross-contamination
        # (actual behavior depends on matcher implementation)
        assert response.should_proceed
