"""
tests/test_evaluation.py

Comprehensive test suite for the task-orchestrator evaluation system.
Covers:
- Trial and GraderResult data structures
- All code-based graders (JSON, Regex, Length, Content)
- Grader pipelines
- Training data export
- Langfuse integration mocking
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import evaluation modules
from src.evaluation.trial import Trial, GraderResult
from src.evaluation.graders.base import Grader, GraderPipeline
from src.evaluation.graders.code import (
    NonEmptyGrader,
    JSONValidGrader,
    JSONSchemaGrader,
    RegexGrader,
    LengthGrader,
    ContainsGrader,
    NotContainsGrader,
)
from src.evaluation.export import TrainingDataExporter


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_schema() -> Dict[str, Any]:
    """Returns a simple JSON schema for testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age"]
    }


@pytest.fixture
def sample_trial() -> Trial:
    """Returns a pre-populated Trial object."""
    trial = Trial(
        operation="spawn_agent",
        input_prompt="Write a Python function that returns 'hello'",
        model="gemini-3-pro-preview"
    )
    trial.output = "def hello():\n    return 'hello'"
    return trial


# -----------------------------------------------------------------------------
# Test Data Structures (Trial & GraderResult)
# -----------------------------------------------------------------------------

class TestTrial:
    def test_trial_initialization(self):
        """Test that a trial is initialized with correct defaults."""
        trial = Trial(operation="test_op", input_prompt="Do X")
        assert trial.operation == "test_op"
        assert trial.input_prompt == "Do X"
        assert trial.id is not None
        assert trial.created_at is not None
        assert trial.pass_fail is True  # Defaults to True until a failure occurs
        assert trial.grader_results == []

    def test_trial_add_grader_result_passing(self):
        """Test adding a passing result keeps trial passing."""
        trial = Trial(operation="test", input_prompt="test")
        result = GraderResult(name="check1", passed=True, score=1.0, reason="ok")
        trial.add_grader_result(result)
        assert trial.pass_fail is True
        assert len(trial.grader_results) == 1

    def test_trial_add_grader_result_failing(self):
        """Test adding a failed result flips the trial pass_fail status."""
        trial = Trial(operation="test", input_prompt="test")

        # Add passing result
        p_result = GraderResult(name="check1", passed=True, score=1.0, reason="ok")
        trial.add_grader_result(p_result)
        assert trial.pass_fail is True

        # Add failing result
        f_result = GraderResult(name="check2", passed=False, score=0.0, reason="Failed check")
        trial.add_grader_result(f_result)
        assert trial.pass_fail is False

        # Add another passing result (should remain False)
        trial.add_grader_result(p_result)
        assert trial.pass_fail is False

    def test_trial_serialization(self):
        """Test serialization to dictionary."""
        trial = Trial(operation="test", input_prompt="test prompt")
        trial.output = "some output"
        trial.model = "test-model"
        trial.metadata = {"custom_key": "custom_value"}

        data = trial.to_dict()
        assert data["operation"] == "test"
        assert "id" in data
        assert "grader_results" in data
        assert data["model"] == "test-model"


class TestGraderResult:
    def test_grader_result_creation(self):
        result = GraderResult(name="Test", passed=True, score=0.9, reason="Good")
        assert result.passed is True
        assert result.score == 0.9
        assert result.reason == "Good"

    def test_grader_result_to_dict(self):
        result = GraderResult(name="Test", passed=True, score=0.5, reason="partial")
        d = result.to_dict()
        assert d["name"] == "Test"
        assert d["score"] == 0.5
        assert d["passed"] is True


# -----------------------------------------------------------------------------
# Test Code Graders
# -----------------------------------------------------------------------------

class TestNonEmptyGrader:
    @pytest.mark.asyncio
    async def test_non_empty_passes(self):
        grader = NonEmptyGrader()
        result = await grader.grade("hello world", {})
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_non_empty_fails_empty_string(self):
        grader = NonEmptyGrader()
        result = await grader.grade("", {})
        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_non_empty_fails_none(self):
        grader = NonEmptyGrader()
        result = await grader.grade(None, {})
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_non_empty_fails_whitespace(self):
        grader = NonEmptyGrader()
        result = await grader.grade("   \n\t  ", {})
        assert result.passed is False


class TestJSONValidGrader:
    @pytest.mark.asyncio
    async def test_valid_json(self):
        grader = JSONValidGrader()
        result = await grader.grade('{"key": "value"}', {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        grader = JSONValidGrader()
        result = await grader.grade('not json', {})
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_json_in_markdown_block(self):
        """Ensure the grader handles LLM markdown output (```json ... ```)."""
        grader = JSONValidGrader()
        llm_output = "Here is the code:\n```json\n{\"a\": 1}\n```"
        result = await grader.grade(llm_output, {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_dict_input(self):
        """Already parsed dict should pass."""
        grader = JSONValidGrader()
        result = await grader.grade({"key": "value"}, {})
        assert result.passed is True


class TestJSONSchemaGrader:
    @pytest.mark.asyncio
    async def test_schema_validation_pass(self, sample_schema):
        grader = JSONSchemaGrader(schema=sample_schema)
        valid_data = '{"name": "Bob", "age": 25, "tags": ["admin"]}'
        result = await grader.grade(valid_data, {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_schema_validation_fail_missing_required(self, sample_schema):
        grader = JSONSchemaGrader(schema=sample_schema)
        # Missing required 'age'
        invalid_data = '{"name": "Bob"}'
        result = await grader.grade(invalid_data, {})
        assert result.passed is False
        assert "age" in result.reason.lower() or "required" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_schema_validation_fail_wrong_type(self, sample_schema):
        grader = JSONSchemaGrader(schema=sample_schema)
        # Age should be int, provided string
        invalid_data = '{"name": "Bob", "age": "twenty"}'
        result = await grader.grade(invalid_data, {})
        assert result.passed is False


class TestRegexGrader:
    @pytest.mark.asyncio
    async def test_regex_matches(self):
        grader = RegexGrader(pattern=r"def \w+\(")
        result = await grader.grade("def hello():", {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_regex_no_match(self):
        grader = RegexGrader(pattern=r"def \w+\(")
        result = await grader.grade("class Foo:", {})
        assert result.passed is False


class TestLengthGrader:
    @pytest.mark.asyncio
    async def test_length_within_bounds(self):
        grader = LengthGrader(min_length=5, max_length=100)
        result = await grader.grade("hello world", {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_length_too_short(self):
        grader = LengthGrader(min_length=50)
        result = await grader.grade("hi", {})
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_length_too_long(self):
        grader = LengthGrader(max_length=5)
        result = await grader.grade("hello world", {})
        assert result.passed is False


class TestContainsGrader:
    @pytest.mark.asyncio
    async def test_contains_all_strings(self):
        grader = ContainsGrader(required_strings=["def", "return"])
        result = await grader.grade("def foo():\n    return 42", {})
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_contains_partial(self):
        grader = ContainsGrader(required_strings=["def", "class", "return"])
        result = await grader.grade("def foo():\n    return 42", {})
        assert result.passed is False
        assert result.score == pytest.approx(2/3)

    @pytest.mark.asyncio
    async def test_contains_case_insensitive(self):
        grader = ContainsGrader(required_strings=["ERROR"], case_sensitive=False)
        result = await grader.grade("No error found", {})
        assert result.passed is True


class TestNotContainsGrader:
    @pytest.mark.asyncio
    async def test_not_contains_pass(self):
        grader = NotContainsGrader(forbidden_strings=["password", "secret"])
        result = await grader.grade("My data is safe", {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_not_contains_fail(self):
        grader = NotContainsGrader(forbidden_strings=["fail", "error"])
        result = await grader.grade("This task failed", {})
        assert result.passed is False


# -----------------------------------------------------------------------------
# Test Pipeline
# -----------------------------------------------------------------------------

class TestGraderPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_all_pass(self):
        pipeline = GraderPipeline([
            NonEmptyGrader(),
            LengthGrader(min_length=1, max_length=1000),
        ])
        results = await pipeline.run("hello world", {})
        assert len(results) == 2
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_pipeline_partial_fail(self):
        pipeline = GraderPipeline([
            NonEmptyGrader(),
            LengthGrader(min_length=100),  # Will fail
        ])
        results = await pipeline.run("hi", {})
        assert results[0].passed is True
        assert results[1].passed is False

    @pytest.mark.asyncio
    async def test_pipeline_add_method(self):
        pipeline = GraderPipeline()
        pipeline.add(NonEmptyGrader()).add(LengthGrader(min_length=1))
        assert len(pipeline.graders) == 2

    @pytest.mark.asyncio
    async def test_pipeline_handles_grader_exception(self):
        """Test that pipeline catches grader exceptions gracefully."""
        class FailingGrader(Grader):
            async def grade(self, output, context):
                raise ValueError("Intentional failure")

        pipeline = GraderPipeline([
            NonEmptyGrader(),
            FailingGrader(name="FailGrader"),
        ])
        results = await pipeline.run("hello", {})
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False
        assert "Grader error" in results[1].reason


# -----------------------------------------------------------------------------
# Test Exporting
# -----------------------------------------------------------------------------

class TestTrainingDataExporter:
    @pytest.mark.asyncio
    async def test_export_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TrainingDataExporter(output_dir=tmpdir)
            trial = Trial(operation="test", input_prompt="test prompt")
            trial.output = "test output"
            trial.pass_fail = True
            exporter.add_trial(trial)

            filepath = await exporter.export(format="jsonl")
            assert filepath.exists()

            with open(filepath) as f:
                data = json.loads(f.readline())
                assert data["label"] == "good"
                assert data["prompt"] == "test prompt"

    @pytest.mark.asyncio
    async def test_export_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TrainingDataExporter(output_dir=tmpdir)
            trial = Trial(operation="test", input_prompt="prompt")
            trial.output = "output"
            trial.pass_fail = False
            exporter.add_trial(trial)

            filepath = await exporter.export(format="json")
            assert filepath.exists()

            with open(filepath) as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["label"] == "bad"

    @pytest.mark.asyncio
    async def test_export_clears_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TrainingDataExporter(output_dir=tmpdir)
            trial = Trial(operation="test", input_prompt="test")
            exporter.add_trial(trial)
            assert exporter.buffer_size() == 1

            await exporter.export(format="jsonl")
            assert exporter.buffer_size() == 0

    @pytest.mark.asyncio
    async def test_export_empty_buffer_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TrainingDataExporter(output_dir=tmpdir)
            with pytest.raises(ValueError, match="No trials to export"):
                await exporter.export()


# -----------------------------------------------------------------------------
# Test Integration (Mocking External Services)
# -----------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.asyncio
    async def test_score_trial_with_mock_tracer(self, sample_trial):
        """Test that trials correctly call the tracer's score method."""
        from src.evaluation import integration

        # Add results to trial
        sample_trial.add_grader_result(
            GraderResult(name="check1", passed=True, score=1.0, reason="ok")
        )
        sample_trial.add_grader_result(
            GraderResult(name="check2", passed=True, score=0.8, reason="good")
        )

        # Mock the tracer
        mock_tracer = Mock()
        mock_tracer.enabled = True

        with patch.object(integration, 'get_tracer', return_value=mock_tracer):
            await integration.score_trial(sample_trial)

        # Should have called score 3 times (2 graders + 1 overall)
        assert mock_tracer.score.call_count == 3

    @pytest.mark.asyncio
    async def test_score_trial_tracer_disabled(self, sample_trial):
        """Test that no-op when tracer is disabled."""
        from src.evaluation import integration

        mock_tracer = Mock()
        mock_tracer.enabled = False

        with patch.object(integration, 'get_tracer', return_value=mock_tracer):
            await integration.score_trial(sample_trial)

        mock_tracer.score.assert_not_called()
