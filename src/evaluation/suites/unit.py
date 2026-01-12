"""
Unit Evaluation Suite for task-orchestrator.

This module provides evaluation tools for testing individual agent responses
against quality criteria. It validates code generation, JSON responses,
and general explanations using the grader pipeline.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..trial import Trial, GraderResult
from ..graders.base import Grader, GraderPipeline
from ..graders.code import (
    NonEmptyGrader,
    JSONValidGrader,
    JSONSchemaGrader,
    RegexGrader,
    LengthGrader,
    ContainsGrader,
)

logger = logging.getLogger(__name__)


class UnitEvalSuite:
    """
    Evaluation suite for unit testing individual agent responses.

    This suite focuses on single-turn interactions, validating the structural
    integrity, syntax, and schema compliance of agent responses.

    Attributes:
        default_model (str): Default model to use for context tracking.
    """

    def __init__(self, default_model: str = "gemini-3-flash-preview"):
        """
        Initialize the UnitEvalSuite.

        Args:
            default_model: The default LLM model identifier for context.
        """
        self.default_model = default_model
        logger.info(f"Initialized UnitEvalSuite with model: {self.default_model}")

    def get_default_pipeline(self) -> GraderPipeline:
        """
        Create a default grader pipeline with basic sanity checks.

        Returns:
            GraderPipeline: Pipeline with NonEmpty and Length graders.
        """
        return GraderPipeline([
            NonEmptyGrader(),
            LengthGrader(min_length=10, max_length=100000),
        ])

    def get_code_pipeline(self) -> GraderPipeline:
        """
        Create a grader pipeline for code generation validation.

        Returns:
            GraderPipeline: Pipeline checking for code patterns.
        """
        return GraderPipeline([
            NonEmptyGrader(),
            LengthGrader(min_length=20),
            RegexGrader(pattern=r"(def |class |function |const |let |var )"),
        ])

    def get_json_pipeline(self, schema: Optional[Dict[str, Any]] = None) -> GraderPipeline:
        """
        Create a grader pipeline for JSON response validation.

        Args:
            schema: Optional JSON schema for validation.

        Returns:
            GraderPipeline: Pipeline with JSON validation graders.
        """
        graders: List[Grader] = [
            NonEmptyGrader(),
            JSONValidGrader(),
        ]
        if schema:
            graders.append(JSONSchemaGrader(schema=schema))
        return GraderPipeline(graders)

    async def run_eval(
        self,
        output: str,
        prompt: str,
        expected_patterns: Optional[List[str]] = None,
        expected_json_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Trial:
        """
        Execute a single evaluation on provided output.

        Args:
            output: The agent's response to evaluate.
            prompt: The original prompt (for context).
            expected_patterns: Optional regex patterns expected in output.
            expected_json_schema: Optional JSON schema to validate against.
            model: Optional model identifier override.

        Returns:
            Trial: The completed trial with grader results.
        """
        start_time = time.time()
        trial = Trial(
            operation="unit_eval",
            input_prompt=prompt,
            model=model or self.default_model,
        )
        trial.output = output

        # Build pipeline based on requirements
        pipeline = self._build_pipeline(expected_patterns, expected_json_schema)

        # Run evaluation
        context = {"prompt": prompt}
        grader_results = await pipeline.run(output, context)

        for result in grader_results:
            trial.add_grader_result(result)

        trial.latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Trial completed. Passed: {trial.pass_fail}")

        return trial

    def _build_pipeline(
        self,
        expected_patterns: Optional[List[str]],
        expected_json_schema: Optional[Dict[str, Any]],
    ) -> GraderPipeline:
        """
        Build a custom grader pipeline based on expectations.

        Args:
            expected_patterns: Regex patterns to match.
            expected_json_schema: JSON schema for validation.

        Returns:
            GraderPipeline: Configured pipeline.
        """
        graders: List[Grader] = [NonEmptyGrader()]

        if expected_json_schema:
            graders.append(JSONValidGrader())
            graders.append(JSONSchemaGrader(schema=expected_json_schema))

        if expected_patterns:
            for i, pattern in enumerate(expected_patterns):
                graders.append(RegexGrader(
                    pattern=pattern,
                    name=f"PatternMatch_{i}"
                ))

        # Default length check if no specific validators
        if len(graders) == 1:
            graders.append(LengthGrader(min_length=10))

        return GraderPipeline(graders)


# =============================================================================
# Preset Evaluation Functions
# =============================================================================

async def eval_code_generation(output: str, prompt: str) -> Trial:
    """
    Evaluate if output contains valid code with function or class definitions.

    Args:
        output: The agent's code response.
        prompt: The original prompt requesting code.

    Returns:
        Trial: Evaluation result checking for code patterns.
    """
    suite = UnitEvalSuite()
    return await suite.run_eval(
        output=output,
        prompt=prompt,
        expected_patterns=[r"(def |class |function |const |let |var )"],
    )


async def eval_json_response(output: str, prompt: str, schema: Dict[str, Any]) -> Trial:
    """
    Evaluate if output is valid JSON matching a schema.

    Args:
        output: The agent's JSON response.
        prompt: The original prompt requesting JSON.
        schema: JSON schema to validate against.

    Returns:
        Trial: Evaluation result validating JSON structure.
    """
    suite = UnitEvalSuite()
    return await suite.run_eval(
        output=output,
        prompt=prompt,
        expected_json_schema=schema,
    )


async def eval_explanation(output: str, prompt: str, min_length: int = 100) -> Trial:
    """
    Evaluate if output is a comprehensive explanation.

    Args:
        output: The agent's explanation response.
        prompt: The original prompt requesting explanation.
        min_length: Minimum expected length.

    Returns:
        Trial: Evaluation result checking content quality.
    """
    suite = UnitEvalSuite()
    trial = Trial(
        operation="unit_eval_explanation",
        input_prompt=prompt,
        model=suite.default_model,
    )
    trial.output = output

    pipeline = GraderPipeline([
        NonEmptyGrader(),
        LengthGrader(min_length=min_length, max_length=50000),
    ])

    results = await pipeline.run(output, {"prompt": prompt})
    for result in results:
        trial.add_grader_result(result)

    return trial


# Export suite class and preset functions
__all__ = [
    "UnitEvalSuite",
    "eval_code_generation",
    "eval_json_response",
    "eval_explanation",
]
