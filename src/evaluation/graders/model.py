"""
Model-Based Graders (LLM-as-Judge) for Task Orchestrator.

This module provides graders that use LLMs to evaluate output quality
based on natural language criteria.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from .base import Grader, GraderResult

logger = logging.getLogger(__name__)


class ModelGrader(Grader):
    """
    A grader that uses an LLM to evaluate output quality.

    Uses Gemini 2.0 Flash for fast, cost-effective evaluation.
    Includes caching to avoid repeated LLM calls.
    """

    # In-memory cache: {hash_key: GraderResult}
    _cache: Dict[str, GraderResult] = {}

    def __init__(
        self,
        criteria: str,
        name: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        timeout_seconds: int = 15,
        pass_threshold: float = 0.7,
    ):
        """
        Initialize the ModelGrader.

        Args:
            criteria: Description of what constitutes a good answer.
            name: Name for this grader instance.
            model_name: The Gemini model to use.
            temperature: LLM temperature (0.0 for deterministic).
            timeout_seconds: Max time to wait for LLM response.
            pass_threshold: Score threshold for passing (0.0-1.0).
        """
        super().__init__(name=name or f"ModelGrader({criteria[:30]}...)")
        self.criteria = criteria
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.pass_threshold = pass_threshold
        self._client = None

    def _get_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                self._client = genai
            except ImportError:
                logger.warning("google-generativeai not installed")
                self._client = None
        return self._client

    def _generate_cache_key(self, content: str, context: Optional[Dict]) -> str:
        """Generate a unique hash for caching."""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        raw_key = f"{self.criteria}::{content[:500]}::{context_str}::{self.model_name}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def _clean_json_response(self, text: str) -> str:
        """Extract JSON from LLM response, removing markdown."""
        text = text.strip()
        # Remove markdown code blocks
        match = re.search(r"```(?:json)?\s*(.*)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with timeout handling."""
        genai = self._get_client()
        if not genai:
            raise ImportError("Google Generative AI SDK not available")

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )

        loop = asyncio.get_running_loop()

        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: model.generate_content(prompt)),
                timeout=self.timeout_seconds,
            )
            return response.text
        except asyncio.TimeoutError:
            logger.error(f"ModelGrader timeout after {self.timeout_seconds}s")
            raise

    async def grade(self, output: Any, context: Dict[str, Any]) -> GraderResult:
        """
        Grade output using an LLM (implements Grader ABC).

        Args:
            output: The output to evaluate.
            context: Additional context (task_description, expected_output, etc.).

        Returns:
            GraderResult with score (0.0-1.0) and reasoning.
        """
        return await self.evaluate(str(output), context)

    async def evaluate(self, content: str, context: Optional[Dict] = None) -> GraderResult:
        """
        Evaluate content using an LLM.

        Args:
            content: The output to evaluate.
            context: Additional context (task_description, expected_output, etc.).

        Returns:
            GraderResult with score (0.0-1.0) and reasoning.
        """
        # Check cache first
        cache_key = self._generate_cache_key(content, context)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for ModelGrader: {cache_key[:16]}")
            return self._cache[cache_key]

        # Build evaluation prompt
        task_context = context.get("task_description", "N/A") if context else "N/A"
        expected = context.get("expected_output", "N/A") if context else "N/A"
        prompt_text = context.get("prompt", "N/A") if context else "N/A"

        system_prompt = f"""You are an expert evaluator for an AI system.
Your task is to grade the provided Output based on specific Criteria.

You must return a JSON object with exactly two keys:
1. "score": A float between 0.0 and 1.0 (0.0 is total failure, 1.0 is perfect)
2. "reasoning": A concise explanation of the score (1-2 sentences)

Criteria: {self.criteria}

Original Prompt: {prompt_text[:500]}
Task Context: {task_context}
Reference/Expected (Optional): {expected}

Output to Evaluate:
{content[:2000]}

Respond ONLY with the JSON. No other text."""

        try:
            raw_response = await self._call_llm(system_prompt)
            cleaned_json = self._clean_json_response(raw_response)
            parsed = json.loads(cleaned_json)

            score = float(parsed.get("score", 0.0))
            reasoning = parsed.get("reasoning", "No reasoning provided.")

            # Clamp score to valid range
            score = max(0.0, min(1.0, score))

            result = GraderResult(
                name=self.name,
                passed=score >= self.pass_threshold,
                score=score,
                reason=reasoning,
            )

            # Cache the result
            self._cache[cache_key] = result
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            return GraderResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason="Evaluation failed: Invalid JSON from grader model",
            )
        except asyncio.TimeoutError:
            return GraderResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason="Evaluation failed: Grader timed out",
            )
        except Exception as e:
            logger.exception("Unexpected error in ModelGrader")
            return GraderResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason=f"Evaluation failed: {str(e)}",
            )


class RelevanceGrader(ModelGrader):
    """Grader that checks if output is relevant to the request."""

    def __init__(self, pass_threshold: float = 0.7):
        super().__init__(
            name="RelevanceGrader",
            criteria="Is the output directly relevant to the user request? Does it address the question or task?",
            pass_threshold=pass_threshold,
        )


class CompletenessGrader(ModelGrader):
    """Grader that checks if output covers all parts of the request."""

    def __init__(self, pass_threshold: float = 0.7):
        super().__init__(
            name="CompletenessGrader",
            criteria="Does the output address all parts of the task? Are there missing components or incomplete sections?",
            pass_threshold=pass_threshold,
        )


class AccuracyGrader(ModelGrader):
    """Grader that checks for factual accuracy and correctness."""

    def __init__(self, pass_threshold: float = 0.7):
        super().__init__(
            name="AccuracyGrader",
            criteria="Is the output factually accurate? Does the code/logic appear correct? Are there obvious errors?",
            pass_threshold=pass_threshold,
        )


class FormatGrader(ModelGrader):
    """Grader that checks if output follows expected format."""

    def __init__(self, expected_format: str = "well-structured text or code", pass_threshold: float = 0.7):
        super().__init__(
            name="FormatGrader",
            criteria=f"Does the output follow the expected format ({expected_format})? Is it properly structured?",
            pass_threshold=pass_threshold,
        )


class CodeQualityGrader(ModelGrader):
    """
    Evaluates code for adherence to best practices, readability, and maintainability.

    Inspects source code for:
    - Naming conventions and readability
    - Modular design and DRY principles
    - Presence of documentation and comments
    - Logical complexity
    """

    def __init__(self, pass_threshold: float = 0.7):
        criteria = (
            "Evaluate the provided code based on software engineering best practices:\n"
            "1. Readability: Are variable/function names descriptive? Is the formatting consistent?\n"
            "2. Maintainability: Is the code modular? Does it follow DRY (Don't Repeat Yourself) principles?\n"
            "3. Conventions: Does it follow language-specific idioms (e.g., PEP 8 for Python)?\n"
            "4. Documentation: Are complex logic blocks explained? Are docstrings present?\n"
            "5. Error Handling: Are exceptions caught and handled appropriately, rather than suppressed?"
        )
        super().__init__(
            name="CodeQualityGrader",
            criteria=criteria,
            pass_threshold=pass_threshold,
        )


class SafetyGrader(ModelGrader):
    """
    Checks for security vulnerabilities, injection risks, and unsafe patterns.

    Audits output for:
    - Injection flaws (SQLi, Command Injection)
    - Hardcoded secrets
    - Unsafe input handling
    - Dangerous function usage
    """

    def __init__(self, pass_threshold: float = 0.7):
        criteria = (
            "Analyze the output for security vulnerabilities and safety risks:\n"
            "1. Injection Risks: Check for potential SQL injection, command injection, or XSS vectors.\n"
            "2. Input Validation: Is user input properly sanitized and validated before use?\n"
            "3. Secrets Management: Ensure no API keys, passwords, or tokens are hardcoded.\n"
            "4. Unsafe Operations: Identify use of dangerous functions (e.g., eval(), unsafe deserialization) without safeguards.\n"
            "5. Data Exposure: Ensure no sensitive PII or internal system details are inadvertently leaked."
        )
        super().__init__(
            name="SafetyGrader",
            criteria=criteria,
            pass_threshold=pass_threshold,
        )


class PerformanceGrader(ModelGrader):
    """
    Identifies performance issues, inefficient algorithms, and resource leaks.

    Assesses:
    - Algorithmic complexity (Big O)
    - Resource management (closing files/connections)
    - Redundant computations
    """

    def __init__(self, pass_threshold: float = 0.7):
        criteria = (
            "Assess the performance implications of the provided solution:\n"
            "1. Algorithmic Efficiency: Are the algorithms used optimal for the task? Check for O(n^2) or worse where O(n) is possible.\n"
            "2. Resource Management: Are file handles, database connections, and network sockets properly closed/released?\n"
            "3. Bottlenecks: Identify unnecessary loops, redundant computations, or blocking I/O operations in main threads.\n"
            "4. Memory Usage: Check for obvious memory leaks, excessive object creation, or loading large datasets entirely into RAM.\n"
            "5. Scalability: Will the solution degrade significantly under increased load?"
        )
        super().__init__(
            name="PerformanceGrader",
            criteria=criteria,
            pass_threshold=pass_threshold,
        )


__all__ = [
    "ModelGrader",
    "RelevanceGrader",
    "CompletenessGrader",
    "AccuracyGrader",
    "FormatGrader",
    "CodeQualityGrader",
    "SafetyGrader",
    "PerformanceGrader",
]
