"""
Prompt Guardrails for Graphiti Immune System.

This module modifies prompts based on past failure patterns to prevent
similar failures from occurring.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .pattern_matcher import MatchedPattern, PatternMatcher

logger = logging.getLogger(__name__)


# Guardrail templates for different failure types
GUARDRAIL_TEMPLATES = {
    "json_invalid": (
        "\n\nIMPORTANT: Your response MUST be valid JSON. "
        "Do not include markdown code blocks, comments, or trailing commas. "
        "Ensure all strings are properly quoted and all brackets are balanced."
    ),
    "schema_violation": (
        "\n\nIMPORTANT: Your response must strictly follow the required schema. "
        "Include all required fields and use the correct data types. "
        "Do not add extra fields not specified in the schema."
    ),
    "empty_response": (
        "\n\nIMPORTANT: You must provide a substantive response. "
        "Do not return an empty or minimal response. "
        "If you cannot complete the task, explain why in detail."
    ),
    "pattern_mismatch": (
        "\n\nIMPORTANT: Your response must follow the expected format. "
        "Review the requirements carefully and ensure your output matches "
        "the specified pattern or structure."
    ),
    "length_violation": (
        "\n\nIMPORTANT: Pay attention to length requirements. "
        "Ensure your response meets the minimum length while staying "
        "within any maximum length constraints."
    ),
    "hallucination": (
        "\n\nIMPORTANT: Only include information you are certain about. "
        "Do not make up facts, functions, or APIs that don't exist. "
        "If unsure, say so explicitly."
    ),
    "evaluation_failed": (
        "\n\nIMPORTANT: Previous similar requests have failed evaluation. "
        "Please be extra careful to follow all requirements exactly. "
        "Double-check your response before submitting."
    ),
}


@dataclass
class GuardrailResult:
    """
    Result of applying guardrails to a prompt.

    Attributes:
        original_prompt: The original prompt
        modified_prompt: The prompt with guardrails applied
        guardrails_applied: List of guardrail types that were applied
        risk_score: The assessed risk score
        warnings: Any warnings about potential issues
    """
    original_prompt: str
    modified_prompt: str
    guardrails_applied: List[str]
    risk_score: float
    warnings: List[str]

    @property
    def was_modified(self) -> bool:
        """Check if the prompt was modified."""
        return self.original_prompt != self.modified_prompt

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "was_modified": self.was_modified,
            "guardrails_applied": self.guardrails_applied,
            "risk_score": self.risk_score,
            "warnings": self.warnings,
        }


class PromptGuardrails:
    """
    Applies guardrails to prompts based on past failures.

    This class:
    - Queries for similar past failures
    - Determines appropriate guardrails to apply
    - Modifies prompts with protective instructions
    - Tracks guardrail effectiveness over time
    """

    def __init__(
        self,
        pattern_matcher: PatternMatcher,
        risk_threshold: float = 0.5,
        auto_apply: bool = True,
    ):
        """
        Initialize prompt guardrails.

        Args:
            pattern_matcher: Matcher for finding similar failures
            risk_threshold: Minimum risk score to trigger guardrails (0.0-1.0)
            auto_apply: Whether to automatically apply guardrails
        """
        self._matcher = pattern_matcher
        self._risk_threshold = risk_threshold
        self._auto_apply = auto_apply
        self._stats = {
            "prompts_checked": 0,
            "guardrails_applied": 0,
            "by_type": {},
        }

    def _get_guardrail_text(self, failure_type: str) -> str:
        """Get the guardrail text for a failure type."""
        return GUARDRAIL_TEMPLATES.get(
            failure_type,
            GUARDRAIL_TEMPLATES["evaluation_failed"],
        )

    def _build_modified_prompt(
        self,
        original: str,
        matches: List[MatchedPattern],
    ) -> tuple[str, List[str]]:
        """Build a modified prompt with guardrails."""
        guardrails_to_apply = set()
        for match in matches:
            guardrails_to_apply.add(match.pattern.failure_type)

        if not guardrails_to_apply:
            return original, []

        # Build the modified prompt
        modified = original
        applied = []

        for failure_type in guardrails_to_apply:
            guardrail_text = self._get_guardrail_text(failure_type)
            modified += guardrail_text
            applied.append(failure_type)

        return modified, applied

    async def apply_guardrails(
        self,
        prompt: str,
        operation: str,
    ) -> GuardrailResult:
        """
        Apply guardrails to a prompt based on past failures.

        Args:
            prompt: The original prompt
            operation: The operation being performed

        Returns:
            GuardrailResult with the (potentially modified) prompt
        """
        self._stats["prompts_checked"] += 1

        # Check for similar past failures
        risk_score, risk_reasons = await self._matcher.check_prompt_risk(
            prompt, operation
        )

        warnings = risk_reasons.copy()

        # If risk is below threshold and auto_apply is off, return original
        if risk_score < self._risk_threshold:
            return GuardrailResult(
                original_prompt=prompt,
                modified_prompt=prompt,
                guardrails_applied=[],
                risk_score=risk_score,
                warnings=warnings,
            )

        # Get matching patterns
        matches = await self._matcher.find_similar_failures(prompt, operation)

        if not matches:
            return GuardrailResult(
                original_prompt=prompt,
                modified_prompt=prompt,
                guardrails_applied=[],
                risk_score=risk_score,
                warnings=warnings,
            )

        # Apply guardrails if auto_apply is enabled
        if self._auto_apply:
            modified_prompt, applied = self._build_modified_prompt(prompt, matches)
        else:
            modified_prompt = prompt
            applied = []
            warnings.append(
                f"Guardrails available but not auto-applied (risk: {risk_score:.0%})"
            )

        # Update stats
        if applied:
            self._stats["guardrails_applied"] += 1
            for guardrail_type in applied:
                self._stats["by_type"][guardrail_type] = (
                    self._stats["by_type"].get(guardrail_type, 0) + 1
                )

        logger.info(
            f"Guardrails applied: {applied}, risk: {risk_score:.2f}"
        )

        return GuardrailResult(
            original_prompt=prompt,
            modified_prompt=modified_prompt,
            guardrails_applied=applied,
            risk_score=risk_score,
            warnings=warnings,
        )

    async def suggest_guardrails(
        self,
        prompt: str,
        operation: str,
    ) -> Dict[str, Any]:
        """
        Suggest guardrails without applying them.

        Args:
            prompt: The prompt to analyze
            operation: The operation being performed

        Returns:
            Dict with suggestions and risk assessment
        """
        matches = await self._matcher.find_similar_failures(prompt, operation)
        risk_score, risk_reasons = await self._matcher.check_prompt_risk(
            prompt, operation
        )

        suggestions = []
        for match in matches:
            suggestions.append({
                "failure_type": match.pattern.failure_type,
                "similarity": match.similarity_score,
                "guardrail": self._get_guardrail_text(match.pattern.failure_type),
                "reasons": match.match_reasons,
            })

        return {
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "suggestions": suggestions,
            "should_apply": risk_score >= self._risk_threshold,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get guardrail statistics."""
        return {
            "prompts_checked": self._stats["prompts_checked"],
            "guardrails_applied": self._stats["guardrails_applied"],
            "application_rate": (
                self._stats["guardrails_applied"] / self._stats["prompts_checked"]
                if self._stats["prompts_checked"] > 0 else 0.0
            ),
            "by_type": self._stats["by_type"].copy(),
            "risk_threshold": self._risk_threshold,
            "auto_apply": self._auto_apply,
        }


__all__ = [
    "GuardrailResult",
    "PromptGuardrails",
    "GUARDRAIL_TEMPLATES",
]
