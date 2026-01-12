"""
Core Immune System for Task Orchestrator.

This module provides the main ImmuneSystem class that coordinates
failure storage, pattern matching, and prompt guardrails.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .failure_store import FailurePattern, FailurePatternStore
from .pattern_matcher import MatchedPattern, PatternMatcher
from .guardrails import GuardrailResult, PromptGuardrails

logger = logging.getLogger(__name__)


@dataclass
class ImmuneResponse:
    """
    Response from the immune system after processing a prompt.

    Attributes:
        original_prompt: The original input prompt
        processed_prompt: The prompt after guardrails (may be same as original)
        risk_score: Assessed risk based on past failures (0.0-1.0)
        guardrails_applied: List of guardrail types that were applied
        matched_failures: List of similar past failures found
        warnings: Any warnings or risk reasons
        should_proceed: Whether to proceed with the operation
    """
    original_prompt: str
    processed_prompt: str
    risk_score: float
    guardrails_applied: List[str]
    matched_failures: List[Dict[str, Any]]
    warnings: List[str]
    should_proceed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/response."""
        return {
            "risk_score": self.risk_score,
            "guardrails_applied": self.guardrails_applied,
            "matched_failures_count": len(self.matched_failures),
            "warnings": self.warnings,
            "should_proceed": self.should_proceed,
            "was_modified": self.original_prompt != self.processed_prompt,
        }


class ImmuneSystem:
    """
    Main entry point for the Graphiti Immune System.

    The ImmuneSystem coordinates:
    - Pre-spawn checking for similar past failures
    - Prompt modification with protective guardrails
    - Post-execution failure recording
    - Statistics and health reporting

    Usage:
        immune = ImmuneSystem()

        # Before spawning agent
        response = await immune.pre_spawn_check(prompt, "spawn_agent")
        if response.should_proceed:
            result = await spawn_agent(response.processed_prompt)

        # After execution, if evaluation failed
        if not result.passed:
            await immune.record_failure(
                operation="spawn_agent",
                prompt=prompt,
                output=result.output,
                grader_results=result.grader_results,
            )
    """

    def __init__(
        self,
        graphiti_client: Optional[Any] = None,
        risk_threshold: float = 0.5,
        auto_apply_guardrails: bool = True,
        block_high_risk: bool = False,
        high_risk_threshold: float = 0.9,
    ):
        """
        Initialize the immune system.

        Args:
            graphiti_client: Optional Graphiti MCP client for persistence
            risk_threshold: Minimum risk to trigger guardrails (0.0-1.0)
            auto_apply_guardrails: Whether to auto-apply guardrails
            block_high_risk: Whether to block very high risk prompts
            high_risk_threshold: Risk score above which to block (if enabled)
        """
        # Initialize components
        self._failure_store = FailurePatternStore(graphiti_client)
        self._pattern_matcher = PatternMatcher(
            failure_store=self._failure_store,
            graphiti_client=graphiti_client,
            similarity_threshold=risk_threshold,
        )
        self._guardrails = PromptGuardrails(
            pattern_matcher=self._pattern_matcher,
            risk_threshold=risk_threshold,
            auto_apply=auto_apply_guardrails,
        )

        self._block_high_risk = block_high_risk
        self._high_risk_threshold = high_risk_threshold
        self._stats = {
            "pre_spawn_checks": 0,
            "failures_recorded": 0,
            "prompts_blocked": 0,
            "guardrails_applied": 0,
        }

        logger.info(
            f"ImmuneSystem initialized (risk_threshold={risk_threshold}, "
            f"auto_apply={auto_apply_guardrails}, block_high_risk={block_high_risk})"
        )

    async def pre_spawn_check(
        self,
        prompt: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ImmuneResponse:
        """
        Check a prompt before spawning an agent.

        This method:
        1. Looks for similar past failures
        2. Assesses the risk level
        3. Applies guardrails if needed
        4. Optionally blocks very high risk prompts

        Args:
            prompt: The input prompt to check
            operation: The operation being performed (e.g., "spawn_agent")
            context: Optional additional context

        Returns:
            ImmuneResponse with processed prompt and risk assessment
        """
        self._stats["pre_spawn_checks"] += 1

        # Find similar failures
        matches = await self._pattern_matcher.find_similar_failures(
            prompt, operation, limit=5
        )

        # Apply guardrails
        guardrail_result = await self._guardrails.apply_guardrails(prompt, operation)

        # Determine if we should proceed
        should_proceed = True
        if self._block_high_risk and guardrail_result.risk_score >= self._high_risk_threshold:
            should_proceed = False
            self._stats["prompts_blocked"] += 1
            logger.warning(
                f"Blocking high-risk prompt (risk={guardrail_result.risk_score:.2f})"
            )

        if guardrail_result.guardrails_applied:
            self._stats["guardrails_applied"] += 1

        return ImmuneResponse(
            original_prompt=prompt,
            processed_prompt=guardrail_result.modified_prompt,
            risk_score=guardrail_result.risk_score,
            guardrails_applied=guardrail_result.guardrails_applied,
            matched_failures=[m.to_dict() for m in matches],
            warnings=guardrail_result.warnings,
            should_proceed=should_proceed,
        )

    async def record_failure(
        self,
        operation: str,
        prompt: str,
        output: str,
        grader_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> FailurePattern:
        """
        Record a failure for future pattern matching.

        Call this after an evaluation fails to build the immune system's
        knowledge of what prompts lead to failures.

        Args:
            operation: The operation that failed
            prompt: The input prompt
            output: The problematic output
            grader_results: List of grader result dicts
            context: Optional additional context (model, cost, etc.)

        Returns:
            The stored FailurePattern
        """
        self._stats["failures_recorded"] += 1

        pattern = await self._failure_store.store_failure(
            operation=operation,
            input_prompt=prompt,
            output=output,
            grader_results=grader_results,
            context=context,
        )

        logger.info(
            f"Recorded failure pattern: {pattern.id} ({pattern.failure_type})"
        )

        return pattern

    async def get_suggestions(
        self,
        prompt: str,
        operation: str,
    ) -> Dict[str, Any]:
        """
        Get guardrail suggestions without applying them.

        Useful for debugging or manual review of what the immune
        system would suggest for a given prompt.

        Args:
            prompt: The prompt to analyze
            operation: The operation being performed

        Returns:
            Dict with suggestions and risk assessment
        """
        return await self._guardrails.suggest_guardrails(prompt, operation)

    def get_stats(self) -> Dict[str, Any]:
        """Get immune system statistics."""
        return {
            "immune_system": {
                "pre_spawn_checks": self._stats["pre_spawn_checks"],
                "failures_recorded": self._stats["failures_recorded"],
                "prompts_blocked": self._stats["prompts_blocked"],
                "guardrails_applied": self._stats["guardrails_applied"],
            },
            "failure_store": self._failure_store.get_stats(),
            "guardrails": self._guardrails.get_stats(),
        }

    def get_health(self) -> Dict[str, Any]:
        """Get immune system health status."""
        store_stats = self._failure_store.get_stats()

        return {
            "status": "healthy",
            "total_patterns": store_stats["total_patterns"],
            "total_occurrences": store_stats["total_occurrences"],
            "checks_performed": self._stats["pre_spawn_checks"],
            "block_rate": (
                self._stats["prompts_blocked"] / self._stats["pre_spawn_checks"]
                if self._stats["pre_spawn_checks"] > 0 else 0.0
            ),
            "guardrail_rate": (
                self._stats["guardrails_applied"] / self._stats["pre_spawn_checks"]
                if self._stats["pre_spawn_checks"] > 0 else 0.0
            ),
        }


# Singleton instance for global access
_immune_system: Optional[ImmuneSystem] = None


def get_immune_system(
    graphiti_client: Optional[Any] = None,
    **kwargs,
) -> ImmuneSystem:
    """
    Get or create the global immune system instance.

    Args:
        graphiti_client: Optional Graphiti client
        **kwargs: Additional arguments for ImmuneSystem

    Returns:
        The global ImmuneSystem instance
    """
    global _immune_system
    if _immune_system is None:
        _immune_system = ImmuneSystem(graphiti_client, **kwargs)
    return _immune_system


def reset_immune_system() -> None:
    """Reset the global immune system instance (for testing)."""
    global _immune_system
    _immune_system = None


__all__ = [
    "ImmuneResponse",
    "ImmuneSystem",
    "get_immune_system",
    "reset_immune_system",
]
