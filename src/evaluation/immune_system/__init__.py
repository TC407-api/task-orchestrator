"""
Graphiti Immune System for Task Orchestrator.

This module provides a self-learning system that:
- Stores evaluation failures in Graphiti
- Matches new prompts against past failure patterns
- Applies protective guardrails to prevent similar failures
- Tracks effectiveness and provides health metrics

Usage:
    from evaluation.immune_system import ImmuneSystem, get_immune_system

    # Get the global instance
    immune = get_immune_system()

    # Pre-spawn check
    response = await immune.pre_spawn_check(prompt, "spawn_agent")
    if response.should_proceed:
        result = await spawn_agent(response.processed_prompt)

    # Record failures
    if not result.passed:
        await immune.record_failure(
            operation="spawn_agent",
            prompt=prompt,
            output=result.output,
            grader_results=result.grader_results,
        )
"""

from .failure_store import (
    FailurePattern,
    FailurePatternStore,
    FAILURE_GROUP_ID,
)
from .pattern_matcher import (
    MatchedPattern,
    PatternMatcher,
)
from .guardrails import (
    GuardrailResult,
    PromptGuardrails,
    GUARDRAIL_TEMPLATES,
)
from .core import (
    ImmuneResponse,
    ImmuneSystem,
    get_immune_system,
    reset_immune_system,
)
from .dashboard import (
    ImmuneDashboard,
    create_dashboard,
)
from .federation import (
    PatternFederation,
    PatternVisibility,
    FederatedPattern,
    ScoredPattern,
)


__all__ = [
    # Core
    "ImmuneResponse",
    "ImmuneSystem",
    "get_immune_system",
    "reset_immune_system",
    # Failure Store
    "FailurePattern",
    "FailurePatternStore",
    "FAILURE_GROUP_ID",
    # Pattern Matcher
    "MatchedPattern",
    "PatternMatcher",
    # Guardrails
    "GuardrailResult",
    "PromptGuardrails",
    "GUARDRAIL_TEMPLATES",
    # Dashboard (Phase 7)
    "ImmuneDashboard",
    "create_dashboard",
    # Federation (Phase 8.3)
    "PatternFederation",
    "PatternVisibility",
    "FederatedPattern",
    "ScoredPattern",
]
