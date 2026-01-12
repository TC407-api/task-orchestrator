"""
Evaluation Module for Task Orchestrator.

This package provides the core primitives for the evaluation system,
allowing for the creation of Trials and the application of Graders to assess
agent performance.

Components:
- Trial: Complete record of an agent execution with evaluation
- GraderResult: Result from a single grader evaluation
- Graders: Various code-based validators (JSON, regex, length, etc.)
- Integration: Langfuse Scores API wrapper
- Export: Training data exporter
- Immune System: Failure pattern storage and guardrails
"""

from .trial import Trial, GraderResult
from .graders import (
    Grader,
    GraderPipeline,
    NonEmptyGrader,
    JSONValidGrader,
    JSONSchemaGrader,
    RegexGrader,
    LengthGrader,
    ContainsGrader,
    NotContainsGrader,
)
from .integration import score_trial, score_grader_result, create_eval_span
from .export import TrainingDataExporter, get_exporter
from .immune_system import (
    ImmuneSystem,
    ImmuneResponse,
    get_immune_system,
    reset_immune_system,
    FailurePattern,
    FailurePatternStore,
    PatternMatcher,
    MatchedPattern,
    PromptGuardrails,
    GuardrailResult,
)

__all__ = [
    # Core
    "Trial",
    "GraderResult",
    # Graders
    "Grader",
    "GraderPipeline",
    "NonEmptyGrader",
    "JSONValidGrader",
    "JSONSchemaGrader",
    "RegexGrader",
    "LengthGrader",
    "ContainsGrader",
    "NotContainsGrader",
    # Integration
    "score_trial",
    "score_grader_result",
    "create_eval_span",
    # Export
    "TrainingDataExporter",
    "get_exporter",
    # Immune System
    "ImmuneSystem",
    "ImmuneResponse",
    "get_immune_system",
    "reset_immune_system",
    "FailurePattern",
    "FailurePatternStore",
    "PatternMatcher",
    "MatchedPattern",
    "PromptGuardrails",
    "GuardrailResult",
]
