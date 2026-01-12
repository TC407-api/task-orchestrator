"""
Graders module for the task-orchestrator evaluation system.

This module exports the base classes and concrete implementations for
various grading tasks, including structure validation, content matching,
logic checks, and LLM-based evaluation.
"""

from .base import Grader, GraderPipeline
from .code import (
    NonEmptyGrader,
    JSONValidGrader,
    JSONSchemaGrader,
    RegexGrader,
    LengthGrader,
    ContainsGrader,
    NotContainsGrader,
)
from .model import (
    ModelGrader,
    RelevanceGrader,
    CompletenessGrader,
    AccuracyGrader,
    FormatGrader,
)

__all__ = [
    # Base
    "Grader",
    "GraderPipeline",
    # Code-based graders
    "NonEmptyGrader",
    "JSONValidGrader",
    "JSONSchemaGrader",
    "RegexGrader",
    "LengthGrader",
    "ContainsGrader",
    "NotContainsGrader",
    # Model-based graders (LLM-as-judge)
    "ModelGrader",
    "RelevanceGrader",
    "CompletenessGrader",
    "AccuracyGrader",
    "FormatGrader",
]
