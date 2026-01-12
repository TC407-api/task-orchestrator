"""
Evaluation Suites for task-orchestrator.

This module exports pre-configured evaluation suites for different testing scenarios.
"""

from .unit import (
    UnitEvalSuite,
    eval_code_generation,
    eval_json_response,
    eval_explanation,
)
from .resilience import (
    ResilienceEvalSuite,
    ResilienceTestResult,
    run_resilience_suite,
)

__all__ = [
    # Unit suite
    "UnitEvalSuite",
    "eval_code_generation",
    "eval_json_response",
    "eval_explanation",
    # Resilience suite
    "ResilienceEvalSuite",
    "ResilienceTestResult",
    "run_resilience_suite",
]
