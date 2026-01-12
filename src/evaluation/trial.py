"""
Core data structures for the evaluation system.

This module defines the schema for capturing agent execution results (Trials)
and the assessment of those results (GraderResults).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Dict
from uuid import uuid4


@dataclass
class GraderResult:
    """
    Result from a single grader evaluation.

    Attributes:
        name: The identifier of the grader (e.g., 'json_validity', 'hallucination_check').
        passed: Boolean indicating if the check passed the threshold.
        score: A normalized score between 0.0 and 1.0.
        reason: Human-readable explanation of the score.
        metadata: Arbitrary dictionary for extra context (tokens used, sub-scores, etc).
    """
    name: str
    passed: bool
    score: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the grader result to a dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class Trial:
    """
    Complete record of an agent execution with evaluation.

    This object serves as the central artifact for the evaluation pipeline.
    It captures the 'What happened' (inputs/outputs) and the 'How well did it go'
    (grader results).

    Attributes:
        operation: Name of the operation or tool being executed.
        input_prompt: The raw input or prompt sent to the model.
        id: Unique identifier for this specific execution.
        output: The raw output from the model/agent.
        model: Model identifier (e.g., 'gemini-3-pro-preview').
        circuit_breaker_state: State of the circuit breaker during execution.
        cost_usd: Estimated cost of the call.
        latency_ms: Execution time in milliseconds.
        langfuse_trace_id: Correlation ID for external observability.
        grader_results: List of assessments performed on this trial.
        pass_fail: Aggregate boolean status (True only if ALL graders pass).
        created_at: Timestamp of creation (UTC).
        metadata: Additional context tags.
    """
    operation: str
    input_prompt: str
    id: str = field(default_factory=lambda: str(uuid4()))
    output: Any = None
    model: str = ""
    circuit_breaker_state: str = "CLOSED"
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    langfuse_trace_id: str = ""
    grader_results: List[GraderResult] = field(default_factory=list)
    pass_fail: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_grader_result(self, result: GraderResult) -> None:
        """
        Appends a grader result and updates the aggregate pass_fail status.

        The Trial is considered failed if ANY single grader fails.
        """
        self.grader_results.append(result)
        self.pass_fail = all(r.passed for r in self.grader_results)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Trial to a dictionary for logging or database storage.

        Note: Large inputs/outputs are truncated in the preview fields.
        """
        return {
            "id": self.id,
            "operation": self.operation,
            "input_prompt": self.input_prompt[:200] if self.input_prompt else "",
            "output_preview": str(self.output)[:200] if self.output else None,
            "model": self.model,
            "circuit_breaker_state": self.circuit_breaker_state,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "langfuse_trace_id": self.langfuse_trace_id,
            "grader_results": [r.to_dict() for r in self.grader_results],
            "pass_fail": self.pass_fail,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
