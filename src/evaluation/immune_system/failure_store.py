"""
Failure Pattern Storage for Graphiti Immune System.

This module stores evaluation failures in Graphiti for later retrieval
and pattern matching, enabling the system to learn from past mistakes.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from hashlib import sha256

logger = logging.getLogger(__name__)

# Group ID for task orchestrator failures in Graphiti
FAILURE_GROUP_ID = "project_task_orchestrator_failures"


@dataclass
class FailurePattern:
    """
    Represents a stored failure pattern.

    Attributes:
        id: Unique identifier for this failure
        operation: The operation that failed (e.g., "spawn_agent")
        failure_type: Category of failure (e.g., "json_invalid", "hallucination")
        input_summary: Summary of the input that triggered the failure
        output_summary: Summary of the problematic output
        grader_scores: Dict of grader names to scores
        context: Additional context about the failure
        created_at: When this failure was recorded
        occurrence_count: How many times this pattern has been seen
    """
    id: str
    operation: str
    failure_type: str
    input_summary: str
    output_summary: str
    grader_scores: Dict[str, float]
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    occurrence_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "operation": self.operation,
            "failure_type": self.failure_type,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "grader_scores": self.grader_scores,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "occurrence_count": self.occurrence_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailurePattern":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            operation=data["operation"],
            failure_type=data["failure_type"],
            input_summary=data["input_summary"],
            output_summary=data["output_summary"],
            grader_scores=data["grader_scores"],
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            occurrence_count=data.get("occurrence_count", 1),
        )


class FailurePatternStore:
    """
    Stores and retrieves failure patterns from Graphiti.

    This class handles:
    - Recording new failures to Graphiti
    - Deduplicating similar failures
    - Generating failure summaries for storage
    - Retrieving failure patterns by type or operation
    """

    def __init__(self, graphiti_client: Optional[Any] = None):
        """
        Initialize the failure store.

        Args:
            graphiti_client: Optional Graphiti client for storage.
                           If None, will use local file-based storage.
        """
        self._graphiti = graphiti_client
        self._local_cache: Dict[str, FailurePattern] = {}
        self._use_graphiti = graphiti_client is not None

    def _generate_failure_id(
        self,
        operation: str,
        failure_type: str,
        input_summary: str,
    ) -> str:
        """Generate a unique ID for a failure pattern."""
        content = f"{operation}:{failure_type}:{input_summary[:100]}"
        return sha256(content.encode()).hexdigest()[:16]

    def _summarize_input(self, prompt: str, max_length: int = 200) -> str:
        """Create a summary of the input prompt."""
        if len(prompt) <= max_length:
            return prompt
        return prompt[:max_length - 3] + "..."

    def _summarize_output(self, output: str, max_length: int = 200) -> str:
        """Create a summary of the output."""
        if not output:
            return "[empty]"
        if len(output) <= max_length:
            return output
        return output[:max_length - 3] + "..."

    def _determine_failure_type(
        self,
        grader_results: List[Dict[str, Any]],
    ) -> str:
        """Determine the primary failure type from grader results."""
        for result in grader_results:
            if not result.get("passed", True):
                name = result.get("name", "unknown")
                if "json" in name.lower():
                    return "json_invalid"
                if "regex" in name.lower() or "pattern" in name.lower():
                    return "pattern_mismatch"
                if "schema" in name.lower():
                    return "schema_violation"
                if "length" in name.lower():
                    return "length_violation"
                if "empty" in name.lower():
                    return "empty_response"
        return "evaluation_failed"

    async def store_failure(
        self,
        operation: str,
        input_prompt: str,
        output: str,
        grader_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> FailurePattern:
        """
        Store a failure pattern.

        Args:
            operation: The operation that failed
            input_prompt: The input that triggered the failure
            output: The problematic output
            grader_results: List of grader result dicts
            context: Additional context (model, cost, etc.)

        Returns:
            The stored FailurePattern
        """
        failure_type = self._determine_failure_type(grader_results)
        input_summary = self._summarize_input(input_prompt)
        output_summary = self._summarize_output(output)

        failure_id = self._generate_failure_id(operation, failure_type, input_summary)

        # Check for existing pattern
        if failure_id in self._local_cache:
            existing = self._local_cache[failure_id]
            existing.occurrence_count += 1
            logger.info(f"Failure pattern {failure_id} seen {existing.occurrence_count} times")
            return existing

        # Create new pattern
        grader_scores = {
            r.get("name", f"grader_{i}"): r.get("score", 0.0)
            for i, r in enumerate(grader_results)
        }

        pattern = FailurePattern(
            id=failure_id,
            operation=operation,
            failure_type=failure_type,
            input_summary=input_summary,
            output_summary=output_summary,
            grader_scores=grader_scores,
            context=context or {},
        )

        # Store locally
        self._local_cache[failure_id] = pattern

        # Store in Graphiti if available
        if self._use_graphiti:
            await self._store_to_graphiti(pattern)

        logger.info(f"Stored failure pattern: {failure_id} ({failure_type})")
        return pattern

    async def _store_to_graphiti(self, pattern: FailurePattern) -> None:
        """Store pattern to Graphiti knowledge graph."""
        try:
            episode_body = json.dumps({
                "type": "failure_pattern",
                "failure_type": pattern.failure_type,
                "operation": pattern.operation,
                "input_summary": pattern.input_summary,
                "output_summary": pattern.output_summary,
                "grader_scores": pattern.grader_scores,
                "context": pattern.context,
            })

            await self._graphiti.add_memory(
                name=f"failure_{pattern.id}",
                episode_body=episode_body,
                group_id=FAILURE_GROUP_ID,
                source="json",
                source_description=f"Evaluation failure: {pattern.failure_type}",
            )
        except Exception as e:
            logger.error(f"Failed to store to Graphiti: {e}")

    async def get_failures_by_type(
        self,
        failure_type: str,
        limit: int = 10,
    ) -> List[FailurePattern]:
        """Get failure patterns by type."""
        return [
            p for p in self._local_cache.values()
            if p.failure_type == failure_type
        ][:limit]

    async def get_failures_by_operation(
        self,
        operation: str,
        limit: int = 10,
    ) -> List[FailurePattern]:
        """Get failure patterns by operation."""
        return [
            p for p in self._local_cache.values()
            if p.operation == operation
        ][:limit]

    async def get_recent_failures(
        self,
        limit: int = 10,
    ) -> List[FailurePattern]:
        """Get most recent failure patterns."""
        sorted_patterns = sorted(
            self._local_cache.values(),
            key=lambda p: p.created_at,
            reverse=True,
        )
        return sorted_patterns[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored failures."""
        patterns = list(self._local_cache.values())
        if not patterns:
            return {
                "total_patterns": 0,
                "total_occurrences": 0,
                "by_type": {},
                "by_operation": {},
            }

        by_type: Dict[str, int] = {}
        by_operation: Dict[str, int] = {}
        total_occurrences = 0

        for p in patterns:
            by_type[p.failure_type] = by_type.get(p.failure_type, 0) + 1
            by_operation[p.operation] = by_operation.get(p.operation, 0) + 1
            total_occurrences += p.occurrence_count

        return {
            "total_patterns": len(patterns),
            "total_occurrences": total_occurrences,
            "by_type": by_type,
            "by_operation": by_operation,
        }

    def get_all_patterns(self) -> List[FailurePattern]:
        """Get all stored failure patterns."""
        return list(self._local_cache.values())

    async def store_pattern(self, pattern: FailurePattern) -> None:
        """
        Store a pre-constructed failure pattern.

        Used for loading patterns from Graphiti or other sources.

        Args:
            pattern: The FailurePattern to store
        """
        if pattern.id in self._local_cache:
            existing = self._local_cache[pattern.id]
            existing.occurrence_count = max(
                existing.occurrence_count,
                pattern.occurrence_count
            )
            logger.debug(f"Updated existing pattern {pattern.id}")
        else:
            self._local_cache[pattern.id] = pattern
            logger.debug(f"Stored pattern {pattern.id}")


__all__ = [
    "FailurePattern",
    "FailurePatternStore",
    "FAILURE_GROUP_ID",
]
