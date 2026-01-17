"""
Pattern Matcher for Graphiti Immune System.

This module queries stored failure patterns to find similar past failures,
enabling pre-emptive guardrails before spawning agents.
"""

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from .failure_store import FailurePattern, FailurePatternStore, FAILURE_GROUP_ID

logger = logging.getLogger(__name__)


@dataclass
class MatchedPattern:
    """
    Represents a matched failure pattern with similarity score.

    Attributes:
        pattern: The matched FailurePattern
        similarity_score: How similar this pattern is (0.0 to 1.0)
        match_reasons: List of reasons why this matched
    """
    pattern: FailurePattern
    similarity_score: float
    match_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern.id,
            "failure_type": self.pattern.failure_type,
            "operation": self.pattern.operation,
            "similarity_score": self.similarity_score,
            "match_reasons": self.match_reasons,
            "occurrence_count": self.pattern.occurrence_count,
            "input_summary": self.pattern.input_summary,
        }


class PatternMatcher:
    """
    Matches incoming prompts against stored failure patterns.

    This class provides:
    - Text similarity matching against past failure inputs
    - Keyword extraction and matching
    - Graphiti semantic search integration
    - Configurable similarity thresholds
    """

    # Keywords that often appear in problematic prompts
    RISK_KEYWORDS = [
        "generate code", "write function", "create class",
        "json output", "structured response", "format as",
        "list all", "comprehensive", "detailed",
        "analyze", "summarize", "extract",
    ]

    def __init__(
        self,
        failure_store: FailurePatternStore,
        graphiti_client: Optional[Any] = None,
        similarity_threshold: float = 0.6,
    ):
        """
        Initialize the pattern matcher.

        Args:
            failure_store: Store of failure patterns
            graphiti_client: Optional Graphiti client for semantic search
            similarity_threshold: Minimum similarity to consider a match (0.0-1.0)
        """
        self._store = failure_store
        self._graphiti = graphiti_client
        self._similarity_threshold = similarity_threshold

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        text_lower = text.lower()
        found = []
        for keyword in self.RISK_KEYWORDS:
            if keyword in text_lower:
                found.append(keyword)
        return found

    def _calculate_keyword_overlap(
        self,
        keywords1: List[str],
        keywords2: List[str],
    ) -> float:
        """Calculate overlap between keyword sets."""
        if not keywords1 or not keywords2:
            return 0.0
        set1, set2 = set(keywords1), set(keywords2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _match_pattern(
        self,
        prompt: str,
        operation: str,
        pattern: FailurePattern,
    ) -> Optional[MatchedPattern]:
        """
        Check if a prompt matches a failure pattern.

        Returns MatchedPattern if similarity exceeds threshold, None otherwise.
        """
        match_reasons = []
        scores = []

        # Operation match (exact)
        if pattern.operation == operation:
            match_reasons.append(f"Same operation: {operation}")
            scores.append(1.0)

        # Text similarity
        text_sim = self._calculate_text_similarity(prompt, pattern.input_summary)
        if text_sim > 0.3:
            match_reasons.append(f"Text similarity: {text_sim:.2f}")
            scores.append(text_sim)

        # Keyword overlap
        prompt_keywords = self._extract_keywords(prompt)
        pattern_keywords = self._extract_keywords(pattern.input_summary)
        keyword_overlap = self._calculate_keyword_overlap(prompt_keywords, pattern_keywords)
        if keyword_overlap > 0.3:
            match_reasons.append(f"Keyword overlap: {keyword_overlap:.2f}")
            scores.append(keyword_overlap)

        # High occurrence count boosts score
        if pattern.occurrence_count > 3:
            match_reasons.append(f"Frequent failure ({pattern.occurrence_count}x)")
            scores.append(min(pattern.occurrence_count / 10, 1.0))

        if not scores:
            return None

        # Weighted average with operation match having highest weight
        avg_score = sum(scores) / len(scores)

        if avg_score >= self._similarity_threshold:
            return MatchedPattern(
                pattern=pattern,
                similarity_score=avg_score,
                match_reasons=match_reasons,
            )

        return None

    async def find_similar_failures(
        self,
        prompt: str,
        operation: str,
        limit: int = 5,
    ) -> List[MatchedPattern]:
        """
        Find failure patterns similar to the given prompt.

        Args:
            prompt: The input prompt to check
            operation: The operation being performed
            limit: Maximum number of matches to return

        Returns:
            List of MatchedPattern objects, sorted by similarity score
        """
        matches = []

        # Get patterns from local cache by operation first
        operation_patterns = await self._store.get_failures_by_operation(operation)
        for pattern in operation_patterns:
            match = self._match_pattern(prompt, operation, pattern)
            if match:
                matches.append(match)

        # Also check recent failures across all operations
        recent_patterns = await self._store.get_recent_failures(limit=20)
        for pattern in recent_patterns:
            if pattern.operation != operation:  # Don't double-check
                match = self._match_pattern(prompt, operation, pattern)
                if match:
                    matches.append(match)

        # Query Graphiti for semantic matches if available
        if self._graphiti:
            graphiti_matches = await self._query_graphiti(prompt, operation)
            matches.extend(graphiti_matches)

        # Sort by similarity score and deduplicate
        seen_ids = set()
        unique_matches = []
        for m in sorted(matches, key=lambda x: x.similarity_score, reverse=True):
            if m.pattern.id not in seen_ids:
                seen_ids.add(m.pattern.id)
                unique_matches.append(m)

        return unique_matches[:limit]

    async def _query_graphiti(
        self,
        prompt: str,
        operation: str,
    ) -> List[MatchedPattern]:
        """Query Graphiti for semantic matches."""
        if self._graphiti is None:
            return []

        try:
            results = await self._graphiti.search_memory_facts(
                query=f"failure pattern for {operation}: {prompt[:100]}",
                group_ids=[FAILURE_GROUP_ID],
                max_facts=5,
            )

            matches = []
            for fact in results:
                # Extract pattern info from fact
                # This is a placeholder - actual implementation depends on Graphiti response format
                pass

            return matches
        except Exception as e:
            logger.warning(f"Graphiti query failed: {e}")
            return []

    async def check_prompt_risk(
        self,
        prompt: str,
        operation: str,
    ) -> Tuple[float, List[str]]:
        """
        Assess the risk level of a prompt based on past failures.

        Args:
            prompt: The input prompt to check
            operation: The operation being performed

        Returns:
            Tuple of (risk_score 0.0-1.0, list of risk reasons)
        """
        matches = await self.find_similar_failures(prompt, operation, limit=3)

        if not matches:
            return 0.0, []

        # Calculate aggregate risk
        risk_reasons = []
        total_score = 0.0

        for match in matches:
            total_score += match.similarity_score
            risk_reasons.append(
                f"Similar to past failure ({match.pattern.failure_type}): "
                f"{match.similarity_score:.0%} match"
            )

        # Normalize to 0-1 range
        risk_score = min(total_score / len(matches), 1.0)

        return risk_score, risk_reasons


__all__ = [
    "MatchedPattern",
    "PatternMatcher",
]
