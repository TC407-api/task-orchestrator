"""
Pattern Relevance Decay System.

Implements exponential decay with reinforcement for pattern relevance scores.
Uses a lazy decay model - scores are calculated on-the-fly when accessed.

Algorithm:
    S(t) = clamp((S_last * 2^(-Δt/h)) + W_outcome, 0.0, 1.0)

Where:
    - S(t): Current score
    - S_last: Score from last interaction
    - Δt: Time elapsed since last update
    - h: Half-life period (hours)
    - W_outcome: Weight of current interaction
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration Constants
DEFAULT_HALF_LIFE_HOURS = 72.0  # Score halves every 3 days if unused
STALENESS_THRESHOLD_DAYS = 14   # Flag as stale after 2 weeks
MIN_RELEVANCE_SCORE = 0.1       # Threshold for pruning
MAX_RELEVANCE_SCORE = 1.0
INITIAL_SCORE = 0.5             # Default starting score


class InteractionOutcome(Enum):
    """Weights for different interaction types."""
    STANDARD_MATCH = 0.05        # Normal usage, slight boost
    CRITICAL_SUCCESS = 0.20     # Prevented a failure, major boost
    PARTIAL_MATCH = 0.01        # Weak match, keep alive but low boost
    FAILURE_PENALTY = -0.15     # False positive, reduce score
    CRITICAL_FAILURE = -0.50    # Caused an error, massive penalty


@dataclass
class DecayMetadata:
    """
    Metadata for tracking pattern decay.

    Stored alongside patterns to enable lazy decay calculation.
    """
    relevance_score: float = INITIAL_SCORE
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    is_stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "relevance_score": self.relevance_score,
            "last_updated": self.last_updated.isoformat(),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "is_stale": self.is_stale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecayMetadata':
        """Create from dictionary."""
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        elif last_updated is None:
            last_updated = datetime.now(timezone.utc)

        return cls(
            relevance_score=data.get("relevance_score", INITIAL_SCORE),
            last_updated=last_updated,
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            is_stale=data.get("is_stale", False),
        )


class PatternDecaySystem:
    """
    Manages pattern relevance decay and reinforcement.

    Features:
    - Lazy decay: Scores calculated on-the-fly, not continuously updated
    - Reinforcement: Successful usage boosts scores
    - Penalty: Failed matches reduce scores
    - Staleness detection: Flag unused patterns
    - Pruning identification: Identify low-value patterns for removal
    """

    def __init__(
        self,
        half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
        staleness_days: int = STALENESS_THRESHOLD_DAYS,
        min_score: float = MIN_RELEVANCE_SCORE,
    ):
        """
        Initialize the decay system.

        Args:
            half_life_hours: Time for score to decay by 50%
            staleness_days: Days until pattern flagged as stale
            min_score: Threshold below which patterns can be pruned
        """
        self.half_life_hours = half_life_hours
        self.staleness_days = staleness_days
        self.min_score = min_score

        # In-memory cache for pattern metadata
        self._metadata_cache: Dict[str, DecayMetadata] = {}

        # Stats
        self._stats = {
            "decay_calculations": 0,
            "reinforcements": 0,
            "penalties": 0,
            "stale_marked": 0,
            "prune_candidates": 0,
        }

    def _calculate_decay(
        self,
        score: float,
        last_updated: datetime,
    ) -> float:
        """
        Calculate decayed score based on elapsed time.

        Formula: S(t) = S_0 * 2^(-delta_t / half_life)

        Args:
            score: Previous score value
            last_updated: When score was last updated

        Returns:
            Decayed score value
        """
        now = datetime.now(timezone.utc)
        elapsed_delta = now - last_updated
        elapsed_hours = elapsed_delta.total_seconds() / 3600.0

        if elapsed_hours <= 0:
            return score

        self._stats["decay_calculations"] += 1

        # Exponential decay
        decay_factor = 2 ** (-elapsed_hours / self.half_life_hours)
        return score * decay_factor

    def _check_staleness(self, last_updated: datetime) -> bool:
        """Check if a pattern is stale based on time threshold."""
        now = datetime.now(timezone.utc)
        return (now - last_updated).days >= self.staleness_days

    def _clamp_score(self, score: float) -> float:
        """Ensure score remains within bounds [0.0, 1.0]."""
        return max(0.0, min(MAX_RELEVANCE_SCORE, score))

    def get_current_relevance(
        self,
        pattern_id: str,
        stored_metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Get the effective relevance score right now.

        Performs lazy decay calculation without persisting.

        Args:
            pattern_id: ID of the pattern
            stored_metadata: Optional stored metadata dict

        Returns:
            Current decayed relevance score
        """
        # Check cache first
        if pattern_id in self._metadata_cache:
            meta = self._metadata_cache[pattern_id]
        elif stored_metadata:
            meta = DecayMetadata.from_dict(stored_metadata)
            self._metadata_cache[pattern_id] = meta
        else:
            # No metadata, return default
            return INITIAL_SCORE

        return self._calculate_decay(meta.relevance_score, meta.last_updated)

    def register_interaction(
        self,
        pattern_id: str,
        outcome: InteractionOutcome,
        stored_metadata: Optional[Dict[str, Any]] = None,
    ) -> DecayMetadata:
        """
        Register an interaction with a pattern.

        Workflow:
        1. Apply decay (catch up to now)
        2. Apply boost/penalty based on outcome
        3. Update metadata
        4. Return updated metadata for persistence

        Args:
            pattern_id: ID of the pattern
            outcome: Type of interaction (success, failure, etc.)
            stored_metadata: Optional stored metadata dict

        Returns:
            Updated DecayMetadata for persistence
        """
        # Get or create metadata
        if pattern_id in self._metadata_cache:
            meta = self._metadata_cache[pattern_id]
        elif stored_metadata:
            meta = DecayMetadata.from_dict(stored_metadata)
        else:
            meta = DecayMetadata()

        # 1. Apply decay first (bring score to 'now')
        current_decayed_score = self._calculate_decay(
            meta.relevance_score,
            meta.last_updated
        )

        # 2. Apply reinforcement/penalty
        new_score = current_decayed_score + outcome.value

        # 3. Update metadata
        meta.relevance_score = self._clamp_score(new_score)
        meta.last_updated = datetime.now(timezone.utc)
        meta.usage_count += 1

        # Track success/failure
        if outcome.value > 0:
            meta.success_count += 1
            meta.is_stale = False  # Reset staleness on success
            self._stats["reinforcements"] += 1
        elif outcome.value < 0:
            meta.failure_count += 1
            self._stats["penalties"] += 1

        # 4. Cache and return
        self._metadata_cache[pattern_id] = meta

        logger.debug(
            f"Pattern {pattern_id}: {outcome.name} -> score {meta.relevance_score:.3f}"
        )

        return meta

    def check_staleness(
        self,
        pattern_id: str,
        stored_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if a pattern is stale.

        Args:
            pattern_id: ID of the pattern
            stored_metadata: Optional stored metadata

        Returns:
            True if pattern is stale
        """
        if pattern_id in self._metadata_cache:
            meta = self._metadata_cache[pattern_id]
        elif stored_metadata:
            meta = DecayMetadata.from_dict(stored_metadata)
        else:
            return False

        is_stale = self._check_staleness(meta.last_updated)

        if is_stale and not meta.is_stale:
            meta.is_stale = True
            self._metadata_cache[pattern_id] = meta
            self._stats["stale_marked"] += 1

        return is_stale

    def should_prune(
        self,
        pattern_id: str,
        stored_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if a pattern should be pruned (removed).

        Criteria: Low score AND stale

        Args:
            pattern_id: ID of the pattern
            stored_metadata: Optional stored metadata

        Returns:
            True if pattern should be pruned
        """
        current_score = self.get_current_relevance(pattern_id, stored_metadata)
        is_stale = self.check_staleness(pattern_id, stored_metadata)

        should_prune = current_score < self.min_score and is_stale

        if should_prune:
            self._stats["prune_candidates"] += 1
            logger.info(
                f"Pattern {pattern_id} marked for pruning "
                f"(score={current_score:.3f}, stale={is_stale})"
            )

        return should_prune

    def batch_evaluate(
        self,
        patterns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate multiple patterns for staleness and pruning.

        Args:
            patterns: List of pattern dicts with 'id' and optional 'decay_metadata'

        Returns:
            Evaluation results with lists of stale and prune candidates
        """
        stale_ids: List[str] = []
        prune_ids: List[str] = []
        scores: Dict[str, float] = {}

        for pattern in patterns:
            pattern_id = pattern.get("id") or pattern.get("pattern_id")
            if not pattern_id:
                continue

            metadata = pattern.get("decay_metadata") or pattern.get("context", {}).get("decay_metadata")

            # Calculate current score
            score = self.get_current_relevance(pattern_id, metadata)
            scores[pattern_id] = score

            # Check staleness
            if self.check_staleness(pattern_id, metadata):
                stale_ids.append(pattern_id)

            # Check pruning
            if self.should_prune(pattern_id, metadata):
                prune_ids.append(pattern_id)

        return {
            "evaluated": len(patterns),
            "stale_count": len(stale_ids),
            "prune_count": len(prune_ids),
            "stale_ids": stale_ids,
            "prune_ids": prune_ids,
            "scores": scores,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get decay system statistics."""
        return {
            "cached_patterns": len(self._metadata_cache),
            "half_life_hours": self.half_life_hours,
            "staleness_days": self.staleness_days,
            "min_score_threshold": self.min_score,
            **self._stats,
        }

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._metadata_cache.clear()


# Module-level singleton
_decay_system: Optional[PatternDecaySystem] = None


def get_decay_system(**kwargs) -> PatternDecaySystem:
    """Get or create the global decay system instance."""
    global _decay_system
    if _decay_system is None:
        _decay_system = PatternDecaySystem(**kwargs)
    return _decay_system


def reset_decay_system() -> None:
    """Reset the global decay system (for testing)."""
    global _decay_system
    _decay_system = None


__all__ = [
    "PatternDecaySystem",
    "DecayMetadata",
    "InteractionOutcome",
    "get_decay_system",
    "reset_decay_system",
    "DEFAULT_HALF_LIFE_HOURS",
    "STALENESS_THRESHOLD_DAYS",
]
