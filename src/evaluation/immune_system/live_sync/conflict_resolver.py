"""
Conflict Resolution System for Graphiti Federation Sync.

Uses version vectors (vector clocks) for causal ordering and
provides multiple resolution strategies for concurrent modifications.
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

NodeId = str
PatternId = str
VectorClock = Dict[NodeId, int]


class ConflictStrategy(Enum):
    """Defines the strategy to use when concurrent modification is detected."""
    LAST_WRITE_WINS = auto()
    MERGE = auto()
    MANUAL = auto()
    LOCAL_WINS = auto()
    REMOTE_WINS = auto()


@dataclass
class VersionVector:
    """Represents a Version Vector (Vector Clock) for causal ordering."""
    clocks: VectorClock = field(default_factory=dict)

    def increment(self, node: NodeId) -> None:
        """Increments the counter for a specific node."""
        self.clocks[node] = self.clocks.get(node, 0) + 1

    def merge(self, other: 'VersionVector') -> 'VersionVector':
        """Returns a new vector with max values from both vectors."""
        new_clocks = self.clocks.copy()
        for node, counter in other.clocks.items():
            new_clocks[node] = max(new_clocks.get(node, 0), counter)
        return VersionVector(new_clocks)

    def compare(self, other: 'VersionVector') -> Optional[int]:
        """
        Compares two vectors.
        Returns:
            -1 if self < other (causally precedes)
             1 if self > other (causally succeeds)
             0 if self == other (identical)
            None if concurrent (conflict)
        """
        keys = set(self.clocks.keys()) | set(other.clocks.keys())
        has_greater = False
        has_lesser = False

        for k in keys:
            v1 = self.clocks.get(k, 0)
            v2 = other.clocks.get(k, 0)
            if v1 > v2:
                has_greater = True
            elif v1 < v2:
                has_lesser = True

        if has_greater and has_lesser:
            return None  # Concurrent
        if has_greater:
            return 1
        if has_lesser:
            return -1
        return 0

    def __repr__(self) -> str:
        return f"VV({self.clocks})"


@dataclass
class Pattern:
    """Represents the data object being synchronized."""
    pattern_id: PatternId
    node_id: NodeId
    data: Dict[str, Any]
    timestamp: float
    version_vector: VersionVector = field(default_factory=VersionVector)

    def copy(self) -> 'Pattern':
        return copy.deepcopy(self)


@dataclass
class ConflictRecord:
    """Audit trail record for a resolved conflict."""
    pattern_id: PatternId
    timestamp: float
    strategy: ConflictStrategy
    local_vector: str
    remote_vector: str
    resolution_source: str


class ConflictResolutionError(Exception):
    """Base exception for resolution failures."""
    pass


class ConflictResolver:
    """
    Handles synchronization logic, conflict detection, and resolution
    for distributed patterns.
    """

    def __init__(
        self,
        node_id: NodeId,
        strategy: ConflictStrategy = ConflictStrategy.LAST_WRITE_WINS,
        on_conflict_resolved: Optional[Callable[[ConflictRecord], None]] = None
    ):
        """
        Args:
            node_id: Unique identifier for this node.
            strategy: Default resolution strategy.
            on_conflict_resolved: Callback triggered after resolution.
        """
        self.node_id = node_id
        self.strategy = strategy
        self.audit_trail: List[ConflictRecord] = []
        self._callback = on_conflict_resolved
        logger.info(
            f"ConflictResolver initialized for node '{node_id}' "
            f"with strategy {strategy.name}"
        )

    def resolve(
        self,
        local: Optional[Pattern],
        remote: Pattern
    ) -> Tuple[Pattern, bool]:
        """
        Syncs a remote pattern against the local state.

        Args:
            local: The current local version (None if new).
            remote: The incoming pattern from another node.

        Returns:
            Tuple of (resulting pattern, whether local state changed).
        """
        # Case 1: New pattern
        if local is None:
            logger.debug(f"New pattern {remote.pattern_id} received. Accepting.")
            return remote, True

        # Case 2: Compare Version Vectors
        comparison = local.version_vector.compare(remote.version_vector)

        if comparison == 0:
            logger.debug(f"Pattern {local.pattern_id}: Vectors identical. No change.")
            return local, False

        if comparison == -1:
            logger.debug(f"Pattern {local.pattern_id}: Local is stale. Updating.")
            return remote, True

        if comparison == 1:
            logger.debug(f"Pattern {local.pattern_id}: Remote is stale. Ignoring.")
            return local, False

        # Case 3: Concurrent modification (comparison is None)
        logger.warning(
            f"Conflict detected for {local.pattern_id}. "
            f"Local: {local.version_vector}, Remote: {remote.version_vector}"
        )
        resolved_pattern, source = self._apply_strategy(local, remote)

        # Update vector clock
        merged_vector = local.version_vector.merge(remote.version_vector)
        merged_vector.increment(self.node_id)
        resolved_pattern.version_vector = merged_vector

        # Audit and Notify
        record = ConflictRecord(
            pattern_id=local.pattern_id,
            timestamp=time.time(),
            strategy=self.strategy,
            local_vector=str(local.version_vector),
            remote_vector=str(remote.version_vector),
            resolution_source=source
        )
        self.audit_trail.append(record)

        if self._callback:
            try:
                self._callback(record)
            except Exception as e:
                logger.error(f"Error in conflict callback: {e}")

        return resolved_pattern, True

    def _apply_strategy(
        self,
        local: Pattern,
        remote: Pattern
    ) -> Tuple[Pattern, str]:
        """Execute the chosen resolution strategy."""
        if self.strategy == ConflictStrategy.LAST_WRITE_WINS:
            return self._resolve_lww(local, remote)

        elif self.strategy == ConflictStrategy.LOCAL_WINS:
            return local.copy(), "local"

        elif self.strategy == ConflictStrategy.REMOTE_WINS:
            return remote.copy(), "remote"

        elif self.strategy == ConflictStrategy.MERGE:
            return self._resolve_merge(local, remote)

        elif self.strategy == ConflictStrategy.MANUAL:
            logger.info("Manual strategy: keeping local but flagging conflict.")
            return local.copy(), "manual_default_local"

        else:
            raise ConflictResolutionError(f"Unknown strategy: {self.strategy}")

    def _resolve_lww(
        self,
        local: Pattern,
        remote: Pattern
    ) -> Tuple[Pattern, str]:
        """Last-Write-Wins based on timestamp with node_id tie-breaker."""
        if remote.timestamp > local.timestamp:
            return remote.copy(), "remote"
        elif local.timestamp > remote.timestamp:
            return local.copy(), "local"

        # Tie-breaker
        if remote.node_id > local.node_id:
            return remote.copy(), "remote"
        else:
            return local.copy(), "local"

    def _resolve_merge(
        self,
        local: Pattern,
        remote: Pattern
    ) -> Tuple[Pattern, str]:
        """Deep merges data dictionaries."""
        merged_data = self._deep_merge(local.data, remote.data)

        resolved = local.copy()
        resolved.data = merged_data
        resolved.timestamp = max(local.timestamp, remote.timestamp)
        resolved.node_id = self.node_id

        return resolved, "merge"

    def _deep_merge(
        self,
        dict1: Dict[str, Any],
        dict2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursive dictionary merge. dict2 overwrites dict1 for conflicts."""
        result = copy.deepcopy(dict1)
        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def get_audit_trail(self) -> List[ConflictRecord]:
        """Returns the history of resolved conflicts."""
        return list(self.audit_trail)


__all__ = [
    "ConflictResolver",
    "ConflictStrategy",
    "VersionVector",
    "Pattern",
    "ConflictRecord",
    "ConflictResolutionError",
]
