"""
Bidirectional Sync Engine for Cross-Project Graphiti Federation.

Handles push/pull synchronization of failure patterns between
portfolio projects with batch processing and state tracking.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SyncDirection(Enum):
    """Direction of sync operation."""
    PUSH = "push"
    PULL = "pull"


@dataclass
class PatternChange:
    """Represents a single unit of change for a pattern."""
    pattern_id: str
    version: int
    timestamp: float
    data: Dict[str, Any]
    deleted: bool = False


@dataclass
class SyncBatch:
    """A batch of changes to be transmitted over the network."""
    source_project_id: str
    from_version: int
    to_version: int
    changes: List[PatternChange]


@dataclass
class PeerSyncState:
    """Tracks synchronization cursors for a specific peer project."""
    peer_id: str
    last_pulled_version: int = 0
    last_pushed_version: int = 0
    last_sync_time: float = 0.0


class PatternStore(ABC):
    """Interface for the local data storage."""

    @abstractmethod
    def get_current_version(self) -> int:
        """Returns the current global version of the local store."""
        pass

    @abstractmethod
    def get_changes_since(
        self,
        version: int,
        limit: int = 100
    ) -> List[PatternChange]:
        """Fetch local changes after the given version."""
        pass

    @abstractmethod
    def apply_remote_changes(
        self,
        source_project_id: str,
        changes: List[PatternChange]
    ) -> None:
        """Apply changes from a remote project to the local store."""
        pass


class NetworkTransport(ABC):
    """Interface for network communication."""

    @abstractmethod
    def push_batch(self, target_peer_id: str, batch: SyncBatch) -> bool:
        """Send a batch of changes to a peer. Returns True if acknowledged."""
        pass

    @abstractmethod
    def pull_batch(
        self,
        target_peer_id: str,
        last_known_version: int
    ) -> Optional[SyncBatch]:
        """Request changes from a peer since last_known_version."""
        pass


class SyncEngine:
    """
    Bidirectional synchronization engine for Graphiti federation.

    Handles:
    1. Tracking sync cursors for all peers.
    2. Pushing local changes to subscribers.
    3. Pulling remote changes from subscriptions.
    4. Batch processing and error handling.
    """

    def __init__(
        self,
        project_id: str,
        store: Optional[PatternStore] = None,
        transport: Optional[NetworkTransport] = None,
        batch_size: int = 50
    ):
        """
        Initialize the Sync Engine.

        Args:
            project_id: Unique identifier for this local project.
            store: Implementation of PatternStore interface.
            transport: Implementation of NetworkTransport interface.
            batch_size: Max records to sync in one network call.
        """
        self.project_id = project_id
        self.store = store
        self.transport = transport
        self.batch_size = batch_size

        self._sync_states: Dict[str, PeerSyncState] = {}
        self.subscribers: Set[str] = set()
        self.subscriptions: Set[str] = set()

    def register_peer(
        self,
        peer_id: str,
        is_subscriber: bool = False,
        is_subscription: bool = False
    ) -> None:
        """Register a peer project and initialize its sync state."""
        if peer_id not in self._sync_states:
            self._sync_states[peer_id] = PeerSyncState(peer_id=peer_id)

        if is_subscriber:
            self.subscribers.add(peer_id)
        if is_subscription:
            self.subscriptions.add(peer_id)

        logger.info(
            f"Registered peer {peer_id} "
            f"(Subscriber: {is_subscriber}, Subscription: {is_subscription})"
        )

    def get_sync_state(self, peer_id: str) -> Optional[PeerSyncState]:
        """Retrieve current sync state for a peer."""
        return self._sync_states.get(peer_id)

    def trigger_push_sync(self) -> Dict[str, str]:
        """
        Iterate through all subscribers and push pending local changes.

        Returns:
            Dict mapping peer_id to status message.
        """
        results = {}

        if not self.store:
            return {"error": "No store configured"}

        current_local_version = self.store.get_current_version()

        for peer_id in list(self.subscribers):
            try:
                results[peer_id] = self._push_to_peer(
                    peer_id,
                    current_local_version
                )
            except Exception as e:
                logger.error(
                    f"Push sync failed for peer {peer_id}: {e}",
                    exc_info=True
                )
                results[peer_id] = f"Error: {str(e)}"

        return results

    def _push_to_peer(self, peer_id: str, current_local_version: int) -> str:
        """Push changes to a specific peer."""
        state = self._sync_states[peer_id]

        if state.last_pushed_version >= current_local_version:
            return "Up to date"

        if not self.store or not self.transport:
            return "No store/transport"

        changes = self.store.get_changes_since(
            version=state.last_pushed_version,
            limit=self.batch_size
        )

        if not changes:
            state.last_pushed_version = current_local_version
            return "Synced (No data)"

        batch_max_version = max(c.version for c in changes)

        batch = SyncBatch(
            source_project_id=self.project_id,
            from_version=state.last_pushed_version,
            to_version=batch_max_version,
            changes=changes
        )

        logger.debug(
            f"Pushing {len(changes)} changes to {peer_id} "
            f"(v{batch.from_version}->v{batch.to_version})"
        )

        success = self.transport.push_batch(peer_id, batch)

        if success:
            state.last_pushed_version = batch_max_version
            state.last_sync_time = time.time()

            if len(changes) >= self.batch_size:
                return "Partial Sync (Batch limit)"
            return "Success"
        else:
            return "Failed (Network/Ack)"

    def trigger_pull_sync(self) -> Dict[str, str]:
        """
        Iterate through all subscriptions and request remote changes.

        Returns:
            Dict mapping peer_id to status message.
        """
        results = {}

        for peer_id in list(self.subscriptions):
            try:
                results[peer_id] = self._pull_from_peer(peer_id)
            except Exception as e:
                logger.error(
                    f"Pull sync failed for peer {peer_id}: {e}",
                    exc_info=True
                )
                results[peer_id] = f"Error: {str(e)}"

        return results

    def _pull_from_peer(self, peer_id: str) -> str:
        """Pull changes from a specific peer."""
        state = self._sync_states[peer_id]

        if not self.transport or not self.store:
            return "No transport/store"

        logger.debug(
            f"Requesting changes from {peer_id} since v{state.last_pulled_version}"
        )

        batch = self.transport.pull_batch(peer_id, state.last_pulled_version)

        if not batch:
            return "No response"

        if not batch.changes:
            state.last_sync_time = time.time()
            return "Up to date"

        if batch.source_project_id != peer_id:
            logger.warning(
                f"Security warning: Batch from {batch.source_project_id} via {peer_id}"
            )
            return "Security Mismatch"

        try:
            self.store.apply_remote_changes(peer_id, batch.changes)

            new_version = max(
                batch.to_version,
                max(c.version for c in batch.changes)
            )
            state.last_pulled_version = new_version
            state.last_sync_time = time.time()

            return f"Synced {len(batch.changes)} items"
        except Exception as e:
            logger.error(f"Failed to apply remote changes from {peer_id}: {e}")
            raise e

    def sync_all(self) -> Dict[str, Any]:
        """Convenience method for a full bidirectional sync cycle."""
        logger.info("Starting full bidirectional sync cycle...")
        pull_results = self.trigger_pull_sync()
        push_results = self.trigger_push_sync()

        summary = {
            "timestamp": time.time(),
            "pull": pull_results,
            "push": push_results
        }
        logger.info(f"Sync cycle complete. Summary: {summary}")
        return summary


__all__ = [
    "SyncEngine",
    "SyncDirection",
    "SyncBatch",
    "PatternChange",
    "PeerSyncState",
    "PatternStore",
    "NetworkTransport",
]
