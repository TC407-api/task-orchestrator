"""
Live Sync Module for Cross-Project Graphiti Federation.

This module provides real-time pattern synchronization across
portfolio projects using WebSocket connections, conflict resolution,
and health monitoring.

Components:
- sync_protocol: WebSocket-based sync protocol with heartbeats
- pattern_subscriber: Real-time pattern subscription manager
- sync_engine: Bidirectional push/pull sync engine
- conflict_resolver: Version vector conflict resolution
- sync_hooks: Middleware-style hook system for sync events
- sync_monitor: Health monitoring and alerting
"""

from .sync_protocol import (
    SyncEventType,
    SyncMessage,
    SyncPayload,
    BackoffStrategy,
    PatternSyncClient,
)
from .pattern_subscriber import (
    PatternSubscriber,
    PatternEvent,
    ConnectionStatus,
)
from .sync_engine import (
    SyncEngine,
    SyncDirection,
    SyncBatch,
    PatternChange,
    PeerSyncState,
)
from .conflict_resolver import (
    ConflictResolver,
    ConflictStrategy,
    VersionVector,
    Pattern,
    ConflictRecord,
)
from .sync_hooks import (
    SyncHooks,
    SyncContext,
    HookEventType,
)
from .sync_monitor import (
    SyncHealthMonitor,
    SyncStatus,
    SyncAlert,
    ProjectSyncState,
)

__all__ = [
    # Protocol
    "SyncEventType",
    "SyncMessage",
    "SyncPayload",
    "BackoffStrategy",
    "PatternSyncClient",
    # Subscriber
    "PatternSubscriber",
    "PatternEvent",
    "ConnectionStatus",
    # Engine
    "SyncEngine",
    "SyncDirection",
    "SyncBatch",
    "PatternChange",
    "PeerSyncState",
    # Conflict
    "ConflictResolver",
    "ConflictStrategy",
    "VersionVector",
    "Pattern",
    "ConflictRecord",
    # Hooks
    "SyncHooks",
    "SyncContext",
    "HookEventType",
    # Monitor
    "SyncHealthMonitor",
    "SyncStatus",
    "SyncAlert",
    "ProjectSyncState",
]
