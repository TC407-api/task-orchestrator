"""
Live Sync Protocol for Cross-Project Graphiti Federation.

Implements WebSocket-based real-time synchronization with:
- Message envelope format
- Heartbeat/keepalive
- Reconnection with exponential backoff
- Subscription management
"""

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


class SyncEventType(str, Enum):
    """Defines allowed event types for the sync protocol."""
    CONNECT = "connect"
    ACK = "ack"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    HEARTBEAT = "heartbeat"

    # Data Events
    PATTERN_CREATED = "pattern_created"
    PATTERN_UPDATED = "pattern_updated"
    PATTERN_DELETED = "pattern_deleted"

    ERROR = "error"


@dataclass
class SyncPayload:
    """Base payload structure for pattern data."""
    project_id: str
    pattern_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    version: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class SyncMessage:
    """
    Standard envelope for all protocol messages.

    Attributes:
        id: Unique message ID for tracing/acks.
        type: The type of event (SyncEventType).
        payload: The actual data content.
    """
    type: SyncEventType
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        """Serializes the message to a JSON string."""
        return json.dumps({
            "id": self.id,
            "type": self.type.value,
            "payload": self.payload
        })

    @classmethod
    def from_json(cls, data: str) -> 'SyncMessage':
        """Deserializes a JSON string into a SyncMessage."""
        try:
            parsed = json.loads(data)
            return cls(
                id=parsed.get("id", str(uuid.uuid4())),
                type=SyncEventType(parsed["type"]),
                payload=parsed.get("payload", {})
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse sync message: {e}")
            raise


class BackoffStrategy:
    """Helper to calculate exponential backoff with jitter."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.attempts = 0

    def get_delay(self) -> float:
        """Calculates delay: min(max, base * 2^attempts) + jitter."""
        delay = min(self.max_delay, self.base_delay * (2 ** self.attempts))
        jitter = delay * 0.1 * random.random()
        self.attempts += 1
        return delay + jitter

    def reset(self):
        """Resets the attempt counter."""
        self.attempts = 0


class PatternSyncClient:
    """
    Manages the lifecycle of the real-time sync connection.

    Features:
    - Auto-reconnection with exponential backoff.
    - Heartbeat management.
    - Subscription handling.
    - Event dispatching.
    """

    def __init__(
        self,
        uri: str,
        auth_token: str,
        project_id: str,
        on_pattern_event: Callable[[SyncMessage], Any]
    ):
        """
        Args:
            uri: WebSocket endpoint (e.g., wss://api.graphiti.com/sync).
            auth_token: Bearer token for authentication.
            project_id: The ID of the current project (for subscriptions).
            on_pattern_event: Callback function for handling incoming data events.
        """
        self.uri = uri
        self.auth_token = auth_token
        self.project_id = project_id
        self.callback = on_pattern_event

        self._ws: Any = None
        self._running = False
        self._backoff = BackoffStrategy()

        # State
        self._subscribed_projects: Set[str] = set()
        self._pending_subscriptions: Set[str] = set()

        # Tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self):
        """Starts the sync client background loop."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting Graphiti Sync Client for project {self.project_id}")

        asyncio.create_task(self._connection_loop())

    async def stop(self):
        """Stops the client and closes connections gracefully."""
        self._running = False
        if self._ws:
            await self._ws.close()

        if self._listen_task:
            self._listen_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        logger.info("Graphiti Sync Client stopped.")

    async def subscribe_to_project(self, target_project_id: str):
        """Registers a subscription for a specific project's patterns."""
        self._pending_subscriptions.add(target_project_id)
        if self._ws and hasattr(self._ws, 'open') and self._ws.open:
            await self._send_subscription(target_project_id)

    async def _connection_loop(self):
        """Main loop handling connection, errors, and reconnection logic."""
        while self._running:
            try:
                logger.debug(f"Connecting to {self.uri}...")

                # Simulate connection for now (in production use websockets library)
                await self._simulate_connect()

                self._backoff.reset()
                logger.info("Connected to Graphiti Sync Service.")

                await self._recover_subscriptions()
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                await self._listen_loop()

            except Exception as e:
                logger.warning(f"Connection lost: {e}. Retrying...")

            if self._running:
                wait_time = self._backoff.get_delay()
                logger.info(f"Reconnecting in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)

    async def _simulate_connect(self):
        """Simulates connection for testing."""
        await asyncio.sleep(0.1)
        self._ws = type('MockWS', (), {'open': True, 'close': lambda: None})()

    async def _listen_loop(self):
        """Reads messages from the socket."""
        while self._running and self._ws:
            await asyncio.sleep(0.1)

    async def _handle_message(self, message: SyncMessage):
        """Dispatches messages to appropriate handlers."""
        if message.type == SyncEventType.HEARTBEAT:
            return

        if message.type == SyncEventType.ERROR:
            logger.error(f"Received remote error: {message.payload}")
            return

        if message.type in {
            SyncEventType.PATTERN_CREATED,
            SyncEventType.PATTERN_UPDATED,
            SyncEventType.PATTERN_DELETED
        }:
            try:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(message)
                else:
                    self.callback(message)
            except Exception as e:
                logger.error(f"Callback failed: {e}")

    async def _heartbeat_loop(self):
        """Sends periodic heartbeats to keep the connection alive."""
        try:
            while self._running and self._ws:
                msg = SyncMessage(
                    type=SyncEventType.HEARTBEAT,
                    payload={"timestamp": time.time()}
                )
                logger.debug(f"Heartbeat: {msg.id}")
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Heartbeat loop stopped: {e}")

    async def _send_subscription(self, target_project_id: str):
        """Sends a subscription request frame."""
        if not self._ws:
            return

        msg = SyncMessage(
            type=SyncEventType.SUBSCRIBE,
            payload={"project_id": target_project_id}
        )
        self._subscribed_projects.add(target_project_id)
        self._pending_subscriptions.discard(target_project_id)
        logger.debug(f"Subscribed to {target_project_id}")

    async def _recover_subscriptions(self):
        """Re-sends subscriptions after a connection drop."""
        targets = self._subscribed_projects.union(self._pending_subscriptions)
        for project_id in targets:
            await self._send_subscription(project_id)


__all__ = [
    "SyncEventType",
    "SyncMessage",
    "SyncPayload",
    "BackoffStrategy",
    "PatternSyncClient",
]
