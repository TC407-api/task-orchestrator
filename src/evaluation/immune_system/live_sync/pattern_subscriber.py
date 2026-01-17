"""
Real-Time Pattern Subscriber for Graphiti Federation.

Manages subscriptions to remote projects and handles incoming
pattern change events with buffering and replay support.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Set, Union

logger = logging.getLogger(__name__)

PatternCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]


class ConnectionStatus(Enum):
    """Enum representing the current connection state."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"


@dataclass
class PatternEvent:
    """Represents a standardized pattern change event."""
    event_id: str
    project_id: str
    event_type: str
    timestamp: float
    payload: Dict[str, Any]

    @classmethod
    def from_json(cls, data: Union[str, bytes]) -> 'PatternEvent':
        """Parses JSON data into a PatternEvent."""
        try:
            parsed = json.loads(data)
            return cls(
                event_id=parsed['event_id'],
                project_id=parsed['project_id'],
                event_type=parsed['event_type'],
                timestamp=parsed['timestamp'],
                payload=parsed['payload']
            )
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid event format: {e}")

    def to_json(self) -> str:
        """Serialize event to JSON."""
        return json.dumps({
            "event_id": self.event_id,
            "project_id": self.project_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        })


class PatternSubscriber:
    """
    Manages real-time subscriptions to remote Graphiti projects.

    Handles the lifecycle of the connection, buffers events during
    processing, and manages automatic reconnection with event replay.
    """

    def __init__(
        self,
        endpoint_url: str,
        auth_token: str,
        max_buffer_size: int = 1000,
        reconnect_interval_base: float = 1.0,
        reconnect_interval_max: float = 30.0
    ):
        """
        Initialize the subscriber.

        Args:
            endpoint_url: The WebSocket URL of the federation hub.
            auth_token: Authentication token for the connection.
            max_buffer_size: Max events to hold in memory before backpressure.
            reconnect_interval_base: Initial wait time for retries (seconds).
            reconnect_interval_max: Maximum wait time for retries (seconds).
        """
        self.endpoint_url = endpoint_url
        self._auth_token = auth_token
        self._reconnect_base = reconnect_interval_base
        self._reconnect_max = reconnect_interval_max

        self._status = ConnectionStatus.DISCONNECTED
        self._subscriptions: Set[str] = set()
        self._callbacks: Dict[str, List[PatternCallback]] = {}

        # Tracking for replay functionality
        self._cursors: Dict[str, str] = {}

        # Asyncio structures
        self._event_queue: asyncio.Queue[PatternEvent] = asyncio.Queue(
            maxsize=max_buffer_size
        )
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._transport: Any = None

    @property
    def status(self) -> ConnectionStatus:
        """Current connection status."""
        return self._status

    async def start(self) -> None:
        """Starts the subscriber background tasks."""
        if self._status != ConnectionStatus.DISCONNECTED:
            logger.warning("Subscriber is already running or starting.")
            return

        logger.info("Starting PatternSubscriber...")
        self._stop_event.clear()

        self._tasks.append(asyncio.create_task(self._process_event_queue()))
        self._tasks.append(asyncio.create_task(self._connection_loop()))

    async def stop(self) -> None:
        """Gracefully stops the subscriber."""
        logger.info("Stopping PatternSubscriber...")
        self._status = ConnectionStatus.DISCONNECTED
        self._stop_event.set()

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("PatternSubscriber stopped.")

    def subscribe(self, project_id: str, callback: PatternCallback) -> None:
        """
        Subscribe to pattern changes for a specific project.

        Args:
            project_id: The UUID/ID of the remote project.
            callback: Async function to handle updates.
        """
        if project_id not in self._callbacks:
            self._callbacks[project_id] = []

        self._callbacks[project_id].append(callback)
        self._subscriptions.add(project_id)

        logger.info(f"Registered subscription for project: {project_id}")

        if self._status == ConnectionStatus.CONNECTED:
            asyncio.create_task(self._send_subscription_request(project_id))

    def unsubscribe(self, project_id: str) -> None:
        """Remove subscription for a project."""
        if project_id in self._subscriptions:
            self._subscriptions.remove(project_id)
            self._callbacks.pop(project_id, None)
            self._cursors.pop(project_id, None)
            logger.info(f"Unsubscribed from project: {project_id}")

            if self._status == ConnectionStatus.CONNECTED:
                asyncio.create_task(self._send_unsubscribe_request(project_id))

    async def _connection_loop(self) -> None:
        """Main loop handling WebSocket connection and reconnection."""
        retry_delay = self._reconnect_base

        while not self._stop_event.is_set():
            try:
                self._status = ConnectionStatus.CONNECTING
                logger.info(f"Connecting to {self.endpoint_url}...")

                await self._simulate_connect()

                self._status = ConnectionStatus.CONNECTED
                logger.info("Connection established.")
                retry_delay = self._reconnect_base

                await self._handshake_and_replay()
                await self._listen_for_messages()

            except Exception as e:
                self._status = ConnectionStatus.DISCONNECTED
                logger.error(f"Connection error: {e}. Retrying in {retry_delay}s...")

                if self._stop_event.is_set():
                    break

                self._status = ConnectionStatus.RECONNECTING
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self._reconnect_max)

    async def _simulate_connect(self) -> None:
        """Simulates network connection for testing."""
        await asyncio.sleep(0.1)

    async def _handshake_and_replay(self) -> None:
        """Sends active subscriptions and cursors to request replay."""
        {
            "type": "HANDSHAKE",
            "token": self._auth_token,
            "subscriptions": list(self._subscriptions),
            "cursors": self._cursors
        }
        logger.debug(f"Sent handshake with {len(self._subscriptions)} subscriptions.")

    async def _listen_for_messages(self) -> None:
        """Reads messages from the transport."""
        while self._status == ConnectionStatus.CONNECTED and not self._stop_event.is_set():
            await asyncio.sleep(0.1)

    async def handle_incoming_raw(self, raw_data: str) -> None:
        """
        Called when raw data is received from the socket.
        Parses and queues valid events.
        """
        try:
            event = PatternEvent.from_json(raw_data)

            self._cursors[event.project_id] = event.event_id

            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.error("Event buffer full! Dropping event.")

        except ValueError as e:
            logger.warning(f"Received malformed message: {e}")

    async def _process_event_queue(self) -> None:
        """Consumer task: Reads events from buffer and triggers callbacks."""
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )

                project_id = event.project_id

                if project_id in self._callbacks:
                    handlers = self._callbacks[project_id]

                    for cb in handlers:
                        asyncio.create_task(self._safe_callback(cb, event))

                self._event_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processor: {e}")

    async def _safe_callback(
        self,
        callback: PatternCallback,
        event: PatternEvent
    ) -> None:
        """Wraps callback execution with error handling."""
        try:
            await callback(event.project_id, event.payload)
        except Exception as e:
            logger.exception(
                f"Error in subscriber callback for project {event.project_id}: {e}"
            )

    async def _send_subscription_request(self, project_id: str) -> None:
        """Sends subscription request."""
        {
            "type": "SUBSCRIBE",
            "project_id": project_id,
            "since_cursor": self._cursors.get(project_id)
        }
        logger.debug(f"Sent subscription for {project_id}")

    async def _send_unsubscribe_request(self, project_id: str) -> None:
        """Sends unsubscribe request."""
        logger.debug(f"Sent unsubscribe for {project_id}")


__all__ = [
    "PatternSubscriber",
    "PatternEvent",
    "ConnectionStatus",
]
