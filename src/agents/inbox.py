"""
Universal Inbox / Approval Queue for Human-In-The-Loop Agent Actions.

Provides centralized event stream with approval gates for high-risk agent actions.
Creates a "glass box" instead of "black box" for transparent agent operations.

Usage:
    inbox = UniversalInbox()
    await inbox.publish(event)

    async for event in inbox.subscribe():
        handle_event(event)

    await inbox.approve(action_id)
"""
import asyncio
import functools
import json
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, AsyncGenerator, Callable, Optional
from uuid import uuid4


class EventType(str, Enum):
    """Types of agent events."""
    AGENT_START = "AGENT_START"
    AGENT_END = "AGENT_END"
    TOOL_USE = "TOOL_USE"
    TEXT_OUTPUT = "TEXT_OUTPUT"
    ERROR = "ERROR"
    AWAITING_INPUT = "AWAITING_INPUT"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ActionRiskLevel(str, Enum):
    """Risk level of actions requiring approval."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ApprovalStatus(str, Enum):
    """Status of approval requests."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class AgentEvent:
    """
    Event emitted by an agent or system.

    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of event
        agent_name: Name of the agent emitting the event
        timestamp: When the event occurred
        data: Event-specific data
        source: Where the event originated (e.g., 'agent_xyz', 'system')
        related_approval_id: Links to approval action if applicable
        agent_credential_id: Cryptographic agent identity (NHI binding)
        signature: Agent's cryptographic signature of this event
    """
    event_id: str = field(default_factory=lambda: str(uuid4())[:8])
    event_type: EventType = EventType.TEXT_OUTPUT
    agent_name: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)
    source: str = "system"
    related_approval_id: Optional[str] = None
    # Security enhancements for NHI binding
    agent_credential_id: Optional[str] = None
    signature: Optional[str] = None  # Base64-encoded signature

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "related_approval_id": self.related_approval_id,
            "agent_credential_id": self.agent_credential_id,
            "signature": self.signature,
        }

    def get_signable_content(self) -> bytes:
        """Get the content that should be signed for verification."""
        import json
        signable = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }
        return json.dumps(signable, sort_keys=True).encode("utf-8")


@dataclass
class PendingAction:
    """
    Action awaiting user approval.

    Attributes:
        action_id: Unique identifier
        action_type: Type of action (e.g., 'delete_file', 'send_email')
        description: Human-readable description
        risk_level: How risky this action is
        agent_name: Which agent requested the action
        payload: Full data needed to execute action
        status: Current approval status
        created_at: When requested
        expires_at: When approval expires
        approved_at: When approved
        approved_by: User who approved
        rejection_reason: Why it was rejected
        execution_result: Result if executed
    """
    action_id: str = field(default_factory=lambda: str(uuid4())[:8])
    action_type: str = ""
    description: str = ""
    risk_level: ActionRiskLevel = ActionRiskLevel.MEDIUM
    agent_name: str = "unknown"
    payload: dict = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    execution_result: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["event_type"] = self.action_type
        data["risk_level"] = self.risk_level.value
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["expires_at"] = self.expires_at.isoformat()
        data["approved_at"] = self.approved_at.isoformat() if self.approved_at else None
        return data

    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        return datetime.now() > self.expires_at


class UniversalInbox:
    """
    Centralized event inbox with approval queue for agent actions.

    Provides:
    - Event publishing and subscription
    - High-risk action approval gates
    - SQLite persistence
    - Async event streaming
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the inbox.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory.
        """
        self.db_path = db_path or ":memory:"
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: list[asyncio.Queue] = []
        self._db_lock = Lock()
        self._pending_actions: dict[str, PendingAction] = {}
        # Keep persistent connection for in-memory databases
        self._conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, using persistent connection for in-memory."""
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                source TEXT NOT NULL,
                related_approval_id TEXT
            )
            """
        )

        # Pending actions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_actions (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                description TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                approved_at TEXT,
                approved_by TEXT,
                rejection_reason TEXT,
                execution_result TEXT
            )
            """
        )

        # Action history table (for audit trail)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS action_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                old_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                changed_by TEXT,
                notes TEXT
            )
            """
        )

        conn.commit()
        if self._conn is None:
            conn.close()

    async def publish(self, event: AgentEvent) -> None:
        """
        Publish an event to the inbox.

        Args:
            event: Event to publish
        """
        # Store in database
        self._store_event(event)

        # Add to queue for subscribers
        await self._event_queue.put(event)

        # Notify all subscribers
        for subscriber_queue in self._subscribers:
            await subscriber_queue.put(event)

    def _store_event(self, event: AgentEvent) -> None:
        """Store event in SQLite database."""
        with self._db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO events
                    (event_id, event_type, agent_name, timestamp, data,
                     source, related_approval_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.event_type.value,
                        event.agent_name,
                        event.timestamp.isoformat(),
                        json.dumps(event.data),
                        event.source,
                        event.related_approval_id,
                    ),
                )
                conn.commit()
            finally:
                if self._conn is None:
                    conn.close()

    async def subscribe(self) -> AsyncGenerator[AgentEvent, None]:
        """
        Subscribe to events from the inbox.

        Yields:
            AgentEvent objects as they are published
        """
        subscriber_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(subscriber_queue)

        try:
            while True:
                event = await subscriber_queue.get()
                yield event
        finally:
            self._subscribers.remove(subscriber_queue)

    async def require_approval(
        self,
        action_type: str,
        description: str,
        agent_name: str,
        payload: dict,
        risk_level: ActionRiskLevel = ActionRiskLevel.HIGH,
        timeout_seconds: int = 300,
    ) -> PendingAction:
        """
        Create an approval request for a high-risk action.

        Args:
            action_type: Type of action (e.g., 'delete_file')
            description: Human-readable description
            agent_name: Agent requesting the action
            payload: Data needed to execute the action
            risk_level: How risky the action is
            timeout_seconds: How long to wait for approval

        Returns:
            PendingAction object

        Raises:
            TimeoutError: If approval not received within timeout
        """
        expires_at = datetime.now()
        expires_at = expires_at.replace(
            microsecond=int(
                (expires_at.timestamp() + timeout_seconds) * 1e6
            ) % int(1e6)
        )

        action = PendingAction(
            action_type=action_type,
            description=description,
            agent_name=agent_name,
            payload=payload,
            risk_level=risk_level,
            expires_at=datetime.fromtimestamp(
                datetime.now().timestamp() + timeout_seconds
            ),
        )

        # Store in memory and database
        self._pending_actions[action.action_id] = action
        self._store_pending_action(action)

        # Publish approval event
        approval_event = AgentEvent(
            event_type=EventType.APPROVAL_REQUIRED,
            agent_name=agent_name,
            data={
                "action_id": action.action_id,
                "action_type": action_type,
                "description": description,
                "risk_level": risk_level.value,
                "payload": payload,
            },
            source="approval_gate",
        )
        await self.publish(approval_event)

        # Wait for approval
        return action

    def _store_pending_action(self, action: PendingAction) -> None:
        """Store pending action in database."""
        with self._db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO pending_actions
                    (action_id, action_type, description, risk_level,
                     agent_name, payload, status, created_at, expires_at,
                     approved_at, approved_by, rejection_reason, execution_result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        action.action_id,
                        action.action_type,
                        action.description,
                        action.risk_level.value,
                        action.agent_name,
                        json.dumps(action.payload),
                        action.status.value,
                        action.created_at.isoformat(),
                        action.expires_at.isoformat(),
                        action.approved_at.isoformat() if action.approved_at else None,
                        action.approved_by,
                        action.rejection_reason,
                        action.execution_result,
                    ),
                )
                conn.commit()
            finally:
                if self._conn is None:
                    conn.close()

    def get_pending_approvals(
        self,
        risk_level: Optional[ActionRiskLevel] = None,
        agent_name: Optional[str] = None,
    ) -> list[PendingAction]:
        """
        Get list of pending approval requests.

        Args:
            risk_level: Filter by risk level (optional)
            agent_name: Filter by agent name (optional)

        Returns:
            List of PendingAction objects with status PENDING
        """
        pending = [
            a for a in self._pending_actions.values()
            if a.status == ApprovalStatus.PENDING and not a.is_expired()
        ]

        if risk_level:
            pending = [a for a in pending if a.risk_level == risk_level]

        if agent_name:
            pending = [a for a in pending if a.agent_name == agent_name]

        return sorted(pending, key=lambda a: a.created_at, reverse=True)

    async def approve(
        self,
        action_id: str,
        approved_by: str = "system",
        execute_callback: Optional[Callable[[PendingAction], Any]] = None,
    ) -> PendingAction:
        """
        Approve a pending action.

        Args:
            action_id: ID of action to approve
            approved_by: User or system approving the action
            execute_callback: Optional async callback to execute the action

        Returns:
            Updated PendingAction

        Raises:
            ValueError: If action not found or already resolved
        """
        action = self._pending_actions.get(action_id)
        if not action:
            raise ValueError(f"Action not found: {action_id}")

        if action.status != ApprovalStatus.PENDING:
            raise ValueError(f"Action already {action.status.value}: {action_id}")

        if action.is_expired():
            action.status = ApprovalStatus.EXPIRED
            self._update_action_status(action)
            raise ValueError(f"Approval expired for action: {action_id}")

        # Update action
        action.status = ApprovalStatus.APPROVED
        action.approved_at = datetime.now()
        action.approved_by = approved_by

        # Execute callback if provided
        if execute_callback:
            try:
                result = await execute_callback(action) if asyncio.iscoroutinefunction(execute_callback) else execute_callback(action)
                action.execution_result = str(result)
            except Exception as e:
                action.execution_result = f"Error: {str(e)}"

        # Update database
        self._update_action_status(action)

        # Publish approved event
        approved_event = AgentEvent(
            event_type=EventType.APPROVED,
            agent_name=action.agent_name,
            data={
                "action_id": action_id,
                "action_type": action.action_type,
                "approved_by": approved_by,
                "execution_result": action.execution_result,
            },
            source="approval_gate",
            related_approval_id=action_id,
        )
        await self.publish(approved_event)

        return action

    async def reject(
        self,
        action_id: str,
        reason: str = "",
        rejected_by: str = "system",
    ) -> PendingAction:
        """
        Reject a pending action.

        Args:
            action_id: ID of action to reject
            reason: Reason for rejection
            rejected_by: User or system rejecting the action

        Returns:
            Updated PendingAction

        Raises:
            ValueError: If action not found or already resolved
        """
        action = self._pending_actions.get(action_id)
        if not action:
            raise ValueError(f"Action not found: {action_id}")

        if action.status != ApprovalStatus.PENDING:
            raise ValueError(f"Action already {action.status.value}: {action_id}")

        # Update action
        action.status = ApprovalStatus.REJECTED
        action.rejection_reason = reason

        # Update database
        self._update_action_status(action)

        # Publish rejected event
        rejected_event = AgentEvent(
            event_type=EventType.REJECTED,
            agent_name=action.agent_name,
            data={
                "action_id": action_id,
                "action_type": action.action_type,
                "rejection_reason": reason,
                "rejected_by": rejected_by,
            },
            source="approval_gate",
            related_approval_id=action_id,
        )
        await self.publish(rejected_event)

        return action

    def _update_action_status(self, action: PendingAction) -> None:
        """Update action status in database."""
        with self._db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    UPDATE pending_actions
                    SET status = ?, approved_at = ?, approved_by = ?,
                        rejection_reason = ?, execution_result = ?
                    WHERE action_id = ?
                    """,
                    (
                        action.status.value,
                        action.approved_at.isoformat() if action.approved_at else None,
                        action.approved_by,
                        action.rejection_reason,
                        action.execution_result,
                        action.action_id,
                    ),
                )
                conn.commit()
            finally:
                if self._conn is None:
                    conn.close()

    def get_action(self, action_id: str) -> Optional[PendingAction]:
        """Get a specific pending action by ID."""
        return self._pending_actions.get(action_id)

    def get_action_history(
        self,
        action_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get action history.

        Args:
            action_id: Get history for specific action (optional)
            limit: Maximum records to return

        Returns:
            List of history records
        """
        with self._db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                if action_id:
                    cursor.execute(
                        """
                        SELECT action_id, timestamp, old_status, new_status,
                               changed_by, notes
                        FROM action_history
                        WHERE action_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (action_id, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT action_id, timestamp, old_status, new_status,
                               changed_by, notes
                        FROM action_history
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )

                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            finally:
                if self._conn is None:
                    conn.close()

    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        agent_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (optional)
            agent_name: Filter by agent name (optional)
            limit: Maximum records to return

        Returns:
            List of event records
        """
        with self._db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                query = "SELECT * FROM events"
                params = []

                if event_type or agent_name:
                    conditions = []
                    if event_type:
                        conditions.append("event_type = ?")
                        params.append(event_type.value)
                    if agent_name:
                        conditions.append("agent_name = ?")
                        params.append(agent_name)
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            finally:
                if self._conn is None:
                    conn.close()

    async def clear_expired_actions(self) -> int:
        """
        Clear expired approval requests.

        Returns:
            Number of actions cleared
        """
        expired = [
            a for a in self._pending_actions.values()
            if a.is_expired() and a.status == ApprovalStatus.PENDING
        ]

        for action in expired:
            action.status = ApprovalStatus.EXPIRED
            self._update_action_status(action)
            del self._pending_actions[action.action_id]

        return len(expired)

    def export_data(self) -> dict:
        """
        Export all inbox data.

        Returns:
            Dictionary with events and pending actions
        """
        return {
            "pending_actions": [
                a.to_dict() for a in self._pending_actions.values()
            ],
            "events": self.get_event_history(limit=1000),
            "action_history": self.get_action_history(limit=1000),
            "exported_at": datetime.now().isoformat(),
        }


def requires_approval(
    action_type: Optional[str] = None,
    risk_level: ActionRiskLevel = ActionRiskLevel.HIGH,
    timeout_seconds: int = 300,
):
    """
    Decorator to mark a function as requiring approval before execution.

    Usage:
        inbox = UniversalInbox()

        @requires_approval(
            action_type="delete_file",
            risk_level=ActionRiskLevel.CRITICAL
        )
        async def delete_file(path: str, inbox: UniversalInbox):
            await inbox.require_approval(...)
            # File deletion code

    Args:
        action_type: Type of action (auto-generated from function name if not provided)
        risk_level: Risk level of the action
        timeout_seconds: Approval timeout
    """
    def decorator(func: Callable) -> Callable:
        final_action_type = action_type or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Extract inbox from kwargs if available
            inbox = kwargs.get("inbox")
            if not inbox:
                raise ValueError(
                    "@requires_approval decorator requires 'inbox' parameter"
                )

            # Extract agent name from caller context if available
            agent_name = kwargs.get("agent_name", "unknown")

            # Build description
            description = f"{final_action_type}: "
            if args:
                description += f"args={args[:2]}"  # First 2 args
            if kwargs:
                description += f"kwargs={dict(list(kwargs.items())[:2])}"

            # Create approval request
            payload = {
                "args": [str(a) for a in args[:2]],  # Limit size
                "kwargs": {k: str(v) for k, v in list(kwargs.items())[:2]},
            }

            pending_action = await inbox.require_approval(
                action_type=final_action_type,
                description=description,
                agent_name=agent_name,
                payload=payload,
                risk_level=risk_level,
                timeout_seconds=timeout_seconds,
            )

            # Wait for approval
            start_time = datetime.now()
            while pending_action.status == ApprovalStatus.PENDING:
                if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                    raise TimeoutError(
                        f"Approval timeout for {final_action_type}: {pending_action.action_id}"
                    )

                # Refresh action status
                pending_action = inbox.get_action(pending_action.action_id)
                if not pending_action:
                    raise ValueError(f"Action not found: {pending_action.action_id}")

                if pending_action.status == ApprovalStatus.REJECTED:
                    raise PermissionError(
                        f"Action rejected: {pending_action.rejection_reason}"
                    )

                await asyncio.sleep(0.1)

            # Execution is allowed
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Extract inbox from kwargs if available
            inbox = kwargs.get("inbox")
            if not inbox:
                raise ValueError(
                    "@requires_approval decorator requires 'inbox' parameter"
                )

            # For sync functions, just execute (approval is async-only)
            return func(*args, **kwargs)

        # Return async wrapper if function is async, else sync wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
