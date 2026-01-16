"""Persistent session context across terminal restarts.

Enhanced with security fields for agent identity binding and risk scoring.
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.governance.agent_identity import AgentCredential
    from src.governance.session_risk_scoring import SessionRiskScore


@dataclass
class SessionContext:
    """
    Data model representing a persistent working session.

    Attributes:
        session_id: Unique identifier for the session
        created_at: When the session was created
        last_active: Last activity timestamp
        files_accessed: Set of file paths accessed during session
        conversation_summary: Summary of conversation history
        token_usage: Total tokens used in session
        agent_credential_id: Bound agent credential ID (NHI)
        risk_score: Most recent risk assessment
        env_fingerprint: Environment fingerprint for device binding
        integrity_chain: Chain of action hashes for audit trail
    """
    session_id: str
    created_at: datetime
    last_active: datetime
    files_accessed: Set[str] = field(default_factory=set)
    conversation_summary: List[Dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    # Security enhancements
    agent_credential_id: Optional[str] = None
    risk_score: Optional[float] = None
    env_fingerprint: Optional[str] = None
    integrity_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "files_accessed": list(self.files_accessed),
            "conversation_summary": self.conversation_summary,
            "token_usage": self.token_usage,
            "agent_credential_id": self.agent_credential_id,
            "risk_score": self.risk_score,
            "env_fingerprint": self.env_fingerprint,
            "integrity_chain": self.integrity_chain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        """Create from dict."""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            files_accessed=set(data.get("files_accessed", [])),
            conversation_summary=data.get("conversation_summary", []),
            token_usage=data.get("token_usage", 0),
            agent_credential_id=data.get("agent_credential_id"),
            risk_score=data.get("risk_score"),
            env_fingerprint=data.get("env_fingerprint"),
            integrity_chain=data.get("integrity_chain", []),
        )

    def bind_agent(self, credential_id: str) -> None:
        """Bind this session to an agent credential."""
        self.agent_credential_id = credential_id

    def update_risk_score(self, score: float) -> None:
        """Update the session's risk score."""
        self.risk_score = score

    def set_fingerprint(self, fingerprint: str) -> None:
        """Set the environment fingerprint."""
        self.env_fingerprint = fingerprint

    def add_to_integrity_chain(self, action_hash: str) -> None:
        """Add an action hash to the integrity chain."""
        self.integrity_chain.append(action_hash)


class SessionManager:
    """
    Manages the lifecycle, persistence, and cleanup of SessionContexts.

    Storage format: ~/.claude/sessions/{session_id}.json
    """

    def __init__(self, storage_path: Path, timeout_hours: float = 24.0):
        """
        Initialize the session manager.

        Args:
            storage_path: Directory where JSON sessions are stored.
            timeout_hours: Hours of inactivity before a session is considered expired.
        """
        self.storage_path = Path(storage_path)
        self.timeout_hours = timeout_hours
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.storage_path / f"{session_id}.json"

    def _save_session(self, session: SessionContext) -> None:
        """Save session to disk."""
        path = self._get_session_path(session.session_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2)

    def create_session(self) -> SessionContext:
        """
        Creates a new session, saves it to disk, and returns the context.

        Returns:
            New SessionContext object
        """
        now = datetime.now()
        session = SessionContext(
            session_id=str(uuid.uuid4()),
            created_at=now,
            last_active=now,
        )
        self._save_session(session)
        return session

    def restore_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Attempts to load a session from disk by ID.

        Args:
            session_id: The session ID to restore

        Returns:
            SessionContext if found, None otherwise
        """
        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return SessionContext.from_dict(data)
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def update_context(
        self,
        session_id: str,
        files: Optional[List[str]] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """
        Updates an existing session with new file accesses or conversation history.
        Updates 'last_active' timestamp.

        Args:
            session_id: The session to update
            files: List of file paths accessed
            messages: List of conversation messages
        """
        session = self.restore_session(session_id)
        if not session:
            return

        # Update files
        if files:
            session.files_accessed.update(files)

        # Update conversation
        if messages:
            session.conversation_summary.extend(messages)

        # Update timestamp
        session = SessionContext(
            session_id=session.session_id,
            created_at=session.created_at,
            last_active=datetime.now(),
            files_accessed=session.files_accessed,
            conversation_summary=session.conversation_summary,
            token_usage=session.token_usage,
        )

        self._save_session(session)

    def cleanup_expired(self) -> int:
        """
        Scans storage path for sessions older than timeout_hours.
        Deletes them.

        Returns:
            int: Number of sessions deleted.
        """
        deleted_count = 0
        cutoff = datetime.now() - timedelta(hours=self.timeout_hours)

        for path in self.storage_path.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                last_active = datetime.fromisoformat(data["last_active"])

                if last_active < cutoff:
                    path.unlink()
                    deleted_count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        return deleted_count
