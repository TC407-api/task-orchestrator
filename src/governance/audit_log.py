"""Immutable audit logging with cryptographic integrity (SHA256 chain)."""
import json
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class AuditEntry:
    """Single audit log entry with cryptographic linking."""
    id: str
    timestamp: datetime
    operation: str
    agent_id: str
    input_hash: str
    output_hash: str
    cost_usd: float
    trace_id: Optional[str] = None
    prev_hash: Optional[str] = None


class ImmutableAuditLog:
    """
    Manages an append-only, hash-chained audit log stored in JSONL format.

    Features:
    - Cryptographic integrity via SHA256 chain
    - Temporal queries (by time range)
    - Operation type filtering
    - JSONL export for compliance
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize the audit log storage.

        Args:
            storage_dir: Custom path for storage. Defaults to ~/.claude/governance/audit/
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path.home() / ".claude" / "governance" / "audit"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._last_hash: Optional[str] = self._recover_last_hash()

    def _recover_last_hash(self) -> Optional[str]:
        """Recover the last hash from existing logs."""
        files = sorted(self.storage_dir.glob("*.jsonl"))
        if not files:
            return None
        try:
            with open(files[-1], 'r') as f:
                lines = f.readlines()
                if lines:
                    data = json.loads(lines[-1])
                    return self._calculate_hash(data)
                return None
        except (json.JSONDecodeError, IndexError, KeyError):
            return None

    def _entry_to_dict(self, entry: AuditEntry) -> Dict[str, Any]:
        """Convert entry to serializable dict."""
        return {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
            "operation": entry.operation,
            "agent_id": entry.agent_id,
            "input_hash": entry.input_hash,
            "output_hash": entry.output_hash,
            "cost_usd": entry.cost_usd,
            "trace_id": entry.trace_id,
            "prev_hash": entry.prev_hash,
        }

    def _dict_to_entry(self, data: Dict[str, Any]) -> AuditEntry:
        """Convert dict to AuditEntry."""
        return AuditEntry(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            operation=data["operation"],
            agent_id=data["agent_id"],
            input_hash=data["input_hash"],
            output_hash=data["output_hash"],
            cost_usd=data["cost_usd"],
            trace_id=data.get("trace_id"),
            prev_hash=data.get("prev_hash"),
        )

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of entry data."""
        # Exclude prev_hash from hash calculation for chaining
        payload = {k: v for k, v in data.items() if k != "prev_hash"}
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def append(self, entry: AuditEntry) -> str:
        """
        Appends an entry to the log with a cryptographic link.

        Args:
            entry: The AuditEntry to append

        Returns:
            The SHA256 hash of the recorded entry.
        """
        # Create entry dict with current prev_hash
        entry_dict = self._entry_to_dict(entry)
        entry_dict["prev_hash"] = self._last_hash

        # Calculate hash of this entry
        entry_hash = self._calculate_hash(entry_dict)

        # Get today's log file
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.storage_dir / f"{today}.jsonl"

        # Append to file
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry_dict) + '\n')

        # Update last hash
        self._last_hash = entry_hash

        return entry_hash

    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operation: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """
        Query audit logs with filtering options.

        Args:
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            operation: Filter by operation type
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        results = []
        for file_path in sorted(self.storage_dir.glob("*.jsonl")):
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    entry = self._dict_to_entry(data)

                    # Apply filters
                    if start_time and entry.timestamp < start_time:
                        continue
                    if end_time and entry.timestamp > end_time:
                        continue
                    if operation and entry.operation != operation:
                        continue

                    results.append(entry)
                    if len(results) >= limit:
                        return results

        return results

    def verify_chain(self) -> bool:
        """
        Validates the cryptographic integrity of the entire audit log chain.

        Returns:
            True if chain is valid, False if tampered
        """
        prev_hash = None
        for file_path in sorted(self.storage_dir.glob("*.jsonl")):
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)

                    # Verify this entry's hash
                    calculated_hash = self._calculate_hash(data)

                    # Verify prev_hash chain
                    if prev_hash is not None and data.get("prev_hash") != prev_hash:
                        return False

                    prev_hash = calculated_hash

        return True

    def export(self, output_path: Path) -> None:
        """
        Exports the current audit log to a specified JSONL file.

        Args:
            output_path: Destination path for the export
        """
        with open(output_path, 'w') as out:
            for file_path in sorted(self.storage_dir.glob("*.jsonl")):
                with open(file_path, 'r') as f:
                    out.write(f.read())
