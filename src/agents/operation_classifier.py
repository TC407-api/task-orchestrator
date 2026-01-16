"""Operation classification for Human-in-the-Loop controls.

Classifies operations as SAFE, REQUIRES_APPROVAL, or BLOCKED based on
their potential for irreversible damage.
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Pattern


class OperationCategory(str, Enum):
    """Classification categories for operations."""
    SAFE = "safe"                     # Auto-execute, read-only operations
    REQUIRES_APPROVAL = "requires_approval"  # Needs human approval
    BLOCKED = "blocked"               # Never execute


@dataclass
class OperationClassification:
    """Result of classifying an operation."""
    category: OperationCategory
    operation: str
    reason: str
    timeout_seconds: int = 300  # Default 5 minute timeout for approvals
    payload_summary: Optional[str] = None


# Patterns that are ALWAYS blocked - never execute these
BLOCKED_PATTERNS: List[Pattern] = [
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"DROP\s+DATABASE", re.IGNORECASE),
    re.compile(r"DROP\s+TABLE\s+\*", re.IGNORECASE),
    re.compile(r"git\s+push\s+.*--force\s+.*main", re.IGNORECASE),
    re.compile(r"git\s+push\s+.*--force\s+.*master", re.IGNORECASE),
    re.compile(r"format\s+[cC]:", re.IGNORECASE),
    re.compile(r"DELETE\s+FROM\s+\w+\s+WHERE\s+1\s*=\s*1", re.IGNORECASE),
    re.compile(r"TRUNCATE\s+TABLE", re.IGNORECASE),
]

# Operations that require approval with their timeout in seconds
APPROVAL_REQUIRED: Dict[str, int] = {
    # File operations
    "file_delete": 300,
    "file_write": 300,
    "directory_delete": 300,

    # Database operations
    "database_write": 300,
    "database_delete": 300,
    "database_migrate": 600,

    # External service operations
    "stripe_charge": 300,
    "stripe_refund": 300,
    "email_send": 300,
    "sms_send": 300,

    # Deployment operations
    "deployment": 600,
    "rollback": 300,

    # Git operations
    "git_push": 300,
    "git_force_push": 600,

    # Cloud operations
    "cloud_delete": 600,
    "cloud_provision": 600,
}

# Safe operations - auto-execute without approval
SAFE_OPERATIONS: List[str] = [
    # Read-only MCP tools
    "tasks_list",
    "inbox_status",
    "cost_summary",
    "healing_status",
    "immune_status",
    "federation_status",
    "sync_status",
    "archetype_info",
    "audit_status",
    "alert_list",

    # Read-only file operations
    "file_read",
    "directory_list",
    "search",
    "grep",

    # Info queries
    "get_status",
    "get_info",
    "list_items",
    "describe",
]


class OperationClassifier:
    """
    Classifies operations based on their risk level.

    SAFE: Read-only operations, auto-executed
    REQUIRES_APPROVAL: Destructive/external operations, needs human approval
    BLOCKED: Dangerous patterns that are never allowed
    """

    def __init__(
        self,
        blocked_patterns: Optional[List[Pattern]] = None,
        approval_required: Optional[Dict[str, int]] = None,
        safe_operations: Optional[List[str]] = None,
    ):
        """
        Initialize the classifier.

        Args:
            blocked_patterns: Regex patterns that are always blocked
            approval_required: Operations requiring approval with timeouts
            safe_operations: Operations that auto-execute
        """
        self._blocked_patterns = blocked_patterns or BLOCKED_PATTERNS
        self._approval_required = approval_required or APPROVAL_REQUIRED
        self._safe_operations = safe_operations or SAFE_OPERATIONS

    def classify(
        self,
        operation: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> OperationClassification:
        """
        Classify an operation.

        Args:
            operation: Name/type of the operation
            payload: Operation parameters

        Returns:
            OperationClassification with category and details
        """
        payload_str = self._payload_to_string(payload)

        # Check if blocked first (highest priority)
        if self.is_blocked(operation, payload):
            return OperationClassification(
                category=OperationCategory.BLOCKED,
                operation=operation,
                reason="Operation matches blocked pattern",
                timeout_seconds=0,
                payload_summary=payload_str[:100] if payload_str else None,
            )

        # Check if it's a known safe operation
        if operation in self._safe_operations:
            return OperationClassification(
                category=OperationCategory.SAFE,
                operation=operation,
                reason="Read-only operation",
                timeout_seconds=300,
                payload_summary=payload_str[:100] if payload_str else None,
            )

        # Check if it requires approval
        if operation in self._approval_required:
            timeout = self._approval_required[operation]
            return OperationClassification(
                category=OperationCategory.REQUIRES_APPROVAL,
                operation=operation,
                reason=f"Operation '{operation}' requires human approval",
                timeout_seconds=timeout,
                payload_summary=payload_str[:100] if payload_str else None,
            )

        # Unknown operations default to requiring approval (safe default)
        return OperationClassification(
            category=OperationCategory.REQUIRES_APPROVAL,
            operation=operation,
            reason="Unknown operation - requires approval by default",
            timeout_seconds=300,
            payload_summary=payload_str[:100] if payload_str else None,
        )

    def _payload_to_string(self, payload: Optional[Dict[str, Any]]) -> str:
        """Convert payload to string for pattern matching."""
        if not payload:
            return ""
        # Combine all string values from payload
        parts = []
        for key, value in payload.items():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                parts.append(self._payload_to_string(value))
        return " ".join(parts)

    def is_blocked(self, operation: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        """Check if operation matches any blocked patterns."""
        # Check payload strings against blocked patterns
        payload_str = self._payload_to_string(payload)
        if payload_str:
            for pattern in self._blocked_patterns:
                if pattern.search(payload_str):
                    return True
        return False

    def get_timeout(self, operation: str) -> int:
        """Get approval timeout for an operation."""
        return self._approval_required.get(operation, 300)
