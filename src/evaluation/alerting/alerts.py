"""
Core alert data structures for the alerting system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """
    Represents a generated alert from the immune system.

    Attributes:
        severity: The severity level of the alert
        rule_name: Name of the rule that triggered this alert
        message: Human-readable alert message
        timestamp: When the alert was generated
        pattern_id: ID of the related failure pattern (if any)
        metadata: Additional context data
    """
    severity: AlertSeverity
    rule_name: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    pattern_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the alert to a dictionary."""
        return {
            "severity": self.severity.value,
            "rule_name": self.rule_name,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "pattern_id": self.pattern_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """Deserialize an alert from a dictionary."""
        return cls(
            severity=AlertSeverity(data["severity"]),
            rule_name=data["rule_name"],
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            pattern_id=data.get("pattern_id"),
            metadata=data.get("metadata", {}),
        )


__all__ = ["Alert", "AlertSeverity"]
