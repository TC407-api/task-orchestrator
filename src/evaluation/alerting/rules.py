"""
Alert rules for the immune system.

Rules define conditions under which alerts should be generated.
"""

import abc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .alerts import Alert, AlertSeverity


@dataclass
class AlertContext:
    """
    Context passed to rules for evaluation.

    Attributes:
        pattern_id: ID of the failure pattern being evaluated
        risk_score: Current risk score (0.0-1.0)
        is_new_pattern: Whether this is a newly discovered pattern
        failure_history: List of past failures for this pattern
        global_history: Global list of recent failures
    """
    pattern_id: str
    risk_score: float
    is_new_pattern: bool
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    global_history: List[Dict[str, Any]] = field(default_factory=list)


class AlertRule(abc.ABC):
    """Abstract base class for alert rules."""

    def __init__(self, name: str, severity: AlertSeverity):
        """
        Initialize the rule.

        Args:
            name: Unique name for this rule
            severity: Default severity for alerts from this rule
        """
        self.name = name
        self.severity = severity

    @abc.abstractmethod
    def evaluate(self, context: AlertContext) -> Optional[Alert]:
        """
        Evaluate the rule against the given context.

        Args:
            context: The alert context to evaluate

        Returns:
            An Alert if the condition is met, None otherwise
        """
        pass


class HighRiskThreshold(AlertRule):
    """
    Alerts when risk score exceeds a threshold.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
    ):
        super().__init__("HighRiskThreshold", severity)
        self.threshold = threshold

    def evaluate(self, context: AlertContext) -> Optional[Alert]:
        if context.risk_score >= self.threshold:
            return Alert(
                severity=self.severity,
                rule_name=self.name,
                message=f"Risk score {context.risk_score:.2f} exceeds threshold {self.threshold}",
                pattern_id=context.pattern_id,
                metadata={
                    "risk_score": context.risk_score,
                    "threshold": self.threshold,
                },
            )
        return None


class NewPatternDetected(AlertRule):
    """
    Alerts when a completely new failure pattern is seen.
    """

    def __init__(self, severity: AlertSeverity = AlertSeverity.WARNING):
        super().__init__("NewPatternDetected", severity)

    def evaluate(self, context: AlertContext) -> Optional[Alert]:
        if context.is_new_pattern:
            return Alert(
                severity=self.severity,
                rule_name=self.name,
                message=f"New failure pattern detected: {context.pattern_id}",
                pattern_id=context.pattern_id,
            )
        return None


class FrequencySpike(AlertRule):
    """
    Alerts when failure rate exceeds N per hour.
    """

    def __init__(
        self,
        max_per_hour: int = 5,
        severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        super().__init__("FrequencySpike", severity)
        self.max_per_hour = max_per_hour

    def evaluate(self, context: AlertContext) -> Optional[Alert]:
        one_hour_ago = datetime.now() - timedelta(hours=1)

        recent_failures = [
            f for f in context.failure_history
            if isinstance(f.get('timestamp'), datetime) and f['timestamp'] > one_hour_ago
        ]

        # Include the current failure
        count = len(recent_failures) + 1

        if count > self.max_per_hour:
            return Alert(
                severity=self.severity,
                rule_name=self.name,
                message=f"Failure frequency spike: {count} failures in last hour (limit: {self.max_per_hour})",
                pattern_id=context.pattern_id,
                metadata={
                    "count": count,
                    "limit": self.max_per_hour,
                },
            )
        return None


class ConsecutiveFailures(AlertRule):
    """
    Alerts after N consecutive failures.
    """

    def __init__(
        self,
        threshold: int = 3,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
    ):
        super().__init__("ConsecutiveFailures", severity)
        self.threshold = threshold

    def evaluate(self, context: AlertContext) -> Optional[Alert]:
        # Check if we have enough history to evaluate
        if len(context.global_history) < self.threshold - 1:
            return None

        # Check the most recent entries (assuming global_history is newest-first)
        recent = context.global_history[:self.threshold - 1]

        # All recent must be failures (no successes in between)
        # In our system, global_history only contains failures
        if len(recent) >= self.threshold - 1:
            return Alert(
                severity=self.severity,
                rule_name=self.name,
                message=f"{self.threshold} consecutive failures detected",
                pattern_id=context.pattern_id,
                metadata={
                    "threshold": self.threshold,
                    "consecutive_count": len(recent) + 1,
                },
            )
        return None


__all__ = [
    "AlertRule",
    "AlertContext",
    "HighRiskThreshold",
    "NewPatternDetected",
    "FrequencySpike",
    "ConsecutiveFailures",
]
