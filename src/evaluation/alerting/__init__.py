"""
Phase 8.2: Alerting Module for Immune System.

This package provides alerting capabilities for high-risk patterns,
including configurable rules and multiple notification channels.
"""

from .alerts import Alert, AlertSeverity
from .rules import (
    AlertRule,
    AlertContext,
    HighRiskThreshold,
    FrequencySpike,
    NewPatternDetected,
    ConsecutiveFailures,
)
from .notifiers import (
    BaseNotifier,
    ConsoleNotifier,
    WebhookNotifier,
    SlackNotifier,
)
from .manager import AlertManager

__all__ = [
    # Core types
    "Alert",
    "AlertSeverity",
    # Rules
    "AlertRule",
    "AlertContext",
    "HighRiskThreshold",
    "FrequencySpike",
    "NewPatternDetected",
    "ConsecutiveFailures",
    # Notifiers
    "BaseNotifier",
    "ConsoleNotifier",
    "WebhookNotifier",
    "SlackNotifier",
    # Manager
    "AlertManager",
]
