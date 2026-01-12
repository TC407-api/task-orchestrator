"""
Alert Manager for coordinating rules and notifiers.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .alerts import Alert
from .rules import AlertRule, AlertContext
from .notifiers import BaseNotifier

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alert rules and distributes generated alerts to notifiers.

    The AlertManager is the central coordination point for the alerting
    system. It evaluates incoming failures against all configured rules
    and dispatches alerts to all registered notifiers.
    """

    def __init__(
        self,
        rules: Optional[List[AlertRule]] = None,
        notifiers: Optional[List[BaseNotifier]] = None,
        max_active_alerts: int = 100,
    ):
        """
        Initialize the alert manager.

        Args:
            rules: Initial list of alert rules
            notifiers: Initial list of notifiers
            max_active_alerts: Maximum number of active alerts to keep
        """
        self.rules: List[AlertRule] = rules or []
        self.notifiers: List[BaseNotifier] = notifiers or []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self._max_active_alerts = max_active_alerts
        self._stats = {
            "alerts_generated": 0,
            "notifications_sent": 0,
            "notification_failures": 0,
        }

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules.append(rule)
        logger.debug(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                logger.debug(f"Removed alert rule: {rule_name}")
                return True
        return False

    def add_notifier(self, notifier: BaseNotifier) -> None:
        """Add a notifier."""
        self.notifiers.append(notifier)
        logger.debug(f"Added notifier: {notifier.__class__.__name__}")

    async def process_failure(
        self,
        pattern_id: str,
        risk_score: float,
        is_new_pattern: bool,
        pattern_history: Optional[List[Dict[str, Any]]] = None,
        global_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Alert]:
        """
        Evaluate a failure against all rules and dispatch alerts.

        Args:
            pattern_id: ID of the failure pattern
            risk_score: Current risk score
            is_new_pattern: Whether this is a new pattern
            pattern_history: History for this specific pattern
            global_history: Global failure history

        Returns:
            List of generated alerts
        """
        context = AlertContext(
            pattern_id=pattern_id,
            risk_score=risk_score,
            is_new_pattern=is_new_pattern,
            failure_history=pattern_history or [],
            global_history=global_history or [],
        )

        generated_alerts: List[Alert] = []

        for rule in self.rules:
            try:
                alert = rule.evaluate(context)
                if alert:
                    generated_alerts.append(alert)
                    self._stats["alerts_generated"] += 1
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

        if generated_alerts:
            await self._dispatch_alerts(generated_alerts)

        return generated_alerts

    async def _dispatch_alerts(self, alerts: List[Alert]) -> None:
        """Internal method to save and send alerts."""
        for alert in alerts:
            # Save to history
            self.alert_history.append(alert)

            # Manage active alerts (FIFO)
            self.active_alerts.append(alert)
            if len(self.active_alerts) > self._max_active_alerts:
                self.active_alerts.pop(0)

            # Notify asynchronously
            if self.notifiers:
                # Create notification tasks
                tasks = [notifier.notify(alert) for notifier in self.notifiers]
                # Fire and forget - don't block on notifications
                asyncio.create_task(self._notify_all(tasks))

    async def _notify_all(self, tasks: List[Any]) -> None:
        """Execute all notification tasks with error handling."""
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Notification failed: {result}")
                self._stats["notification_failures"] += 1
            elif result is True:
                self._stats["notifications_sent"] += 1
            else:
                self._stats["notification_failures"] += 1

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent alerts for API/Tools.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries (newest first)
        """
        return [a.to_dict() for a in reversed(self.alert_history[-limit:])]

    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get currently active alerts.

        Args:
            severity: Filter by severity level (optional)

        Returns:
            List of alert dictionaries
        """
        alerts = self.active_alerts
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity.lower()]
        return [a.to_dict() for a in alerts]

    def clear_active_alerts(self) -> int:
        """
        Clear all active alerts.

        Returns:
            Number of alerts cleared
        """
        count = len(self.active_alerts)
        self.active_alerts.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        return {
            **self._stats,
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "rules_count": len(self.rules),
            "notifiers_count": len(self.notifiers),
        }


__all__ = ["AlertManager"]
