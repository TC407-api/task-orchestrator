"""
Notification channels for the alerting system.

Notifiers are responsible for delivering alerts to various destinations.
"""

import abc
import logging
from typing import Any, Dict, Optional

from .alerts import Alert

logger = logging.getLogger(__name__)


class BaseNotifier(abc.ABC):
    """Abstract base class for all notifiers."""

    @abc.abstractmethod
    async def notify(self, alert: Alert) -> bool:
        """
        Send notification for an alert.

        Args:
            alert: The alert to notify about

        Returns:
            True if notification was successful, False otherwise
        """
        pass


class ConsoleNotifier(BaseNotifier):
    """
    Logs alerts to the console/logger.

    Useful for development and debugging.
    """

    async def notify(self, alert: Alert) -> bool:
        log_msg = f"[ALERT:{alert.severity.name}] {alert.rule_name}: {alert.message}"

        if alert.severity.name == "CRITICAL":
            logger.error(log_msg)
        elif alert.severity.name == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        return True


class WebhookNotifier(BaseNotifier):
    """
    Sends alerts to a generic webhook URL.

    Uses HTTP POST with JSON payload.
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the webhook notifier.

        Args:
            url: The webhook URL to POST to
            headers: Optional custom headers
        """
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    async def notify(self, alert: Alert) -> bool:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=alert.to_dict(),
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False
                    return True
        except ImportError:
            logger.warning("aiohttp not installed. Using httpx fallback.")
            return await self._notify_httpx(alert)
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

    async def _notify_httpx(self, alert: Alert) -> bool:
        """Fallback using httpx if aiohttp is not available."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json=alert.to_dict(),
                    headers=self.headers,
                    timeout=5.0,
                )
                return response.status_code < 400
        except ImportError:
            logger.error("Neither aiohttp nor httpx is installed for webhook notifications.")
            return False
        except Exception as e:
            logger.error(f"Failed to send webhook via httpx: {e}")
            return False


class SlackNotifier(BaseNotifier):
    """
    Sends formatted alerts to Slack.

    Uses Slack Block Kit formatting for rich messages.
    """

    def __init__(self, webhook_url: str):
        """
        Initialize the Slack notifier.

        Args:
            webhook_url: Slack incoming webhook URL
        """
        self.webhook_url = webhook_url

    async def notify(self, alert: Alert) -> bool:
        # Color mapping for Slack attachments
        color_map = {
            "INFO": "#36a64f",       # Green
            "WARNING": "#ffcc00",     # Yellow
            "CRITICAL": "#ff0000",    # Red
        }

        payload = {
            "attachments": [
                {
                    "color": color_map.get(alert.severity.name, "#cccccc"),
                    "pretext": f"Immune System Alert: {alert.rule_name}",
                    "title": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.name, "short": True},
                        {"title": "Pattern ID", "value": str(alert.pattern_id or "N/A"), "short": True},
                        {"title": "Time", "value": alert.timestamp.isoformat(), "short": False},
                    ],
                    "footer": "Task Orchestrator Immune System",
                }
            ]
        }

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    return resp.status == 200
        except ImportError:
            # Try httpx fallback
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.webhook_url,
                        json=payload,
                        timeout=5.0,
                    )
                    return response.status_code == 200
            except ImportError:
                logger.error("Neither aiohttp nor httpx is installed for Slack notifications.")
                return False
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False


__all__ = [
    "BaseNotifier",
    "ConsoleNotifier",
    "WebhookNotifier",
    "SlackNotifier",
]
