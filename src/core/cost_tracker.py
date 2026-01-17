"""API cost tracking with budgets, alerts, and circuit breakers.

Includes 4-state governance model for cost protection:
- NORMAL: Full operation, all API calls allowed
- WARN: Alerts triggered, monitoring increased
- THROTTLE: Rate limiting applied, non-essential calls blocked
- SHUTDOWN: All paid API calls blocked, notifications sent
"""
import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported API providers."""
    GOOGLE_GEMINI = "google_gemini"
    GOOGLE_GMAIL = "google_gmail"
    GOOGLE_CALENDAR = "google_calendar"
    OPENAI = "openai"
    GRAPHITI = "graphiti"
    ANTHROPIC = "anthropic"


class GovernanceState(str, Enum):
    """
    4-state governance model for cost protection.

    State transitions based on budget usage:
    - NORMAL (0-60%): Full operation
    - WARN (60-80%): Alerts triggered, monitoring increased
    - THROTTLE (80-95%): Rate limiting, non-essential blocked
    - SHUTDOWN (95%+): All paid calls blocked, notifications sent
    """
    NORMAL = "NORMAL"
    WARN = "WARN"
    THROTTLE = "THROTTLE"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class GovernanceConfig:
    """Configuration for 4-state cost governance."""
    warn_threshold_pct: float = 0.60      # Enter WARN at 60%
    throttle_threshold_pct: float = 0.80  # Enter THROTTLE at 80%
    shutdown_threshold_pct: float = 0.95  # Enter SHUTDOWN at 95%

    # TTL/cooldown settings for auto-recovery
    cooldown_seconds: float = 300.0       # 5 minutes before state can improve
    auto_recover_on_new_day: bool = True  # Reset to NORMAL on new day

    # Notification settings
    voice_notify_enabled: bool = True     # Voice notification for SHUTDOWN
    webhook_url: Optional[str] = None     # Webhook URL for notifications
    slack_webhook_url: Optional[str] = None  # Slack webhook for alerts


@dataclass
class UsageRecord:
    """Single API usage record."""
    provider: Provider
    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    model: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider.value,
            "operation": self.operation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "metadata": self.metadata,
        }


@dataclass
class BudgetConfig:
    """Budget configuration per provider."""
    daily_limit_usd: float = 5.0
    monthly_limit_usd: float = 50.0
    alert_threshold_pct: float = 0.8  # Alert at 80%
    hard_stop_enabled: bool = True  # Stop API calls when limit hit


# Default conservative budgets (updated Dec 2025)
# Strategy: Lean on Anthropic Max Plan (free), use Gemini 3 Preview (free)
# Only Gemini paid fallback needs real budget
DEFAULT_BUDGETS = {
    Provider.GOOGLE_GEMINI: BudgetConfig(daily_limit_usd=1.0, monthly_limit_usd=20.0),  # Conservative
    Provider.GOOGLE_GMAIL: BudgetConfig(daily_limit_usd=0.25, monthly_limit_usd=5.0),
    Provider.GOOGLE_CALENDAR: BudgetConfig(daily_limit_usd=0.25, monthly_limit_usd=5.0),
    Provider.OPENAI: BudgetConfig(daily_limit_usd=0.50, monthly_limit_usd=10.0),  # For Graphiti embeddings
    Provider.GRAPHITI: BudgetConfig(daily_limit_usd=0.50, monthly_limit_usd=10.0),
    Provider.ANTHROPIC: BudgetConfig(daily_limit_usd=0.0, monthly_limit_usd=0.0),  # Max plan = free
}

@dataclass
class GovernanceStateRecord:
    """Persistent state record for cost governance."""
    state: GovernanceState = GovernanceState.NORMAL
    last_transition: datetime = field(default_factory=datetime.now)
    last_notification: Optional[datetime] = None
    reason: str = ""
    daily_spend_at_transition: float = 0.0
    monthly_spend_at_transition: float = 0.0


class CostGovernor:
    """
    4-state cost governance with TTL/cooldown and notifications.

    States:
        NORMAL -> WARN -> THROTTLE -> SHUTDOWN

    Auto-recovery via cooldown or new day boundary.
    Voice/webhook notifications on SHUTDOWN.
    """

    def __init__(
        self,
        config: Optional[GovernanceConfig] = None,
        state_path: Optional[Path] = None,
    ):
        self.config = config or GovernanceConfig()
        self.state_path = state_path or Path.home() / ".claude" / "grade5" / "cost-governance.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self._state_record = self._load_state()
        self._notification_callbacks: list[Callable] = []

    def _load_state(self) -> GovernanceStateRecord:
        """Load persisted governance state."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                return GovernanceStateRecord(
                    state=GovernanceState(data.get("state", "NORMAL")),
                    last_transition=datetime.fromisoformat(data["last_transition"]),
                    last_notification=datetime.fromisoformat(data["last_notification"]) if data.get("last_notification") else None,
                    reason=data.get("reason", ""),
                    daily_spend_at_transition=data.get("daily_spend_at_transition", 0.0),
                    monthly_spend_at_transition=data.get("monthly_spend_at_transition", 0.0),
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load governance state: {e}")
        return GovernanceStateRecord()

    def _save_state(self):
        """Persist governance state (non-blocking via thread pool)."""
        data = {
            "state": self._state_record.state.value,
            "last_transition": self._state_record.last_transition.isoformat(),
            "last_notification": self._state_record.last_notification.isoformat() if self._state_record.last_notification else None,
            "reason": self._state_record.reason,
            "daily_spend_at_transition": self._state_record.daily_spend_at_transition,
            "monthly_spend_at_transition": self._state_record.monthly_spend_at_transition,
        }
        # PERF: Use thread pool to avoid blocking event loop
        self._write_file_async(self.state_path, json.dumps(data, indent=2))

    def _write_file_async(self, path: Path, content: str):
        """Write file in background thread to avoid blocking event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, path.write_text, content)
        except RuntimeError:
            # No event loop running, write synchronously
            path.write_text(content)

    @property
    def current_state(self) -> GovernanceState:
        """Get current governance state."""
        return self._state_record.state

    def evaluate_state(
        self,
        daily_spend: float,
        daily_limit: float,
        monthly_spend: float,
        monthly_limit: float,
    ) -> GovernanceState:
        """
        Evaluate and transition governance state based on budget usage.

        Returns the new state (may be same as current).
        """
        # Check for auto-recovery on new day
        if self.config.auto_recover_on_new_day:
            if self._state_record.last_transition.date() < datetime.now().date():
                if self._state_record.state != GovernanceState.NORMAL:
                    self._transition_to(
                        GovernanceState.NORMAL,
                        "New day - auto-recovery",
                        daily_spend,
                        monthly_spend,
                    )
                    return self._state_record.state

        # Calculate usage percentages (use higher of daily/monthly)
        daily_pct = (daily_spend / daily_limit) if daily_limit > 0 else 0
        monthly_pct = (monthly_spend / monthly_limit) if monthly_limit > 0 else 0
        usage_pct = max(daily_pct, monthly_pct)

        # Determine target state based on thresholds
        if usage_pct >= self.config.shutdown_threshold_pct:
            target_state = GovernanceState.SHUTDOWN
        elif usage_pct >= self.config.throttle_threshold_pct:
            target_state = GovernanceState.THROTTLE
        elif usage_pct >= self.config.warn_threshold_pct:
            target_state = GovernanceState.WARN
        else:
            target_state = GovernanceState.NORMAL

        current = self._state_record.state

        # State can only worsen immediately, improvement requires cooldown
        state_order = [GovernanceState.NORMAL, GovernanceState.WARN, GovernanceState.THROTTLE, GovernanceState.SHUTDOWN]
        current_idx = state_order.index(current)
        target_idx = state_order.index(target_state)

        if target_idx > current_idx:
            # Worsening - transition immediately
            reason = f"Budget at {usage_pct*100:.1f}% (daily: ${daily_spend:.2f}, monthly: ${monthly_spend:.2f})"
            self._transition_to(target_state, reason, daily_spend, monthly_spend)
        elif target_idx < current_idx:
            # Improving - check cooldown
            elapsed = (datetime.now() - self._state_record.last_transition).total_seconds()
            if elapsed >= self.config.cooldown_seconds:
                reason = f"Budget recovered to {usage_pct*100:.1f}% after cooldown"
                self._transition_to(target_state, reason, daily_spend, monthly_spend)

        return self._state_record.state

    def _transition_to(
        self,
        new_state: GovernanceState,
        reason: str,
        daily_spend: float,
        monthly_spend: float,
    ):
        """Transition to a new governance state."""
        old_state = self._state_record.state

        self._state_record.state = new_state
        self._state_record.last_transition = datetime.now()
        self._state_record.reason = reason
        self._state_record.daily_spend_at_transition = daily_spend
        self._state_record.monthly_spend_at_transition = monthly_spend

        self._save_state()

        logger.info(f"Cost governance: {old_state.value} -> {new_state.value}: {reason}")

        # Send notifications for critical transitions
        if new_state == GovernanceState.SHUTDOWN:
            self._schedule_notification(reason)
        elif new_state == GovernanceState.THROTTLE and old_state != GovernanceState.SHUTDOWN:
            logger.warning(f"THROTTLE MODE: {reason}")

    def _schedule_notification(self, reason: str):
        """Schedule shutdown notification, handling missing event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._notify_shutdown(reason))
        except RuntimeError:
            # No running event loop - run synchronously in a new loop
            try:
                asyncio.run(self._notify_shutdown(reason))
            except Exception as e:
                # If async fails completely, at least send voice notification synchronously
                logger.error(f"Async notification failed: {e}")
                self._send_voice_notification_sync(reason)

    def _send_voice_notification_sync(self, message: str):
        """Synchronous fallback for voice notification."""
        script_path = Path.home() / ".claude" / "scripts" / "voice-notify.ps1"
        if not script_path.exists():
            return

        try:
            subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", str(script_path),
                    "-Message", message,
                    "-Force",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(f"Sync voice notification failed: {e}")

    async def _notify_shutdown(self, reason: str):
        """Send voice and webhook notifications for SHUTDOWN state."""
        # Rate limit notifications (max once per 5 minutes)
        if self._state_record.last_notification:
            elapsed = (datetime.now() - self._state_record.last_notification).total_seconds()
            if elapsed < 300:
                return

        self._state_record.last_notification = datetime.now()
        self._save_state()

        message = f"Cost governance shutdown activated. {reason}"

        # Voice notification via PowerShell script
        if self.config.voice_notify_enabled:
            await self._send_voice_notification(message)

        # Webhook notification
        if self.config.webhook_url:
            await self._send_webhook_notification(message)

        # Slack notification
        if self.config.slack_webhook_url:
            await self._send_slack_notification(message)

        # Custom callbacks
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

    async def _send_voice_notification(self, message: str):
        """Send voice notification via ElevenLabs/Windows TTS."""
        script_path = Path.home() / ".claude" / "scripts" / "voice-notify.ps1"
        if not script_path.exists():
            logger.warning("Voice notify script not found")
            return

        try:
            # Run PowerShell script in background (non-blocking)
            subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", str(script_path),
                    "-Message", message,
                    "-Force",  # Skip rate limit for critical alerts
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Voice notification sent")
        except Exception as e:
            logger.error(f"Voice notification failed: {e}")

    async def _send_webhook_notification(self, message: str):
        """Send webhook notification."""
        if not self.config.webhook_url:
            return

        payload = {
            "event": "cost_governance_shutdown",
            "state": self._state_record.state.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "daily_spend": self._state_record.daily_spend_at_transition,
            "monthly_spend": self._state_record.monthly_spend_at_transition,
        }

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=5.0,
                )
            logger.info("Webhook notification sent")
        except ImportError:
            logger.warning("httpx not installed for webhook notifications")
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")

    async def _send_slack_notification(self, message: str):
        """Send Slack notification for SHUTDOWN."""
        if not self.config.slack_webhook_url:
            return

        payload = {
            "attachments": [
                {
                    "color": "#ff0000",  # Red for critical
                    "pretext": "🚨 COST GOVERNANCE SHUTDOWN",
                    "title": "API Spending Limit Reached",
                    "text": message,
                    "fields": [
                        {"title": "State", "value": "SHUTDOWN", "short": True},
                        {"title": "Daily Spend", "value": f"${self._state_record.daily_spend_at_transition:.2f}", "short": True},
                        {"title": "Monthly Spend", "value": f"${self._state_record.monthly_spend_at_transition:.2f}", "short": True},
                    ],
                    "footer": "Task Orchestrator Cost Governance",
                    "ts": int(datetime.now().timestamp()),
                }
            ]
        }

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=5.0,
                )
            logger.info("Slack notification sent")
        except ImportError:
            logger.warning("httpx not installed for Slack notifications")
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")

    def add_notification_callback(self, callback: Callable):
        """Add a custom notification callback for SHUTDOWN events."""
        self._notification_callbacks.append(callback)

    def can_proceed(self, is_essential: bool = False) -> tuple[bool, str]:
        """
        Check if an API call should proceed based on governance state.

        Args:
            is_essential: If True, call is allowed in THROTTLE but not SHUTDOWN

        Returns:
            Tuple of (can_proceed, reason)
        """
        state = self._state_record.state

        if state == GovernanceState.SHUTDOWN:
            return False, f"SHUTDOWN: {self._state_record.reason}"

        if state == GovernanceState.THROTTLE and not is_essential:
            return False, f"THROTTLE: Non-essential calls blocked. {self._state_record.reason}"

        return True, ""

    def get_status(self) -> dict:
        """Get current governance status."""
        return {
            "state": self._state_record.state.value,
            "last_transition": self._state_record.last_transition.isoformat(),
            "reason": self._state_record.reason,
            "cooldown_remaining": max(
                0,
                self.config.cooldown_seconds - (datetime.now() - self._state_record.last_transition).total_seconds()
            ),
            "daily_spend_at_transition": self._state_record.daily_spend_at_transition,
            "monthly_spend_at_transition": self._state_record.monthly_spend_at_transition,
        }

    def force_state(self, state: GovernanceState, reason: str = "Manual override"):
        """Force a specific governance state (for testing/admin)."""
        self._transition_to(state, reason, 0.0, 0.0)


# Cost per 1K tokens (approximate, update as needed)
TOKEN_COSTS = {
    # Gemini models
    "gemini-3-flash-preview": {"input": 0.0, "output": 0.0},  # Preview = free
    "gemini-3-pro-preview": {"input": 0.0, "output": 0.0},
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-flash-lite": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},
    # OpenAI models (for Graphiti/memory)
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}


class CostTracker:
    """
    Centralized API cost tracking with budgets, circuit breakers, and 4-state governance.

    Features:
    - Track usage per provider/model
    - Daily and monthly budgets
    - Alert thresholds
    - Hard stops when budget exceeded
    - 4-state governance (NORMAL/WARN/THROTTLE/SHUTDOWN)
    - Voice/webhook notifications for SHUTDOWN
    - Persistent storage
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        budgets: Optional[dict[Provider, BudgetConfig]] = None,
        governance_config: Optional[GovernanceConfig] = None,
    ):
        self.storage_path = storage_path or Path.home() / ".claude" / "cost-tracking"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.budgets = budgets or DEFAULT_BUDGETS.copy()
        self.usage_file = self.storage_path / "usage.json"
        self.alerts_file = self.storage_path / "alerts.json"

        # Initialize 4-state governance with state path relative to storage
        governance_state_path = self.storage_path / "governance-state.json"
        self.governor = CostGovernor(config=governance_config, state_path=governance_state_path)

        self._usage: list[UsageRecord] = []
        self._alerts: list[dict] = []
        self._load_state()

    def _load_state(self):
        """Load persisted usage data."""
        if self.usage_file.exists():
            try:
                data = json.loads(self.usage_file.read_text())
                for record in data:
                    record["provider"] = Provider(record["provider"])
                    record["timestamp"] = datetime.fromisoformat(record["timestamp"])
                    self._usage.append(UsageRecord(**record))
            except (json.JSONDecodeError, KeyError):
                self._usage = []

    def _save_state(self):
        """Persist usage data (non-blocking via thread pool)."""
        data = [r.to_dict() for r in self._usage]
        content = json.dumps(data, indent=2)
        # PERF: Use thread pool to avoid blocking event loop
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self.usage_file.write_text, content)
        except RuntimeError:
            # No event loop running, write synchronously
            self.usage_file.write_text(content)

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for token usage."""
        costs = TOKEN_COSTS.get(model, {"input": 0.001, "output": 0.002})
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return round(input_cost + output_cost, 6)

    async def record_usage(
        self,
        provider: Provider,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        metadata: Optional[dict] = None,
    ) -> UsageRecord:
        """Record API usage and check budgets."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            provider=provider,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=model,
            metadata=metadata or {},
        )

        self._usage.append(record)
        self._save_state()

        # Check budgets
        await self._check_budgets(provider)

        return record

    async def _check_budgets(self, provider: Provider):
        """Check if budget thresholds are exceeded and update governance state."""
        budget = self.budgets.get(provider)
        if not budget:
            return

        daily = self.get_daily_spend(provider)
        monthly = self.get_monthly_spend(provider)

        # Check daily limit
        if daily >= budget.daily_limit_usd * budget.alert_threshold_pct:
            alert = {
                "type": "daily_threshold",
                "provider": provider.value,
                "current": daily,
                "limit": budget.daily_limit_usd,
                "timestamp": datetime.now().isoformat(),
            }
            self._alerts.append(alert)
            logger.warning(f"COST ALERT: {provider.value} at ${daily:.2f}/{budget.daily_limit_usd:.2f} daily")

        # Check monthly limit
        if monthly >= budget.monthly_limit_usd * budget.alert_threshold_pct:
            alert = {
                "type": "monthly_threshold",
                "provider": provider.value,
                "current": monthly,
                "limit": budget.monthly_limit_usd,
                "timestamp": datetime.now().isoformat(),
            }
            self._alerts.append(alert)
            logger.warning(f"COST ALERT: {provider.value} at ${monthly:.2f}/{budget.monthly_limit_usd:.2f} monthly")

        # Update governance state based on total spending
        total_daily = self.get_total_daily_spend()
        total_monthly = self.get_total_monthly_spend()
        total_daily_limit = sum(b.daily_limit_usd for b in self.budgets.values())
        total_monthly_limit = sum(b.monthly_limit_usd for b in self.budgets.values())

        self.governor.evaluate_state(
            daily_spend=total_daily,
            daily_limit=total_daily_limit,
            monthly_spend=total_monthly,
            monthly_limit=total_monthly_limit,
        )

    def check_can_proceed(
        self,
        provider: Provider,
        is_essential: bool = False,
    ) -> tuple[bool, str]:
        """
        Check if API call should proceed based on budget and governance state.

        Args:
            provider: The API provider
            is_essential: If True, call is allowed in THROTTLE state

        Returns:
            Tuple of (can_proceed, reason)
        """
        # First check governance state (4-state model)
        can_proceed, reason = self.governor.can_proceed(is_essential=is_essential)
        if not can_proceed:
            return False, reason

        # Then check provider-specific budget
        budget = self.budgets.get(provider)
        if not budget or not budget.hard_stop_enabled:
            return True, ""

        daily = self.get_daily_spend(provider)
        monthly = self.get_monthly_spend(provider)

        if daily >= budget.daily_limit_usd:
            return False, f"Daily budget exceeded for {provider.value}: ${daily:.2f}/${budget.daily_limit_usd:.2f}"

        if monthly >= budget.monthly_limit_usd:
            return False, f"Monthly budget exceeded for {provider.value}: ${monthly:.2f}/${budget.monthly_limit_usd:.2f}"

        return True, ""

    def get_total_daily_spend(self) -> float:
        """Get total spend across all providers for today."""
        today = datetime.now().date()
        return sum(
            r.cost_usd for r in self._usage
            if r.timestamp.date() == today
        )

    def get_total_monthly_spend(self) -> float:
        """Get total spend across all providers for current month."""
        now = datetime.now()
        return sum(
            r.cost_usd for r in self._usage
            if r.timestamp.year == now.year and r.timestamp.month == now.month
        )

    def get_governance_status(self) -> dict:
        """Get current governance status."""
        return self.governor.get_status()

    def get_daily_spend(self, provider: Provider) -> float:
        """Get total spend for today."""
        today = datetime.now().date()
        return sum(
            r.cost_usd for r in self._usage
            if r.provider == provider and r.timestamp.date() == today
        )

    def get_monthly_spend(self, provider: Provider) -> float:
        """Get total spend for current month."""
        now = datetime.now()
        return sum(
            r.cost_usd for r in self._usage
            if r.provider == provider
            and r.timestamp.year == now.year
            and r.timestamp.month == now.month
        )

    def get_total_spend(self, provider: Optional[Provider] = None) -> float:
        """Get total spend (optionally filtered by provider)."""
        if provider:
            return sum(r.cost_usd for r in self._usage if r.provider == provider)
        return sum(r.cost_usd for r in self._usage)

    def get_summary(self) -> dict:
        """Get comprehensive cost summary."""
        now = datetime.now()

        summary = {
            "generated_at": now.isoformat(),
            "providers": {},
            "totals": {
                "today": 0.0,
                "this_month": 0.0,
                "all_time": 0.0,
            },
        }

        for provider in Provider:
            budget = self.budgets.get(provider, BudgetConfig())
            daily = self.get_daily_spend(provider)
            monthly = self.get_monthly_spend(provider)
            total = self.get_total_spend(provider)

            summary["providers"][provider.value] = {
                "today": {
                    "spent": round(daily, 4),
                    "limit": budget.daily_limit_usd,
                    "remaining": round(budget.daily_limit_usd - daily, 4),
                    "pct_used": round((daily / budget.daily_limit_usd * 100) if budget.daily_limit_usd > 0 else 0, 1),
                },
                "month": {
                    "spent": round(monthly, 4),
                    "limit": budget.monthly_limit_usd,
                    "remaining": round(budget.monthly_limit_usd - monthly, 4),
                    "pct_used": round((monthly / budget.monthly_limit_usd * 100) if budget.monthly_limit_usd > 0 else 0, 1),
                },
                "all_time": round(total, 4),
            }

            summary["totals"]["today"] += daily
            summary["totals"]["this_month"] += monthly
            summary["totals"]["all_time"] += total

        # Round totals
        for key in summary["totals"]:
            summary["totals"][key] = round(summary["totals"][key], 4)

        # Add governance status
        summary["governance"] = self.get_governance_status()

        return summary

    def get_usage_by_model(self, days: int = 30) -> dict:
        """Get usage breakdown by model."""
        cutoff = datetime.now() - timedelta(days=days)

        by_model = {}
        for record in self._usage:
            if record.timestamp >= cutoff:
                model = record.model or "unknown"
                if model not in by_model:
                    by_model[model] = {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost": 0.0,
                    }
                by_model[model]["calls"] += 1
                by_model[model]["input_tokens"] += record.input_tokens
                by_model[model]["output_tokens"] += record.output_tokens
                by_model[model]["cost"] += record.cost_usd

        return by_model

    def set_budget(
        self,
        provider: Provider,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
    ):
        """Update budget for a provider."""
        if provider not in self.budgets:
            self.budgets[provider] = BudgetConfig()

        if daily_limit is not None:
            self.budgets[provider].daily_limit_usd = daily_limit
        if monthly_limit is not None:
            self.budgets[provider].monthly_limit_usd = monthly_limit

    def clear_old_records(self, days: int = 90):
        """Clear records older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        self._usage = [r for r in self._usage if r.timestamp >= cutoff]
        self._save_state()


# Global instances
_tracker: Optional[CostTracker] = None
_governor: Optional[CostGovernor] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def get_cost_governor() -> CostGovernor:
    """Get the global cost governor (via cost tracker)."""
    return get_cost_tracker().governor


__all__ = [
    # Enums
    "Provider",
    "GovernanceState",
    # Dataclasses
    "UsageRecord",
    "BudgetConfig",
    "GovernanceConfig",
    "GovernanceStateRecord",
    # Classes
    "CostTracker",
    "CostGovernor",
    # Constants
    "DEFAULT_BUDGETS",
    "TOKEN_COSTS",
    # Functions
    "get_cost_tracker",
    "get_cost_governor",
]
