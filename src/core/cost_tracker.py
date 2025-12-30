"""API cost tracking with budgets, alerts, and circuit breakers."""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional
import asyncio


class Provider(Enum):
    """Supported API providers."""
    GOOGLE_GEMINI = "google_gemini"
    GOOGLE_GMAIL = "google_gmail"
    GOOGLE_CALENDAR = "google_calendar"
    OPENAI = "openai"
    GRAPHITI = "graphiti"
    ANTHROPIC = "anthropic"


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


# Default conservative budgets
DEFAULT_BUDGETS = {
    Provider.GOOGLE_GEMINI: BudgetConfig(daily_limit_usd=2.0, monthly_limit_usd=30.0),
    Provider.GOOGLE_GMAIL: BudgetConfig(daily_limit_usd=0.50, monthly_limit_usd=10.0),
    Provider.GOOGLE_CALENDAR: BudgetConfig(daily_limit_usd=0.50, monthly_limit_usd=10.0),
    Provider.OPENAI: BudgetConfig(daily_limit_usd=1.0, monthly_limit_usd=20.0),
    Provider.GRAPHITI: BudgetConfig(daily_limit_usd=1.0, monthly_limit_usd=20.0),
    Provider.ANTHROPIC: BudgetConfig(daily_limit_usd=0.0, monthly_limit_usd=0.0),  # Max plan = free
}

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
    Centralized API cost tracking with budgets and circuit breakers.

    Features:
    - Track usage per provider/model
    - Daily and monthly budgets
    - Alert thresholds
    - Hard stops when budget exceeded
    - Persistent storage
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        budgets: Optional[dict[Provider, BudgetConfig]] = None,
    ):
        self.storage_path = storage_path or Path.home() / ".claude" / "cost-tracking"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.budgets = budgets or DEFAULT_BUDGETS.copy()
        self.usage_file = self.storage_path / "usage.json"
        self.alerts_file = self.storage_path / "alerts.json"

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
        """Persist usage data."""
        data = [r.to_dict() for r in self._usage]
        self.usage_file.write_text(json.dumps(data, indent=2))

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
        """Check if budget thresholds are exceeded."""
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
            print(f"⚠️  COST ALERT: {provider.value} at ${daily:.2f}/{budget.daily_limit_usd:.2f} daily")

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
            print(f"⚠️  COST ALERT: {provider.value} at ${monthly:.2f}/{budget.monthly_limit_usd:.2f} monthly")

    def check_can_proceed(self, provider: Provider) -> tuple[bool, str]:
        """Check if API call should proceed based on budget."""
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
        today = now.date()

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


# Global instance
_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker
