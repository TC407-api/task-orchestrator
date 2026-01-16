"""Enterprise cost dashboard with real-time tracking and alerts."""
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from src.core.cost_tracker import CostTracker, Provider


@dataclass
class TimeRange:
    """Time range for queries."""
    start: datetime
    end: datetime


@dataclass
class TrendPoint:
    """Single point in a cost trend."""
    timestamp: datetime
    cost: float
    provider: Optional[str] = None


@dataclass
class CostAlert:
    """Alert for budget threshold violations."""
    severity: str  # "WARNING" or "CRITICAL"
    message: str
    threshold: float
    current_value: float


@dataclass
class DashboardSummary:
    """Summary of costs across all providers and models."""
    total_cost: float
    by_provider: Dict[Provider, float]
    by_model: Dict[str, float]
    top_operations: List[str]


@dataclass
class ProjectCosts:
    """Cost breakdown for a specific project."""
    project_id: str
    total_cost: float
    operations: List[str]
    by_model: Dict[str, float]


class CostDashboard:
    """
    Central governance dashboard for visualizing costs, trends, and managing
    budget alerts across the enterprise task orchestration system.
    """

    def __init__(self, cost_tracker: CostTracker):
        """
        Initialize the dashboard.

        Args:
            cost_tracker: The CostTracker instance to query
        """
        self.tracker = cost_tracker

    def get_summary(self, time_range: TimeRange) -> DashboardSummary:
        """
        Aggregates cost data within a specific time range.

        Args:
            time_range: The start and end dates for the report.

        Returns:
            DashboardSummary object containing totals and breakdowns.
        """
        records = self.tracker.get_history()
        total_cost = 0.0
        by_provider: Dict[Provider, float] = defaultdict(float)
        by_model: Dict[str, float] = defaultdict(float)
        op_costs: Dict[str, float] = defaultdict(float)

        for r in records:
            cost = r.cost_usd
            total_cost += cost
            by_provider[r.provider] += cost
            by_model[r.model] += cost
            op_costs[r.operation] += cost

        # Get top operations
        top_ops = sorted(op_costs.keys(), key=lambda x: op_costs[x], reverse=True)[:5]

        return DashboardSummary(
            total_cost=round(total_cost, 6),
            by_provider=dict(by_provider),
            by_model=dict(by_model),
            top_operations=top_ops,
        )

    def get_by_project(self, project_id: str) -> ProjectCosts:
        """
        Filters cost records by project ID (derived from operation prefixes).

        Args:
            project_id: The identifier string for the project.

        Returns:
            ProjectCosts object with project-specific metrics.
        """
        records = self.tracker.get_history()
        total_cost = 0.0
        operations: List[str] = []
        by_model: Dict[str, float] = defaultdict(float)
        prefix = f"{project_id}:"

        for r in records:
            if r.operation.startswith(prefix):
                cost = r.cost_usd
                total_cost += cost
                # Extract operation name after prefix
                op_name = r.operation[len(prefix):]
                if op_name not in operations:
                    operations.append(op_name)
                by_model[r.model] += cost

        return ProjectCosts(
            project_id=project_id,
            total_cost=round(total_cost, 6),
            operations=operations,
            by_model=dict(by_model),
        )

    def get_trends(self, granularity: str = "daily") -> List[TrendPoint]:
        """
        Calculates cost trends over time.

        Args:
            granularity: 'daily', 'weekly', or 'monthly'.

        Returns:
            List of TrendPoint objects for plotting.
        """
        records = self.tracker.get_history()
        grouped: Dict[str, float] = defaultdict(float)

        for r in records:
            ts = r.timestamp
            if granularity == "weekly":
                key = f"{ts.year}-W{ts.isocalendar()[1]}"
            elif granularity == "monthly":
                key = ts.strftime("%Y-%m")
            else:  # daily
                key = ts.strftime("%Y-%m-%d")
            grouped[key] += r.cost_usd

        # Convert to TrendPoints
        trends = []
        for key in sorted(grouped.keys()):
            # Parse key back to datetime (approximate)
            if granularity == "monthly":
                ts = datetime.strptime(key, "%Y-%m")
            elif granularity == "weekly":
                year, week = key.split("-W")
                ts = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
            else:
                ts = datetime.strptime(key, "%Y-%m-%d")

            trends.append(TrendPoint(
                timestamp=ts,
                cost=round(grouped[key], 6),
            ))

        return trends

    def generate_alerts(self) -> List[CostAlert]:
        """
        Checks current spend against defined budgets in the CostTracker.

        Returns:
            List of active CostAlerts.
        """
        alerts = []

        daily_spend = self.tracker.get_daily_spend(Provider.OPENAI)  # Check main provider
        daily_limit = self.tracker.daily_limit_usd if hasattr(self.tracker, 'daily_limit_usd') else 10.0

        if daily_spend > daily_limit:
            alerts.append(CostAlert(
                severity="CRITICAL",
                message=f"Daily limit exceeded: ${daily_spend:.2f} > ${daily_limit:.2f}",
                threshold=daily_limit,
                current_value=daily_spend,
            ))

        return alerts

    def export_csv(self, output_path: str) -> None:
        """
        Exports the entire available cost history to a CSV file.

        Args:
            output_path: Filesystem path to write the CSV.
        """
        headers = ["Timestamp", "Provider", "Cost (USD)", "Operation", "Model"]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for r in self.tracker.get_history():
                writer.writerow([
                    r.timestamp.isoformat(),
                    r.provider.value if hasattr(r.provider, 'value') else str(r.provider),
                    r.cost_usd,
                    r.operation,
                    r.model,
                ])
