import pytest
import csv
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import List

from src.core.cost_tracker import CostTracker, UsageRecord, Provider
from src.governance.cost_dashboard import (
    CostDashboard,
    TimeRange,
    DashboardSummary,
    ProjectCosts,
    TrendPoint,
    CostAlert
)


@pytest.fixture
def mock_cost_tracker():
    """Creates a mock CostTracker with predefined usage history."""
    tracker = MagicMock(spec=CostTracker)

    # Create sample records
    now = datetime.now()
    records = [
        UsageRecord(
            provider=Provider.OPENAI,
            operation="proj_alpha:chat",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.02,
            timestamp=now,
            model="gpt-4"
        ),
        UsageRecord(
            provider=Provider.ANTHROPIC,
            operation="proj_beta:summarize",
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.05,
            timestamp=now - timedelta(hours=2),
            model="claude-3-opus"
        ),
        UsageRecord(
            provider=Provider.OPENAI,
            operation="proj_alpha:embedding",
            input_tokens=50,
            output_tokens=0,
            cost_usd=0.001,
            timestamp=now - timedelta(days=1),
            model="text-embedding-3"
        )
    ]

    # Assume the dashboard accesses a method to get raw records or public property
    # We mock a hypothetical accessor for the purpose of the dashboard logic
    tracker.get_history = MagicMock(return_value=records)

    # Mock budget configs
    tracker.daily_limit_usd = 10.0
    tracker.monthly_limit_usd = 100.0
    tracker.get_daily_spend.return_value = 0.071

    return tracker


@pytest.fixture
def dashboard(mock_cost_tracker):
    return CostDashboard(cost_tracker=mock_cost_tracker)


def test_dashboard_aggregates_by_provider(dashboard):
    """Test that the dashboard correctly sums costs per provider."""
    time_range = TimeRange(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now()
    )

    summary = dashboard.get_summary(time_range)

    assert isinstance(summary, DashboardSummary)
    assert summary.total_cost == 0.071
    assert summary.by_provider[Provider.OPENAI] == 0.021
    assert summary.by_provider[Provider.ANTHROPIC] == 0.05


def test_dashboard_aggregates_by_project(dashboard):
    """
    Test that costs can be filtered/aggregated by project ID.
    Assumes 'operation' field contains project identifiers (e.g., 'proj_alpha:task').
    """
    project_costs = dashboard.get_by_project("proj_alpha")

    assert isinstance(project_costs, ProjectCosts)
    assert project_costs.project_id == "proj_alpha"
    assert project_costs.total_cost == 0.021
    assert "chat" in project_costs.operations
    assert "embedding" in project_costs.operations


def test_dashboard_shows_trends(dashboard):
    """Test that the dashboard calculates cost trends over time."""
    trends = dashboard.get_trends(granularity="daily")

    assert isinstance(trends, list)
    assert len(trends) > 0
    assert isinstance(trends[0], TrendPoint)
    assert trends[0].cost > 0
    assert isinstance(trends[0].timestamp, datetime)


def test_dashboard_generates_alerts(dashboard, mock_cost_tracker):
    """Test that alerts are generated when spend exceeds thresholds."""
    # Simulate high spend
    mock_cost_tracker.get_daily_spend.return_value = 15.0  # Limit is 10.0

    alerts = dashboard.generate_alerts()

    assert isinstance(alerts, list)
    assert len(alerts) > 0
    assert isinstance(alerts[0], CostAlert)
    assert alerts[0].severity == "CRITICAL"
    assert alerts[0].current_value == 15.0
    assert alerts[0].threshold == 10.0


def test_dashboard_exports_csv(dashboard, tmp_path):
    """Test that the dashboard can export data to a CSV file."""
    output_file = tmp_path / "cost_report.csv"
    output_path_str = str(output_file)

    dashboard.export_csv(output_path_str)

    assert output_file.exists()

    # Verify content format
    with open(output_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "Timestamp" in header
        assert "Provider" in header
        assert "Cost (USD)" in header
