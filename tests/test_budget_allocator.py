import pytest
from datetime import datetime, timedelta
from src.governance.budget_allocator import (
    BudgetAllocator,
    Department,
    BudgetCheckResult,
    ChargebackReport
)
from src.core.cost_tracker import BudgetConfig


# Mocks/Fixtures
@pytest.fixture
def allocator(tmp_path):
    """Returns a BudgetAllocator instance using a temp directory."""
    return BudgetAllocator(storage_path=str(tmp_path / "budget.db"))


@pytest.fixture
def default_config():
    """Returns a standard BudgetConfig for testing."""
    return BudgetConfig(
        daily_limit_usd=100.0,
        monthly_limit_usd=3000.0,
        alert_threshold_pct=0.8,
        hard_stop_enabled=True
    )


def test_allocator_creates_department_budget(allocator, default_config):
    """Test that a department can be created with a specific budget config."""
    dept = allocator.create_department(
        name="Engineering",
        parent_id=None,
        budget_config=default_config
    )

    assert isinstance(dept, Department)
    assert dept.name == "Engineering"
    assert dept.budget_config == default_config
    assert dept.id is not None


def test_allocator_enforces_limits(allocator, default_config):
    """Test that the allocator blocks costs exceeding the hard stop limit."""
    dept = allocator.create_department("Sales", None, default_config)

    # First check: Within budget
    result_ok = allocator.check_budget(dept.id, 50.0)
    assert result_ok.allowed is True

    # Second check: Exceeds daily limit (100.0)
    # 50 existing + 60 new = 110 > 100
    result_fail = allocator.check_budget(dept.id, 60.0)
    assert result_fail.allowed is False
    assert "exceeded" in result_fail.reason.lower()


def test_allocator_supports_hierarchical_budgets(allocator, default_config):
    """Test creation of parent-child department relationships."""
    parent = allocator.create_department("HQ", None, default_config)
    child = allocator.create_department("R&D", parent.id, default_config)

    assert child.parent_id == parent.id

    # Ensure checking child budget is valid
    result = allocator.check_budget(child.id, 10.0)
    assert isinstance(result, BudgetCheckResult)


def test_allocator_generates_chargeback_report(allocator, default_config):
    """Test generation of a chargeback report for a specific time range."""
    dept = allocator.create_department("Ops", None, default_config)
    allocator.allocate_budget(dept.id, 5000.0)

    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    report = allocator.generate_chargeback((start_date, end_date))

    assert isinstance(report, ChargebackReport)
    assert dept.id in report.allocations
    assert dept.id in report.costs
    assert report.time_range == (start_date, end_date)


def test_allocator_alerts_on_threshold(allocator, default_config):
    """Test that approaching the budget threshold triggers an alert reason."""
    # Threshold is 80% (0.8) of 100.0 daily = 80.0
    dept = allocator.create_department("Marketing", None, default_config)

    # Spend 85.0 (over 80.0 threshold, but under 100.0 limit)
    result = allocator.check_budget(dept.id, 85.0)

    assert result.allowed is True
    assert "alert" in result.reason.lower() or result.remaining < 15.0
