"""Department/project budget allocation and chargeback reporting."""
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime

# Import BudgetConfig from cost_tracker
from src.core.cost_tracker import BudgetConfig


@dataclass
class BudgetCheckResult:
    """Result of a budget check operation."""
    allowed: bool
    remaining: float
    reason: str


@dataclass
class Department:
    """Represents a department with a specific budget configuration."""
    id: str
    name: str
    parent_id: Optional[str]
    budget_config: BudgetConfig


@dataclass
class ChargebackReport:
    """Report detailing costs and allocations over a time range."""
    departments: List[Department]
    costs: Dict[str, float]  # Maps Department ID to total cost
    allocations: Dict[str, float]  # Maps Department ID to allocated budget
    time_range: Tuple[datetime, datetime]


class BudgetAllocator:
    """Manages department budgets, hierarchies, and chargeback reporting."""

    def __init__(self, storage_path: str) -> None:
        """
        Initialize the budget allocator.

        Args:
            storage_path: Path to store budget data
        """
        self.storage_path = storage_path
        self._departments: Dict[str, Department] = {}
        self._allocations: Dict[str, float] = {}
        self._spending: Dict[str, float] = {}

    def create_department(
        self,
        name: str,
        parent_id: Optional[str],
        budget_config: BudgetConfig
    ) -> Department:
        """
        Creates a new department with the specified budget configuration.

        Args:
            name: Department name
            parent_id: Optional parent department ID for hierarchy
            budget_config: Budget limits and thresholds

        Returns:
            The created Department object
        """
        dept_id = str(uuid.uuid4())
        department = Department(
            id=dept_id,
            name=name,
            parent_id=parent_id,
            budget_config=budget_config,
        )
        self._departments[dept_id] = department
        self._spending[dept_id] = 0.0
        self._allocations[dept_id] = 0.0
        return department

    def allocate_budget(self, dept_id: str, amount: float) -> None:
        """
        Allocates a specific monetary amount to a department.

        Args:
            dept_id: Department ID to allocate to
            amount: Amount in USD to allocate
        """
        if dept_id in self._departments:
            self._allocations[dept_id] = self._allocations.get(dept_id, 0.0) + amount

    def check_budget(self, dept_id: str, cost: float) -> BudgetCheckResult:
        """
        Checks if a cost can be incurred based on the department's budget.

        Args:
            dept_id: Department ID to check
            cost: Proposed cost amount

        Returns:
            BudgetCheckResult with allowed status and remaining budget
        """
        if dept_id not in self._departments:
            return BudgetCheckResult(
                allowed=False,
                remaining=0.0,
                reason="Department not found"
            )

        dept = self._departments[dept_id]
        config = dept.budget_config
        current_spend = self._spending.get(dept_id, 0.0)
        projected = current_spend + cost
        limit = config.daily_limit_usd
        remaining = limit - projected

        # Check if exceeds limit
        if projected > limit:
            return BudgetCheckResult(
                allowed=False,
                remaining=remaining,
                reason=f"Budget exceeded: ${projected:.2f} > ${limit:.2f}"
            )

        # Track the spending (simulate the cost being incurred)
        self._spending[dept_id] = projected

        # Check if at alert threshold (80%+)
        threshold_pct = config.alert_threshold_pct
        if projected >= (limit * threshold_pct):
            return BudgetCheckResult(
                allowed=True,
                remaining=remaining,
                reason=f"Alert: {threshold_pct*100:.0f}% threshold reached"
            )

        return BudgetCheckResult(
            allowed=True,
            remaining=remaining,
            reason="OK"
        )

    def generate_chargeback(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> ChargebackReport:
        """
        Generates a chargeback report for the specified time range.

        Args:
            time_range: Tuple of (start_datetime, end_datetime)

        Returns:
            ChargebackReport with department costs and allocations
        """
        return ChargebackReport(
            departments=list(self._departments.values()),
            costs=self._spending.copy(),
            allocations=self._allocations.copy(),
            time_range=time_range,
        )
