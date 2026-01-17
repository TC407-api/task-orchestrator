"""Coordinator agent that orchestrates other agents."""
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from .email_agent import EmailAgent, ExtractedTask, TaskPriority
from .calendar_agent import CalendarAgent, ScheduledTask
from .llm_mixin import LLMMixin

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm import ModelRouter as ModelRouterType

try:
    from ..llm import ModelRouter, TaskType
except ImportError:
    ModelRouter = None  # type: ignore[misc, assignment]
    TaskType = None  # type: ignore[misc, assignment]


class TaskStatus(Enum):
    """Task lifecycle status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskSource(Enum):
    """Where the task originated."""
    EMAIL = "email"
    CALENDAR = "calendar"
    MANUAL = "manual"
    CODE_COMMIT = "code_commit"
    CLAUDE_CODE = "claude_code"


@dataclass
class Task:
    """A tracked task in the system."""
    id: str
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    source: TaskSource
    created_at: datetime
    due_date: Optional[datetime] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    calendar_event_id: Optional[str] = None
    source_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    estimated_minutes: int = 30

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "scheduled_start": (
                self.scheduled_start.isoformat() if self.scheduled_start else None
            ),
            "scheduled_end": (
                self.scheduled_end.isoformat() if self.scheduled_end else None
            ),
            "calendar_event_id": self.calendar_event_id,
            "source_id": self.source_id,
            "tags": self.tags,
            "notes": self.notes,
            "estimated_minutes": self.estimated_minutes,
        }


class CoordinatorAgent(LLMMixin):
    """
    Central coordinator that orchestrates email and calendar agents.

    Responsibilities:
    - Receive tasks from all sources
    - Prioritize and deduplicate tasks
    - Delegate to specialized agents
    - Track completion status
    - Provide unified task view
    """

    def __init__(
        self,
        email_agent: Optional[EmailAgent] = None,
        calendar_agent: Optional[CalendarAgent] = None,
        llm_router: Optional["ModelRouterType"] = None,
    ):
        self.email_agent = email_agent
        self.calendar_agent = calendar_agent
        self.llm = llm_router
        self.tasks: dict[str, Task] = {}
        self._task_lock = asyncio.Lock()

    async def add_task(
        self,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        source: TaskSource = TaskSource.MANUAL,
        due_date: Optional[datetime] = None,
        source_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        estimated_minutes: int = 30,
        auto_schedule: bool = False,
    ) -> Task:
        """
        Add a new task to the system.

        Args:
            title: Task title
            description: Detailed description
            priority: Priority level
            source: Where task came from
            due_date: When task is due
            source_id: ID from source system (email ID, etc.)
            tags: Task tags
            estimated_minutes: Estimated time to complete
            auto_schedule: Automatically schedule on calendar

        Returns:
            Created Task
        """
        task_id = str(uuid4())[:8]

        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            source=source,
            created_at=datetime.now(),
            due_date=due_date,
            source_id=source_id,
            tags=tags or [],
            estimated_minutes=estimated_minutes,
        )

        async with self._task_lock:
            self.tasks[task_id] = task

        # Auto-schedule if requested
        if auto_schedule and self.calendar_agent:
            await self.schedule_task(task_id)

        return task

    async def add_task_from_email(self, extracted: ExtractedTask) -> Task:
        """Add a task extracted from email."""
        return await self.add_task(
            title=extracted.description,
            description=extracted.context,
            priority=extracted.priority,
            source=TaskSource.EMAIL,
            due_date=extracted.due_date,
            source_id=extracted.source_email_id,
            tags=extracted.tags,
        )

    async def sync_from_email(self) -> list[Task]:
        """
        Sync tasks from unread emails.

        Returns:
            List of newly created tasks
        """
        if not self.email_agent:
            raise ValueError("Email agent not configured")

        analyses = await self.email_agent.process_unread()
        new_tasks = []

        for analysis in analyses:
            for extracted in analysis.extracted_tasks:
                # Check for duplicates
                if not self._is_duplicate(extracted):
                    task = await self.add_task_from_email(extracted)
                    new_tasks.append(task)

        return new_tasks

    def _is_duplicate(self, extracted: ExtractedTask) -> bool:
        """Check if task already exists."""
        for task in self.tasks.values():
            if (
                task.source == TaskSource.EMAIL
                and task.source_id == extracted.source_email_id
            ):
                return True
        return False

    async def schedule_task(
        self,
        task_id: str,
        preferred_time: Optional[datetime] = None,
    ) -> Optional[ScheduledTask]:
        """
        Schedule a task on the calendar.

        Args:
            task_id: Task to schedule
            preferred_time: Preferred start time

        Returns:
            ScheduledTask if successful
        """
        if not self.calendar_agent:
            raise ValueError("Calendar agent not configured")

        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        scheduled = self.calendar_agent.schedule_task(
            task_id=task_id,
            title=task.title,
            duration_minutes=task.estimated_minutes,
            preferred_time=preferred_time,
            deadline=task.due_date,
        )

        if scheduled:
            task.status = TaskStatus.SCHEDULED
            task.scheduled_start = scheduled.start
            task.scheduled_end = scheduled.end
            task.calendar_event_id = scheduled.event_id

        return scheduled

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        notes: str = "",
    ) -> Task:
        """Update task status."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        task.status = status
        if notes:
            task.notes = notes

        return task

    async def complete_task(self, task_id: str, notes: str = "") -> Task:
        """Mark a task as completed."""
        return await self.update_task_status(
            task_id, TaskStatus.COMPLETED, notes
        )

    async def prioritize_tasks(self) -> list[Task]:
        """
        Get tasks sorted by priority and due date.

        Returns:
            Sorted list of pending/scheduled tasks
        """
        active_tasks = [
            t for t in self.tasks.values()
            if t.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED, TaskStatus.IN_PROGRESS]
        ]

        # Sort by priority (critical first) then by due date
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
        }

        return sorted(
            active_tasks,
            key=lambda t: (
                priority_order[t.priority],
                t.due_date or datetime.max,
            ),
        )

    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Get all tasks with given status."""
        return [t for t in self.tasks.values() if t.status == status]

    def get_tasks_by_source(self, source: TaskSource) -> list[Task]:
        """Get all tasks from given source."""
        return [t for t in self.tasks.values() if t.source == source]

    def get_overdue_tasks(self) -> list[Task]:
        """Get tasks past their due date."""
        now = datetime.now()
        return [
            t for t in self.tasks.values()
            if t.due_date and t.due_date < now
            and t.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
        ]

    async def get_daily_summary(self) -> dict:
        """
        Get summary of today's tasks and schedule.

        Returns:
            Dict with task counts, schedule info, recommendations
        """
        prioritized = await self.prioritize_tasks()
        overdue = self.get_overdue_tasks()

        calendar_summary = None
        if self.calendar_agent:
            calendar_summary = self.calendar_agent.get_day_summary()

        # Group by priority
        by_priority = {}
        for p in TaskPriority:
            by_priority[p.value] = [
                t.to_dict() for t in prioritized if t.priority == p
            ]

        summary = {
            "date": datetime.now().isoformat(),
            "total_active_tasks": len(prioritized),
            "overdue_count": len(overdue),
            "tasks_by_priority": by_priority,
            "overdue_tasks": [t.to_dict() for t in overdue],
            "top_3_tasks": [t.to_dict() for t in prioritized[:3]],
        }

        if calendar_summary:
            summary["calendar"] = {
                "meetings_today": calendar_summary.total_meetings,
                "meeting_hours": calendar_summary.meeting_hours,
                "focus_hours_available": calendar_summary.focus_hours,
                "busy_percentage": calendar_summary.busy_percentage,
            }

        return summary

    async def auto_schedule_pending(
        self,
        max_tasks: int = 5,
    ) -> list[ScheduledTask]:
        """
        Automatically schedule pending tasks.

        Args:
            max_tasks: Maximum tasks to schedule

        Returns:
            List of scheduled tasks
        """
        if not self.calendar_agent:
            return []

        pending = self.get_tasks_by_status(TaskStatus.PENDING)
        prioritized = sorted(
            pending,
            key=lambda t: (
                {TaskPriority.CRITICAL: 0, TaskPriority.HIGH: 1,
                 TaskPriority.MEDIUM: 2, TaskPriority.LOW: 3}[t.priority],
                t.due_date or datetime.max,
            ),
        )

        scheduled = []
        for task in prioritized[:max_tasks]:
            result = await self.schedule_task(task.id)
            if result:
                scheduled.append(result)

        return scheduled

    def export_tasks(self) -> str:
        """Export all tasks as JSON."""
        return json.dumps(
            [t.to_dict() for t in self.tasks.values()],
            indent=2,
        )

    def import_tasks(self, json_str: str) -> int:
        """
        Import tasks from JSON.

        Args:
            json_str: JSON string of task list

        Returns:
            Number of tasks imported
        """
        data = json.loads(json_str)
        count = 0

        for item in data:
            task = Task(
                id=item["id"],
                title=item["title"],
                description=item.get("description", ""),
                priority=TaskPriority(item.get("priority", "medium")),
                status=TaskStatus(item.get("status", "pending")),
                source=TaskSource(item.get("source", "manual")),
                created_at=datetime.fromisoformat(item["created_at"]),
                due_date=(
                    datetime.fromisoformat(item["due_date"])
                    if item.get("due_date")
                    else None
                ),
                tags=item.get("tags", []),
                notes=item.get("notes", ""),
                estimated_minutes=item.get("estimated_minutes", 30),
            )
            self.tasks[task.id] = task
            count += 1

        return count

