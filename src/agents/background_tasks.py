"""
Scheduled Background Tasks System for Task Orchestrator.

Allows agents to schedule tasks that run without blocking in background processes.
Provides task scheduling, status tracking, and result publishing to UniversalInbox.

Usage:
    scheduler = BackgroundTaskScheduler(inbox=inbox)

    task = ScheduledTask(
        name="sync_emails",
        func=email_agent.sync_emails,
        schedule_type=TaskScheduleType.RECURRING,
        interval_seconds=300,
    )

    task_id = await scheduler.schedule_task(task)
    status = await scheduler.get_task_status(task_id)
    await scheduler.cancel_task(task_id)
"""
import asyncio
import json
import logging
import sqlite3
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional
from uuid import uuid4

from .inbox import UniversalInbox, AgentEvent, EventType

logger = logging.getLogger(__name__)


class TaskScheduleType(str, Enum):
    """Types of task scheduling."""
    ONE_TIME = "ONE_TIME"
    RECURRING = "RECURRING"
    DEFERRED = "DEFERRED"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    task_name: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: Optional[str] = None
    error: Optional[str] = None
    execution_count: int = 1
    next_scheduled: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["started_at"] = self.started_at.isoformat()
        data["completed_at"] = (
            self.completed_at.isoformat() if self.completed_at else None
        )
        data["next_scheduled"] = (
            self.next_scheduled.isoformat() if self.next_scheduled else None
        )
        return data


@dataclass
class ScheduledTask:
    """Definition of a task to be scheduled."""
    name: str
    func: Callable
    schedule_type: TaskScheduleType
    task_id: str = field(default_factory=lambda: str(uuid4())[:8])
    run_at: Optional[datetime] = None
    interval_seconds: Optional[int] = None
    max_retries: int = 3
    timeout_seconds: int = 300
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            "task_id": self.task_id,
            "name": self.name,
            "schedule_type": self.schedule_type.value,
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "interval_seconds": self.interval_seconds,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "is_active": self.is_active,
        }
        return data


class BackgroundTaskScheduler:
    """
    Manages background task scheduling and execution.

    Features:
    - Schedule one-time, recurring, and deferred tasks
    - Async task execution without blocking
    - Subprocess isolation for reliability
    - Task status tracking
    - Results published to UniversalInbox
    - SQLite persistence
    """

    def __init__(
        self,
        inbox: UniversalInbox,
        db_path: Optional[str] = None,
        max_workers: int = 4,
    ):
        """
        Initialize the scheduler.

        Args:
            inbox: UniversalInbox instance for publishing results
            db_path: Path to SQLite database. If None, uses in-memory.
            max_workers: Maximum concurrent task workers
        """
        self.inbox = inbox
        self.db_path = db_path or ":memory:"
        self.max_workers = max_workers
        self._db_lock = Lock()
        self._scheduled_tasks: dict[str, ScheduledTask] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._worker_semaphore = asyncio.Semaphore(max_workers)
        # Keep persistent connection for in-memory databases
        self._conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()
        self._scheduler_task: Optional[asyncio.Task] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, using persistent connection for in-memory."""
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Scheduled tasks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                task_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                schedule_type TEXT NOT NULL,
                run_at TEXT,
                interval_seconds INTEGER,
                max_retries INTEGER,
                timeout_seconds INTEGER,
                created_at TEXT NOT NULL,
                last_run TEXT,
                next_run TEXT,
                is_active BOOLEAN
            )
            """
        )

        # Task results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_results (
                task_id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_seconds REAL,
                output TEXT,
                error TEXT,
                execution_count INTEGER,
                next_scheduled TEXT
            )
            """
        )

        # Task execution history
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                output TEXT,
                error TEXT,
                duration_seconds REAL
            )
            """
        )

        conn.commit()
        # Don't close persistent in-memory connection
        if self._conn is None:
            if self._conn is None: conn.close()

    async def schedule_task(
        self,
        task: ScheduledTask,
    ) -> str:
        """
        Schedule a task for execution.

        Args:
            task: ScheduledTask to schedule

        Returns:
            Task ID

        Raises:
            ValueError: If task configuration is invalid
        """
        if task.schedule_type == TaskScheduleType.ONE_TIME and not task.run_at:
            raise ValueError("ONE_TIME tasks must have run_at specified")

        if task.schedule_type == TaskScheduleType.RECURRING:
            if not task.interval_seconds or task.interval_seconds <= 0:
                raise ValueError("RECURRING tasks must have positive interval_seconds")
            if not task.next_run:
                task.next_run = datetime.now() + timedelta(
                    seconds=task.interval_seconds
                )

        if task.schedule_type == TaskScheduleType.DEFERRED:
            task.next_run = datetime.now()

        # Store task
        self._scheduled_tasks[task.task_id] = task
        self._store_task(task)

        # Publish scheduling event
        await self.inbox.publish(
            AgentEvent(
                event_type=EventType.TEXT_OUTPUT,
                agent_name="BackgroundTaskScheduler",
                data={
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "schedule_type": task.schedule_type.value,
                    "message": f"Task '{task.name}' scheduled",
                },
                source="background_scheduler",
            )
        )

        logger.info(f"Scheduled task {task.task_id}: {task.name}")
        return task.task_id

    def _store_task(self, task: ScheduledTask) -> None:
        """Store task in database."""
        with self._db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO scheduled_tasks
                    (task_id, name, schedule_type, run_at, interval_seconds,
                     max_retries, timeout_seconds, created_at, last_run, next_run,
                     is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.task_id,
                        task.name,
                        task.schedule_type.value,
                        task.run_at.isoformat() if task.run_at else None,
                        task.interval_seconds,
                        task.max_retries,
                        task.timeout_seconds,
                        task.created_at.isoformat(),
                        task.last_run.isoformat() if task.last_run else None,
                        task.next_run.isoformat() if task.next_run else None,
                        task.is_active,
                    ),
                )
                conn.commit()
            finally:
                if self._conn is None: conn.close()

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if cancelled, False if not found
        """
        task = self._scheduled_tasks.get(task_id)
        if not task:
            return False

        task.is_active = False
        self._store_task(task)

        # Cancel running task if active
        if task_id in self._running_tasks:
            running = self._running_tasks[task_id]
            if not running.done():
                running.cancel()
            del self._running_tasks[task_id]

        # Publish cancellation event
        await self.inbox.publish(
            AgentEvent(
                event_type=EventType.TEXT_OUTPUT,
                agent_name="BackgroundTaskScheduler",
                data={
                    "task_id": task_id,
                    "message": f"Task '{task.name}' cancelled",
                },
                source="background_scheduler",
            )
        )

        logger.info(f"Cancelled task {task_id}")
        return True

    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """
        Get current status of a task.

        Args:
            task_id: ID of task

        Returns:
            Dict with task info and status, or None if not found
        """
        task = self._scheduled_tasks.get(task_id)
        if not task:
            return None

        result = self._task_results.get(task_id)
        is_running = task_id in self._running_tasks

        return {
            "task": task.to_dict(),
            "result": result.to_dict() if result else None,
            "is_running": is_running,
            "status": (
                TaskStatus.RUNNING.value
                if is_running
                else (result.status.value if result else TaskStatus.PENDING.value)
            ),
        }

    async def list_scheduled_tasks(
        self,
        active_only: bool = True,
        schedule_type: Optional[TaskScheduleType] = None,
    ) -> list[dict]:
        """
        List all scheduled tasks.

        Args:
            active_only: Only return active tasks
            schedule_type: Filter by schedule type (optional)

        Returns:
            List of task dictionaries
        """
        tasks = list(self._scheduled_tasks.values())

        if active_only:
            tasks = [t for t in tasks if t.is_active]

        if schedule_type:
            tasks = [t for t in tasks if t.schedule_type == schedule_type]

        return [t.to_dict() for t in sorted(
            tasks, key=lambda t: t.created_at, reverse=True
        )]

    async def start(self) -> None:
        """
        Start the background scheduler loop.

        This should be called once at application startup.
        """
        if self._scheduler_task:
            return

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Background task scheduler started")

    async def stop(self) -> None:
        """
        Stop the background scheduler loop.

        This should be called at application shutdown.
        """
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        # Cancel all running tasks
        for task in list(self._running_tasks.values()):
            if not task.done():
                task.cancel()

        logger.info("Background task scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that checks for tasks to run."""
        try:
            while True:
                now = datetime.now()

                for task_id, task in list(self._scheduled_tasks.items()):
                    if not task.is_active:
                        continue

                    should_run = False

                    if task.schedule_type == TaskScheduleType.ONE_TIME:
                        should_run = (
                            task.run_at and now >= task.run_at and not task.last_run
                        )

                    elif task.schedule_type == TaskScheduleType.RECURRING:
                        should_run = task.next_run and now >= task.next_run

                    elif task.schedule_type == TaskScheduleType.DEFERRED:
                        # Run deferred tasks when system appears idle
                        should_run = task.next_run and now >= task.next_run

                    if should_run and task_id not in self._running_tasks:
                        self._running_tasks[task_id] = asyncio.create_task(
                            self._execute_task(task)
                        )

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")

    async def _execute_task(self, task: ScheduledTask) -> None:
        """
        Execute a single task with retry logic.

        Args:
            task: Task to execute
        """
        async with self._worker_semaphore:
            try:
                result = await self._run_task_with_retries(task)

                # Store result
                self._task_results[task.task_id] = result
                self._store_result(result)

                # Publish result event
                await self.inbox.publish(
                    AgentEvent(
                        event_type=EventType.TEXT_OUTPUT,
                        agent_name="BackgroundTaskScheduler",
                        data={
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "status": result.status.value,
                            "duration_seconds": result.duration_seconds,
                            "output": result.output,
                            "error": result.error,
                            "execution_count": result.execution_count,
                        },
                        source="background_scheduler",
                    )
                )

                # Schedule next execution if recurring
                if task.schedule_type == TaskScheduleType.RECURRING and result.status == TaskStatus.COMPLETED:
                    task.next_run = datetime.now() + timedelta(
                        seconds=task.interval_seconds or 300
                    )
                    task.last_run = result.completed_at
                    self._store_task(task)

                logger.info(
                    f"Task {task.task_id} completed in {result.duration_seconds:.2f}s"
                )

            except Exception as e:
                logger.error(f"Error executing task {task.task_id}: {e}")
                result = TaskResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    status=TaskStatus.FAILED,
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    error=str(e),
                )
                self._task_results[task.task_id] = result
                self._store_result(result)

            finally:
                # Clean up running task
                if task.task_id in self._running_tasks:
                    del self._running_tasks[task.task_id]

    async def _run_task_with_retries(
        self,
        task: ScheduledTask,
        attempt: int = 1,
    ) -> TaskResult:
        """
        Run a task with retry logic.

        Args:
            task: Task to run
            attempt: Current attempt number

        Returns:
            TaskResult with execution info
        """
        start_time = datetime.now()

        try:
            task.last_run = start_time

            # Execute function
            if asyncio.iscoroutinefunction(task.func):
                output = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=task.timeout_seconds,
                )
            else:
                output = await asyncio.wait_for(
                    asyncio.to_thread(task.func, *task.args, **task.kwargs),
                    timeout=task.timeout_seconds,
                )

            completed_at = datetime.now()

            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                status=TaskStatus.COMPLETED,
                started_at=start_time,
                completed_at=completed_at,
                duration_seconds=(completed_at - start_time).total_seconds(),
                output=str(output)[:1000] if output else None,
                execution_count=attempt,
            )

        except asyncio.TimeoutError:
            error_msg = f"Task timed out after {task.timeout_seconds}s"
            if attempt < task.max_retries:
                logger.warning(f"{error_msg}, retrying... (attempt {attempt + 1})")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                return await self._run_task_with_retries(task, attempt + 1)

            completed_at = datetime.now()
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                status=TaskStatus.FAILED,
                started_at=start_time,
                completed_at=completed_at,
                duration_seconds=(completed_at - start_time).total_seconds(),
                error=error_msg,
                execution_count=attempt,
            )

        except Exception as e:
            error_msg = str(e)
            if attempt < task.max_retries and "retryable" in error_msg.lower():
                logger.warning(f"Task error: {error_msg}, retrying... (attempt {attempt + 1})")
                await asyncio.sleep(2 ** attempt)
                return await self._run_task_with_retries(task, attempt + 1)

            completed_at = datetime.now()
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                status=TaskStatus.FAILED,
                started_at=start_time,
                completed_at=completed_at,
                duration_seconds=(completed_at - start_time).total_seconds(),
                error=error_msg,
                execution_count=attempt,
            )

    def _store_result(self, result: TaskResult) -> None:
        """Store task result in database."""
        with self._db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO task_results
                    (task_id, task_name, status, started_at, completed_at,
                     duration_seconds, output, error, execution_count, next_scheduled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.task_id,
                        result.task_name,
                        result.status.value,
                        result.started_at.isoformat(),
                        result.completed_at.isoformat() if result.completed_at else None,
                        result.duration_seconds,
                        result.output,
                        result.error,
                        result.execution_count,
                        result.next_scheduled.isoformat() if result.next_scheduled else None,
                    ),
                )

                # Also store in history
                cursor.execute(
                    """
                    INSERT INTO task_history
                    (task_id, timestamp, status, output, error, duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.task_id,
                        result.completed_at.isoformat() if result.completed_at else datetime.now().isoformat(),
                        result.status.value,
                        result.output,
                        result.error,
                        result.duration_seconds,
                    ),
                )

                conn.commit()
            finally:
                if self._conn is None: conn.close()

    def get_task_history(
        self,
        task_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get task execution history.

        Args:
            task_id: Get history for specific task (optional)
            limit: Maximum records to return

        Returns:
            List of history records
        """
        with self._db_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            try:
                if task_id:
                    cursor.execute(
                        """
                        SELECT task_id, timestamp, status, output, error,
                               duration_seconds
                        FROM task_history
                        WHERE task_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (task_id, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT task_id, timestamp, status, output, error,
                               duration_seconds
                        FROM task_history
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )

                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            finally:
                if self._conn is None: conn.close()

    async def get_statistics(self) -> dict:
        """
        Get scheduler statistics.

        Returns:
            Dict with stats about tasks and executions
        """
        all_tasks = list(self._scheduled_tasks.values())
        active_tasks = [t for t in all_tasks if t.is_active]
        completed_results = [
            r for r in self._task_results.values()
            if r.status == TaskStatus.COMPLETED
        ]

        total_duration = sum(r.duration_seconds for r in completed_results)

        return {
            "total_scheduled_tasks": len(all_tasks),
            "active_tasks": len(active_tasks),
            "running_tasks": len(self._running_tasks),
            "completed_executions": len(completed_results),
            "total_execution_time_seconds": total_duration,
            "average_execution_time_seconds": (
                total_duration / len(completed_results)
                if completed_results
                else 0
            ),
            "failed_tasks": len(
                [r for r in self._task_results.values()
                 if r.status == TaskStatus.FAILED]
            ),
        }
