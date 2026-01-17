"""FastAPI server for Task Orchestrator."""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .auth import get_current_user, TokenData
from ..agents.coordinator import (
    CoordinatorAgent,
    Task,
    TaskPriority,
    TaskSource,
    TaskStatus,
)
from ..agents.email_agent import EmailAgent
from ..agents.calendar_agent import CalendarAgent
from ..integrations.gmail import GmailClient
from ..integrations.calendar import CalendarClient
from ..core.auth import get_oauth_credentials, get_all_scopes
from ..core.config import settings

logger = logging.getLogger(__name__)

# Global coordinator instance
coordinator: Optional[CoordinatorAgent] = None

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)

# Type alias for authenticated user dependency
AuthenticatedUser = Annotated[TokenData, Depends(get_current_user)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global coordinator

    try:
        # Check if OAuth token exists before attempting to load credentials
        if not settings.oauth_token_path.exists():
            logger.warning("OAuth token not found. Starting without Google integration.")
            logger.warning("Run OAuth flow manually to enable Gmail/Calendar sync.")
            coordinator = CoordinatorAgent()
            yield
            return

        # Get OAuth credentials
        creds = get_oauth_credentials(get_all_scopes())

        # Initialize clients
        gmail_client = GmailClient(creds)
        calendar_client = CalendarClient(creds)

        # Initialize agents
        email_agent = EmailAgent(gmail_client)
        calendar_agent = CalendarAgent(calendar_client)

        # Initialize coordinator
        coordinator = CoordinatorAgent(
            email_agent=email_agent,
            calendar_agent=calendar_agent,
        )

        logger.info("Task Orchestrator initialized with Google integration")
        yield

    except FileNotFoundError as e:
        logger.warning(f"Starting without Google integration: {e}")
        coordinator = CoordinatorAgent()
        yield

    except Exception as e:
        logger.warning(f"Error initializing Google integration: {e}")
        coordinator = CoordinatorAgent()
        yield

    finally:
        logger.info("Task Orchestrator shutting down")


app = FastAPI(
    title="Task Orchestrator API",
    description="Unified task management with Gmail and Calendar integration",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter to app state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Request/Response Models ---

class TaskCreate(BaseModel):
    """Request model for creating a task."""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    priority: str = Field(default="medium")
    due_date: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    estimated_minutes: int = Field(default=30, ge=5, le=480)
    auto_schedule: bool = False


class TaskUpdate(BaseModel):
    """Request model for updating a task."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    priority: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[datetime] = None
    notes: Optional[str] = None


class TaskResponse(BaseModel):
    """Response model for a task."""
    id: str
    title: str
    description: str
    priority: str
    status: str
    source: str
    created_at: datetime
    due_date: Optional[datetime]
    scheduled_start: Optional[datetime]
    scheduled_end: Optional[datetime]
    calendar_event_id: Optional[str]
    tags: list[str]
    notes: str
    estimated_minutes: int


class SyncResponse(BaseModel):
    """Response model for sync operation."""
    tasks_created: int
    task_ids: list[str]


class ScheduleResponse(BaseModel):
    """Response model for schedule operation."""
    scheduled: bool
    event_id: Optional[str]
    start: Optional[datetime]
    end: Optional[datetime]


# --- Endpoints ---

@app.get("/")
@app.get("/health")
@limiter.limit("300/minute")  # SECURITY: Mild rate limit to prevent DoS on health checks
async def health_check(request: Request):
    """Health check endpoint - no authentication required."""
    return {
        "status": "healthy",
        "service": "task-orchestrator",
        "version": "1.0.0",
    }


@app.get("/tasks", response_model=list[TaskResponse])
@limiter.limit("100/minute")
async def list_tasks(
    request: Request,
    current_user: AuthenticatedUser,
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    source: Optional[str] = Query(None, description="Filter by source"),
):
    """List all tasks with optional filters. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    tasks = list(coordinator.tasks.values())

    # Apply filters
    if status:
        try:
            status_enum = TaskStatus(status)
            tasks = [t for t in tasks if t.status == status_enum]
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    if priority:
        try:
            priority_enum = TaskPriority(priority)
            tasks = [t for t in tasks if t.priority == priority_enum]
        except ValueError:
            raise HTTPException(400, f"Invalid priority: {priority}")

    if source:
        try:
            source_enum = TaskSource(source)
            tasks = [t for t in tasks if t.source == source_enum]
        except ValueError:
            raise HTTPException(400, f"Invalid source: {source}")

    return [_task_to_response(t) for t in tasks]


@app.post("/tasks", response_model=TaskResponse, status_code=201)
@limiter.limit("30/minute")
async def create_task(
    request: Request,
    current_user: AuthenticatedUser,
    task: TaskCreate,
):
    """Create a new task. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        priority = TaskPriority(task.priority)
    except ValueError:
        raise HTTPException(400, f"Invalid priority: {task.priority}")

    new_task = await coordinator.add_task(
        title=task.title,
        description=task.description,
        priority=priority,
        source=TaskSource.MANUAL,
        due_date=task.due_date,
        tags=task.tags,
        estimated_minutes=task.estimated_minutes,
        auto_schedule=task.auto_schedule,
    )

    return _task_to_response(new_task)


@app.get("/tasks/{task_id}", response_model=TaskResponse)
@limiter.limit("100/minute")
async def get_task(
    request: Request,
    current_user: AuthenticatedUser,
    task_id: str,
):
    """Get a specific task. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    task = coordinator.tasks.get(task_id)
    if not task:
        raise HTTPException(404, f"Task not found: {task_id}")

    return _task_to_response(task)


@app.patch("/tasks/{task_id}", response_model=TaskResponse)
@limiter.limit("30/minute")
async def update_task(
    request: Request,
    current_user: AuthenticatedUser,
    task_id: str,
    update: TaskUpdate,
):
    """Update a task. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    task = coordinator.tasks.get(task_id)
    if not task:
        raise HTTPException(404, f"Task not found: {task_id}")

    if update.title is not None:
        task.title = update.title
    if update.description is not None:
        task.description = update.description
    if update.priority is not None:
        try:
            task.priority = TaskPriority(update.priority)
        except ValueError:
            raise HTTPException(400, f"Invalid priority: {update.priority}")
    if update.status is not None:
        try:
            task.status = TaskStatus(update.status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {update.status}")
    if update.due_date is not None:
        task.due_date = update.due_date
    if update.notes is not None:
        task.notes = update.notes

    return _task_to_response(task)


@app.delete("/tasks/{task_id}", status_code=204)
@limiter.limit("30/minute")
async def delete_task(
    request: Request,
    current_user: AuthenticatedUser,
    task_id: str,
):
    """Delete a task. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if task_id not in coordinator.tasks:
        raise HTTPException(404, f"Task not found: {task_id}")

    del coordinator.tasks[task_id]


@app.post("/tasks/{task_id}/complete", response_model=TaskResponse)
@limiter.limit("30/minute")
async def complete_task(
    request: Request,
    current_user: AuthenticatedUser,
    task_id: str,
    notes: str = "",
):
    """Mark a task as completed. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        task = await coordinator.complete_task(task_id, notes)
        return _task_to_response(task)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/tasks/{task_id}/schedule", response_model=ScheduleResponse)
@limiter.limit("30/minute")
async def schedule_task(
    request: Request,
    current_user: AuthenticatedUser,
    task_id: str,
    preferred_time: Optional[datetime] = None,
):
    """Schedule a task on the calendar. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        scheduled = await coordinator.schedule_task(task_id, preferred_time)

        if scheduled:
            return ScheduleResponse(
                scheduled=True,
                event_id=scheduled.event_id,
                start=scheduled.start,
                end=scheduled.end,
            )
        else:
            return ScheduleResponse(
                scheduled=False,
                event_id=None,
                start=None,
                end=None,
            )

    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/sync/email", response_model=SyncResponse)
@limiter.limit("10/minute")
async def sync_from_email(
    request: Request,
    current_user: AuthenticatedUser,
):
    """Sync tasks from unread emails. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        new_tasks = await coordinator.sync_from_email()
        return SyncResponse(
            tasks_created=len(new_tasks),
            task_ids=[t.id for t in new_tasks],
        )
    except ValueError as e:
        raise HTTPException(503, str(e))


@app.get("/summary/daily")
@limiter.limit("30/minute")
async def get_daily_summary(
    request: Request,
    current_user: AuthenticatedUser,
):
    """Get daily summary of tasks and schedule. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return await coordinator.get_daily_summary()


@app.get("/tasks/prioritized", response_model=list[TaskResponse])
@limiter.limit("60/minute")
async def get_prioritized_tasks(
    request: Request,
    current_user: AuthenticatedUser,
):
    """Get tasks sorted by priority. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    tasks = await coordinator.prioritize_tasks()
    return [_task_to_response(t) for t in tasks]


@app.get("/tasks/overdue", response_model=list[TaskResponse])
@limiter.limit("60/minute")
async def get_overdue_tasks(
    request: Request,
    current_user: AuthenticatedUser,
):
    """Get overdue tasks. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    tasks = coordinator.get_overdue_tasks()
    return [_task_to_response(t) for t in tasks]


@app.post("/schedule/auto")
@limiter.limit("10/minute")
async def auto_schedule_tasks(
    request: Request,
    current_user: AuthenticatedUser,
    max_tasks: int = 5,
):
    """Automatically schedule pending tasks. Requires authentication."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    scheduled = await coordinator.auto_schedule_pending(max_tasks)
    return {
        "scheduled_count": len(scheduled),
        "tasks": [
            {
                "task_id": s.task_id,
                "title": s.title,
                "start": s.start,
                "end": s.end,
            }
            for s in scheduled
        ],
    }


@app.post("/focus/block")
@limiter.limit("10/minute")
async def block_focus_time(
    request: Request,
    current_user: AuthenticatedUser,
    duration_minutes: int = 120,
    days_ahead: int = 5,
):
    """Block focus time on calendar. Requires authentication."""
    if coordinator is None or coordinator.calendar_agent is None:
        raise HTTPException(503, "Calendar not available")

    events = coordinator.calendar_agent.block_focus_time(
        duration_minutes=duration_minutes,
        days_ahead=days_ahead,
    )

    return {
        "blocked_count": len(events),
        "events": [
            {
                "id": e.id,
                "start": e.start,
                "end": e.end,
            }
            for e in events
        ],
    }


# --- Helper Functions ---

def _task_to_response(task: Task) -> TaskResponse:
    """Convert Task to TaskResponse."""
    return TaskResponse(
        id=task.id,
        title=task.title,
        description=task.description,
        priority=task.priority.value,
        status=task.status.value,
        source=task.source.value,
        created_at=task.created_at,
        due_date=task.due_date,
        scheduled_start=task.scheduled_start,
        scheduled_end=task.scheduled_end,
        calendar_event_id=task.calendar_event_id,
        tags=task.tags,
        notes=task.notes,
        estimated_minutes=task.estimated_minutes,
    )


# Run with: uvicorn src.api.server:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
