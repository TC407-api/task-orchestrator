"""Task management handler methods for MCP server."""
from datetime import datetime

from ..agents.coordinator import TaskStatus
from ..agents.email_agent import TaskPriority
from ..core.cost_tracker import Provider
from ..observability import trace_operation


class TaskHandlers:
    """Mixin providing task-related MCP handler methods."""

    @trace_operation("tasks_list")
    async def _handle_tasks_list(self, args: dict) -> dict:
        """List tasks."""
        status_filter = args.get("status", "all")
        limit = args.get("limit", 10)

        if status_filter == "all":
            tasks = await self.coordinator.prioritize_tasks()
        else:
            status = TaskStatus(status_filter)
            tasks = self.coordinator.get_tasks_by_status(status)

        return {
            "count": len(tasks[:limit]),
            "tasks": [t.to_dict() for t in tasks[:limit]],
        }

    @trace_operation("tasks_add")
    async def _handle_tasks_add(self, args: dict) -> dict:
        """Add a new task."""
        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
        }

        due_date = None
        if args.get("due_date"):
            due_date = datetime.fromisoformat(args["due_date"])

        task = await self.coordinator.add_task(
            title=args["title"],
            description=args.get("description", ""),
            priority=priority_map.get(args.get("priority", "medium"), TaskPriority.MEDIUM),
            due_date=due_date,
            tags=args.get("tags", []),
            estimated_minutes=args.get("estimated_minutes", 30),
            auto_schedule=args.get("auto_schedule", False),
        )

        return {"success": True, "task": task.to_dict()}

    @trace_operation("tasks_sync_email")
    async def _handle_tasks_sync_email(self, args: dict) -> dict:
        """Sync from email."""
        can_proceed, retry_after = self._gmail_breaker.is_available()
        if not can_proceed:
            return {
                "error": f"Gmail service circuit breaker is open. Retry after {retry_after:.1f}s",
                "circuit_breaker_open": True,
                "retry_after_seconds": retry_after,
            }

        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GMAIL)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.email_agent:
            return {"error": "Email agent not configured. Run oauth_setup.py first."}

        try:
            new_tasks = await self.coordinator.sync_from_email()
            self._gmail_breaker.record_success()
            await self.cost_tracker.record_usage(
                provider=Provider.GOOGLE_GMAIL,
                operation="sync_email",
                metadata={"tasks_created": len(new_tasks)},
            )
            return {
                "success": True,
                "new_tasks_count": len(new_tasks),
                "tasks": [t.to_dict() for t in new_tasks],
            }
        except Exception as e:
            self._gmail_breaker.record_failure(e)
            raise

    @trace_operation("tasks_schedule")
    async def _handle_tasks_schedule(self, args: dict) -> dict:
        """Schedule a task."""
        can_proceed, retry_after = self._calendar_breaker.is_available()
        if not can_proceed:
            return {
                "error": f"Calendar service circuit breaker is open. Retry after {retry_after:.1f}s",
                "circuit_breaker_open": True,
                "retry_after_seconds": retry_after,
            }

        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_CALENDAR)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        preferred_time = None
        if args.get("preferred_time"):
            preferred_time = datetime.fromisoformat(args["preferred_time"])

        try:
            scheduled = await self.coordinator.schedule_task(
                args["task_id"],
                preferred_time=preferred_time,
            )

            if scheduled:
                self._calendar_breaker.record_success()
                await self.cost_tracker.record_usage(
                    provider=Provider.GOOGLE_CALENDAR,
                    operation="schedule_task",
                )
                return {"success": True, "scheduled": True, "event_id": scheduled.event_id}

            return {"success": False, "message": "Could not find available slot"}
        except Exception as e:
            self._calendar_breaker.record_failure(e)
            raise

    @trace_operation("tasks_complete")
    async def _handle_tasks_complete(self, args: dict) -> dict:
        """Complete a task."""
        task = await self.coordinator.complete_task(
            args["task_id"],
            notes=args.get("notes", ""),
        )
        return {"success": True, "task": task.to_dict()}

    @trace_operation("tasks_analyze")
    async def _handle_tasks_analyze(self, args: dict) -> dict:
        """AI task analysis."""
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.llm:
            return {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}

        analysis = await self.coordinator.analyze_task_with_llm(args["task_id"])

        await self.cost_tracker.record_usage(
            provider=Provider.GOOGLE_GEMINI,
            operation="analyze_task",
            input_tokens=500,
            output_tokens=300,
            model="gemini-2.5-flash",
        )

        return {"success": True, "analysis": analysis}

    @trace_operation("tasks_briefing")
    async def _handle_tasks_briefing(self, args: dict) -> dict:
        """AI daily briefing."""
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.llm:
            return {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}

        briefing = await self.coordinator.get_ai_daily_briefing()

        await self.cost_tracker.record_usage(
            provider=Provider.GOOGLE_GEMINI,
            operation="daily_briefing",
            input_tokens=800,
            output_tokens=500,
            model="gemini-2.5-flash",
        )

        return {"success": True, "briefing": briefing}
