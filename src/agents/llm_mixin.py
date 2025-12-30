"""LLM-powered methods for the coordinator agent."""
import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..llm import ModelRouter, TaskType


class LLMMixin:
    """Mixin class providing LLM-powered methods for CoordinatorAgent."""

    async def analyze_task_with_llm(self, task_id: str) -> dict:
        """
        Use LLM to analyze a task and provide insights.

        Args:
            task_id: Task to analyze

        Returns:
            Dict with analysis results
        """
        if not self.llm:
            raise ValueError("LLM router not configured")

        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        due_str = task.due_date.isoformat() if task.due_date else "Not set"
        tags_str = ", ".join(task.tags) if task.tags else "None"

        prompt = f"""Analyze this task and provide insights:

Task: {task.title}
Description: {task.description}
Priority: {task.priority.value}
Due: {due_str}
Tags: {tags_str}

Provide:
1. Estimated time to complete (in minutes)
2. Suggested subtasks (if complex)
3. Potential blockers or dependencies
4. Priority recommendation (low/medium/high/critical)
5. Any additional tags that should be added

Respond in JSON format."""

        try:
            from ..llm import TaskType
            task_type = TaskType.ANALYSIS
        except ImportError:
            task_type = None

        response = await self.llm.generate(
            prompt,
            task_type=task_type,
            temperature=0.3,
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"raw_analysis": response.content}

    async def suggest_task_priority(
        self,
        title: str,
        description: str = "",
        context: str = "",
    ):
        """
        Use LLM to suggest priority for a new task.

        Args:
            title: Task title
            description: Task description
            context: Additional context

        Returns:
            Suggested TaskPriority
        """
        from .email_agent import TaskPriority

        if not self.llm:
            return TaskPriority.MEDIUM

        prompt = f"""Based on this task, suggest a priority level (low, medium, high, critical):

Task: {title}
Description: {description}
Context: {context}

Consider:
- Urgency and deadlines
- Business impact
- Dependencies from other work
- Complexity

Respond with just the priority level: low, medium, high, or critical"""

        try:
            from ..llm import TaskType
            task_type = TaskType.FAST_RESPONSE
        except ImportError:
            task_type = None

        response = await self.llm.generate(
            prompt,
            task_type=task_type,
            prefer_fast=True,
            temperature=0.1,
        )

        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
        }

        priority_str = response.content.strip().lower()
        return priority_map.get(priority_str, TaskPriority.MEDIUM)

    async def generate_task_breakdown(
        self,
        task_id: str,
        max_subtasks: int = 5,
    ) -> list[dict]:
        """
        Use LLM to break down a complex task into subtasks.

        Args:
            task_id: Task to break down
            max_subtasks: Maximum subtasks to generate

        Returns:
            List of subtask dicts with title, description, estimated_minutes
        """
        if not self.llm:
            raise ValueError("LLM router not configured")

        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        prompt = f"""Break down this task into {max_subtasks} or fewer subtasks:

Task: {task.title}
Description: {task.description}
Estimated total time: {task.estimated_minutes} minutes

For each subtask provide:
- title: Brief title
- description: What needs to be done
- estimated_minutes: Time estimate

Respond as a JSON array of subtask objects."""

        try:
            from ..llm import TaskType
            task_type = TaskType.REASONING
        except ImportError:
            task_type = None

        response = await self.llm.generate(
            prompt,
            task_type=task_type,
            temperature=0.5,
        )

        try:
            subtasks = json.loads(response.content)
            if isinstance(subtasks, list):
                return subtasks[:max_subtasks]
            return []
        except json.JSONDecodeError:
            return []

    async def get_ai_daily_briefing(self) -> str:
        """
        Generate an AI-powered daily briefing of tasks.

        Returns:
            Natural language briefing string
        """
        if not self.llm:
            return "LLM not configured for briefings."

        summary = await self.get_daily_summary()
        prioritized = await self.prioritize_tasks()

        task_lines = []
        for t in prioritized[:10]:
            due_part = ""
            if t.due_date:
                due_part = f" (due: {t.due_date.strftime('%m/%d')})"
            task_lines.append(f"- [{t.priority.value.upper()}] {t.title}{due_part}")

        tasks_text = "\n".join(task_lines)

        calendar_info = summary.get("calendar", "No calendar data")

        prompt = f"""Generate a brief, actionable daily briefing based on this task summary:

Total active tasks: {summary['total_active_tasks']}
Overdue tasks: {summary['overdue_count']}

Top tasks:
{tasks_text}

Calendar: {calendar_info}

Provide:
1. A quick summary of the day
2. Top 3 priorities to focus on
3. Any warnings about overdue items
4. A motivational closing

Keep it concise and actionable."""

        try:
            from ..llm import TaskType
            task_type = TaskType.CREATIVE
        except ImportError:
            task_type = None

        response = await self.llm.generate(
            prompt,
            task_type=task_type,
            temperature=0.7,
        )

        return response.content
