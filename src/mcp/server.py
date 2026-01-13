"""MCP Server for Task Orchestrator integration with Claude Code."""
import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Optional

from ..core.cost_tracker import Provider, get_cost_tracker
from ..agents.coordinator import CoordinatorAgent, TaskStatus
from ..agents.email_agent import TaskPriority
from ..agents.archetype_registry import (
    Archetype,
    ArchetypeRegistry,
    get_archetype_registry,
)
from ..agents.audit_workflow import AuditWorkflow
from ..agents.inbox import (
    UniversalInbox,
    AgentEvent,
    EventType,
    PendingAction,
    ActionRiskLevel,
    ApprovalStatus,
)
from ..observability import trace_operation
from ..self_healing import (
    CircuitBreaker,
    get_healing_status,
)


class TaskOrchestratorMCP:
    """
    MCP Server exposing task orchestrator functionality.

    Tools:
    - tasks_list: List and prioritize tasks
    - tasks_add: Create a new task
    - tasks_sync_email: Pull tasks from Gmail
    - tasks_schedule: Schedule task on calendar
    - tasks_complete: Mark task complete
    - tasks_analyze: AI analysis of a task
    - tasks_briefing: Get AI daily briefing
    - cost_summary: View API cost summary
    - cost_check: Check if API call is within budget
    """

    def __init__(self):
        self.coordinator: Optional[CoordinatorAgent] = None
        self.cost_tracker = get_cost_tracker()
        self._initialized = False

        # Initialize circuit breakers for external services
        self._gmail_breaker = CircuitBreaker.get("gmail_service")
        self._calendar_breaker = CircuitBreaker.get("calendar_service")
        self._gemini_breaker = CircuitBreaker.get("gemini_service")

        # Initialize archetype-based agent components (yoink features)
        self._archetype_registry = get_archetype_registry()
        self._universal_inbox = UniversalInbox()
        self._audit_workflow: Optional[AuditWorkflow] = None

    async def initialize(self):
        """Initialize the coordinator with agents."""
        if self._initialized:
            return

        # Import here to avoid circular imports
        from ..llm import GeminiProvider, ModelRouter

        # Check budget before initializing LLM
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            raise RuntimeError(f"Cannot initialize: {msg}")

        try:
            provider = GeminiProvider()
            router = ModelRouter({"google": provider})
            self.coordinator = CoordinatorAgent(llm_router=router)
            self._initialized = True
        except ValueError:
            # API key not set - coordinator without LLM
            self.coordinator = CoordinatorAgent()
            self._initialized = True

    def get_tools(self) -> list[dict]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "tasks_list",
                "description": "List tasks sorted by priority. Returns pending, scheduled, and in-progress tasks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["all", "pending", "scheduled", "in_progress", "completed"],
                            "description": "Filter by status (default: all active)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max tasks to return (default: 10)",
                        },
                    },
                },
            },
            {
                "name": "tasks_add",
                "description": "Create a new task in the system.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Task title"},
                        "description": {"type": "string", "description": "Task details"},
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "Priority level (default: medium)",
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date in ISO format (optional)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization",
                        },
                        "estimated_minutes": {
                            "type": "integer",
                            "description": "Estimated time in minutes (default: 30)",
                        },
                        "auto_schedule": {
                            "type": "boolean",
                            "description": "Auto-schedule on calendar (default: false)",
                        },
                    },
                    "required": ["title"],
                },
            },
            {
                "name": "tasks_sync_email",
                "description": "Sync tasks from unread emails. Extracts actionable items from Gmail.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_emails": {
                            "type": "integer",
                            "description": "Max emails to process (default: 10)",
                        },
                    },
                },
            },
            {
                "name": "tasks_schedule",
                "description": "Schedule a task on Google Calendar.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID to schedule"},
                        "preferred_time": {
                            "type": "string",
                            "description": "Preferred start time in ISO format (optional)",
                        },
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "tasks_complete",
                "description": "Mark a task as completed.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID to complete"},
                        "notes": {"type": "string", "description": "Completion notes"},
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "tasks_analyze",
                "description": "Use AI to analyze a task and get insights (estimated time, subtasks, blockers).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID to analyze"},
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "tasks_briefing",
                "description": "Get an AI-generated daily briefing of tasks and priorities.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "cost_summary",
                "description": "View API cost summary across all providers (Gemini, OpenAI, etc.).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "provider": {
                            "type": "string",
                            "enum": ["all", "google_gemini", "openai", "graphiti"],
                            "description": "Filter by provider (default: all)",
                        },
                    },
                },
            },
            {
                "name": "cost_set_budget",
                "description": "Set daily/monthly budget limits for a provider.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "provider": {
                            "type": "string",
                            "enum": ["google_gemini", "openai", "graphiti", "google_gmail", "google_calendar"],
                            "description": "Provider to configure",
                        },
                        "daily_limit": {
                            "type": "number",
                            "description": "Daily budget in USD",
                        },
                        "monthly_limit": {
                            "type": "number",
                            "description": "Monthly budget in USD",
                        },
                    },
                    "required": ["provider"],
                },
            },
            {
                "name": "healing_status",
                "description": "Get self-healing system status including circuit breakers and retry state.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "spawn_agent",
                "description": "Spawn a Gemini agent to execute a code task. Returns the agent's response.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Task prompt for the agent"},
                        "model": {
                            "type": "string",
                            "enum": ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-flash"],
                            "description": "Model to use (default: gemini-3-flash-preview)",
                        },
                        "system_prompt": {"type": "string", "description": "Optional system prompt"},
                        "max_tokens": {"type": "integer", "description": "Max output tokens (default: 8192)"},
                        "working_dir": {"type": "string", "description": "Working directory context"},
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "spawn_parallel_agents",
                "description": "Spawn multiple Gemini agents in parallel to execute code tasks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of task prompts",
                        },
                        "model": {
                            "type": "string",
                            "enum": ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-flash"],
                            "description": "Model for all agents (default: gemini-3-flash-preview)",
                        },
                        "system_prompt": {"type": "string", "description": "Shared system prompt"},
                        "max_tokens": {"type": "integer", "description": "Max output tokens per agent (default: 8192)"},
                    },
                    "required": ["prompts"],
                },
            },
            {
                "name": "immune_status",
                "description": "Get immune system health and statistics including failure patterns and guardrail effectiveness.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "immune_check",
                "description": "Pre-check a prompt for risks without executing it. Returns risk score and suggestions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The prompt to evaluate"},
                        "operation": {
                            "type": "string",
                            "description": "Operation type (spawn_agent, spawn_parallel_agent)",
                            "default": "spawn_agent",
                        },
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "immune_failures",
                "description": "List recent failure patterns stored in the immune system memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum patterns to return (default: 10)",
                            "default": 10,
                        },
                    },
                },
            },
            {
                "name": "immune_dashboard",
                "description": "Get a comprehensive dashboard report of the immune system including health metrics, failure trends, and top patterns.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "json"],
                            "description": "Output format (default: markdown)",
                            "default": "markdown",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days for trend analysis (default: 7)",
                            "default": 7,
                        },
                    },
                },
            },
            {
                "name": "immune_sync",
                "description": "Synchronize immune system patterns with Graphiti for persistent cross-session memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "alert_list",
                "description": "List recent alerts from the alerting system. Shows high-risk patterns, frequency spikes, and consecutive failures.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum alerts to return (default: 10)",
                            "default": 10,
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "critical"],
                            "description": "Filter by severity level (optional)",
                        },
                    },
                },
            },
            {
                "name": "alert_clear",
                "description": "Clear all active alerts from the alerting system.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "predict_risk",
                "description": "Use ML model to predict failure risk for a prompt before execution.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The prompt to analyze"},
                        "tool": {
                            "type": "string",
                            "description": "Tool being used (spawn_agent, spawn_parallel_agents)",
                            "default": "spawn_agent",
                        },
                    },
                    "required": ["prompt"],
                },
            },
            # Federation Tools (Phase 9)
            {
                "name": "federation_status",
                "description": "Get federation health status, subscriptions, and shared patterns across portfolio projects.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_projects": {
                            "type": "boolean",
                            "description": "Include full project details (default: false)",
                            "default": False,
                        },
                    },
                },
            },
            {
                "name": "federation_subscribe",
                "description": "Subscribe to patterns from another portfolio project.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID to subscribe to (e.g., 'construction-connect')",
                        },
                    },
                    "required": ["project_id"],
                },
            },
            {
                "name": "federation_search",
                "description": "Search for patterns across subscribed federated projects.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for patterns",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "federation_decay",
                "description": "Evaluate pattern decay status and identify stale/prunable patterns.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "evaluate", "prune_candidates"],
                            "description": "Action to perform (default: status)",
                            "default": "status",
                        },
                    },
                },
            },
            # Live Sync Tools (Phase 10)
            {
                "name": "sync_status",
                "description": "Get live sync health status for all federated projects.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Filter to specific project (optional)",
                        },
                    },
                },
            },
            {
                "name": "sync_trigger",
                "description": "Trigger a manual sync cycle for federated patterns.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["push", "pull", "both"],
                            "description": "Sync direction (default: both)",
                            "default": "both",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Target specific project (optional)",
                        },
                    },
                },
            },
            {
                "name": "sync_alerts",
                "description": "Get sync-related alerts for federation health issues.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["healthy", "degraded", "critical"],
                            "description": "Filter by severity (optional)",
                        },
                    },
                },
            },
            # Archetype Agent Tools (Yoinked from Anti-gravity)
            {
                "name": "spawn_archetype_agent",
                "description": "Spawn an agent with a specific archetype role (architect, builder, qc, researcher). Each archetype has filtered tools and role-specific system prompts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "archetype": {
                            "type": "string",
                            "enum": ["architect", "builder", "qc", "researcher"],
                            "description": "Agent archetype role determining tools and behavior",
                        },
                        "prompt": {"type": "string", "description": "Task prompt for the agent"},
                        "model": {
                            "type": "string",
                            "enum": ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-flash"],
                            "description": "Model to use (default: gemini-3-flash-preview)",
                        },
                        "inject_audit": {
                            "type": "boolean",
                            "description": "Inject audit history into agent context (default: true)",
                        },
                        "max_tokens": {"type": "integer", "description": "Max output tokens (default: 8192)"},
                    },
                    "required": ["archetype", "prompt"],
                },
            },
            {
                "name": "inbox_status",
                "description": "Get universal inbox status including pending approvals and recent events.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "risk_level": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                            "description": "Filter pending approvals by risk level (optional)",
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Filter by agent name (optional)",
                        },
                        "include_history": {
                            "type": "boolean",
                            "description": "Include recent event history (default: false)",
                        },
                        "history_limit": {
                            "type": "integer",
                            "description": "Max history events to return (default: 20)",
                        },
                    },
                },
            },
            {
                "name": "approve_action",
                "description": "Approve or reject a pending action in the approval queue.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action_id": {"type": "string", "description": "ID of the action to approve/reject"},
                        "approve": {
                            "type": "boolean",
                            "description": "True to approve, False to reject",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for rejection (required if rejecting)",
                        },
                        "approved_by": {
                            "type": "string",
                            "description": "User approving/rejecting (default: system)",
                        },
                    },
                    "required": ["action_id", "approve"],
                },
            },
            {
                "name": "audit_status",
                "description": "Get audit workflow status including decisions, errors, and patterns.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_root": {
                            "type": "string",
                            "description": "Project root to load audit.md from (default: cwd)",
                        },
                        "query_topic": {
                            "type": "string",
                            "description": "Search for decisions matching this topic (optional)",
                        },
                        "query_error_type": {
                            "type": "string",
                            "description": "Filter errors by type (runtime, logic, api, etc.)",
                        },
                    },
                },
            },
            {
                "name": "audit_append",
                "description": "Append a new entry to the audit log (decision, error, or pattern).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entry_type": {
                            "type": "string",
                            "enum": ["decision", "error", "pattern"],
                            "description": "Type of entry to append",
                        },
                        "title": {"type": "string", "description": "Title of the entry"},
                        "content": {"type": "string", "description": "Main content/description"},
                        "project_root": {
                            "type": "string",
                            "description": "Project root for audit.md (default: cwd)",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata (severity, context, etc.)",
                        },
                    },
                    "required": ["entry_type", "content"],
                },
            },
            {
                "name": "archetype_info",
                "description": "Get information about available agent archetypes and their tool permissions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "archetype": {
                            "type": "string",
                            "enum": ["architect", "builder", "qc", "researcher"],
                            "description": "Get details for specific archetype (optional, returns all if not specified)",
                        },
                    },
                },
            },
        ]

    async def handle_tool_call(self, name: str, arguments: dict) -> Any:
        """Handle an MCP tool call."""
        await self.initialize()

        handlers = {
            "tasks_list": self._handle_tasks_list,
            "tasks_add": self._handle_tasks_add,
            "tasks_sync_email": self._handle_tasks_sync_email,
            "tasks_schedule": self._handle_tasks_schedule,
            "tasks_complete": self._handle_tasks_complete,
            "tasks_analyze": self._handle_tasks_analyze,
            "tasks_briefing": self._handle_tasks_briefing,
            "cost_summary": self._handle_cost_summary,
            "cost_set_budget": self._handle_cost_set_budget,
            "healing_status": self._handle_healing_status,
            "spawn_agent": self._handle_spawn_agent,
            "spawn_parallel_agents": self._handle_spawn_parallel_agents,
            "immune_status": self._handle_immune_status,
            "immune_check": self._handle_immune_check,
            "immune_failures": self._handle_immune_failures,
            "immune_dashboard": self._handle_immune_dashboard,
            "immune_sync": self._handle_immune_sync,
            "alert_list": self._handle_alert_list,
            "alert_clear": self._handle_alert_clear,
            "predict_risk": self._handle_predict_risk,
            "federation_status": self._handle_federation_status,
            "federation_subscribe": self._handle_federation_subscribe,
            "federation_search": self._handle_federation_search,
            "federation_decay": self._handle_federation_decay,
            # Live Sync handlers
            "sync_status": self._handle_sync_status,
            "sync_trigger": self._handle_sync_trigger,
            "sync_alerts": self._handle_sync_alerts,
            # Archetype Agent handlers (Yoinked from Anti-gravity)
            "spawn_archetype_agent": self._handle_spawn_archetype_agent,
            "inbox_status": self._handle_inbox_status,
            "approve_action": self._handle_approve_action,
            "audit_status": self._handle_audit_status,
            "audit_append": self._handle_audit_append,
            "archetype_info": self._handle_archetype_info,
        }

        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}

        try:
            return await handler(arguments)
        except Exception as e:
            return {"error": str(e)}

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
        # Check circuit breaker first
        can_proceed, retry_after = self._gmail_breaker.is_available()
        if not can_proceed:
            return {
                "error": f"Gmail service circuit breaker is open. Retry after {retry_after:.1f}s",
                "circuit_breaker_open": True,
                "retry_after_seconds": retry_after,
            }

        # Check budget
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GMAIL)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.email_agent:
            return {"error": "Email agent not configured. Run oauth_setup.py first."}

        try:
            new_tasks = await self.coordinator.sync_from_email()

            # Record success in circuit breaker
            self._gmail_breaker.record_success()

            # Track usage
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
            # Record failure in circuit breaker
            self._gmail_breaker.record_failure(e)
            raise

    @trace_operation("tasks_schedule")
    async def _handle_tasks_schedule(self, args: dict) -> dict:
        """Schedule a task."""
        # Check circuit breaker first
        can_proceed, retry_after = self._calendar_breaker.is_available()
        if not can_proceed:
            return {
                "error": f"Calendar service circuit breaker is open. Retry after {retry_after:.1f}s",
                "circuit_breaker_open": True,
                "retry_after_seconds": retry_after,
            }

        # Check budget
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
                # Record success in circuit breaker
                self._calendar_breaker.record_success()

                # Track usage
                await self.cost_tracker.record_usage(
                    provider=Provider.GOOGLE_CALENDAR,
                    operation="schedule_task",
                )
                return {"success": True, "scheduled": True, "event_id": scheduled.event_id}

            return {"success": False, "message": "Could not find available slot"}
        except Exception as e:
            # Record failure in circuit breaker
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
        # Check budget
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.llm:
            return {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}

        analysis = await self.coordinator.analyze_task_with_llm(args["task_id"])

        # Track usage (estimate since we don't have exact tokens here)
        await self.cost_tracker.record_usage(
            provider=Provider.GOOGLE_GEMINI,
            operation="analyze_task",
            input_tokens=500,  # Estimate
            output_tokens=300,
            model="gemini-2.5-flash",
        )

        return {"success": True, "analysis": analysis}

    @trace_operation("tasks_briefing")
    async def _handle_tasks_briefing(self, args: dict) -> dict:
        """AI daily briefing."""
        # Check budget
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.llm:
            return {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}

        briefing = await self.coordinator.get_ai_daily_briefing()

        # Track usage
        await self.cost_tracker.record_usage(
            provider=Provider.GOOGLE_GEMINI,
            operation="daily_briefing",
            input_tokens=800,
            output_tokens=500,
            model="gemini-2.5-flash",
        )

        return {"success": True, "briefing": briefing}

    @trace_operation("cost_summary")
    async def _handle_cost_summary(self, args: dict) -> dict:
        """Get cost summary."""
        summary = self.cost_tracker.get_summary()

        provider_filter = args.get("provider", "all")
        if provider_filter != "all":
            # Filter to specific provider
            filtered = {
                "generated_at": summary["generated_at"],
                "providers": {provider_filter: summary["providers"].get(provider_filter, {})},
                "totals": summary["totals"],
            }
            return filtered

        return summary

    @trace_operation("cost_set_budget")
    async def _handle_cost_set_budget(self, args: dict) -> dict:
        """Set budget limits."""
        provider = Provider(args["provider"])

        self.cost_tracker.set_budget(
            provider,
            daily_limit=args.get("daily_limit"),
            monthly_limit=args.get("monthly_limit"),
        )

        return {
            "success": True,
            "provider": provider.value,
            "new_limits": {
                "daily": self.cost_tracker.budgets[provider].daily_limit_usd,
                "monthly": self.cost_tracker.budgets[provider].monthly_limit_usd,
            },
        }

    @trace_operation("healing_status")
    async def _handle_healing_status(self, args: dict) -> dict:
        """Get self-healing system status."""
        status = get_healing_status()

        # Add local circuit breaker status
        status["circuit_breakers"]["gmail_service"] = self._gmail_breaker.get_stats()
        status["circuit_breakers"]["calendar_service"] = self._calendar_breaker.get_stats()
        status["circuit_breakers"]["gemini_service"] = self._gemini_breaker.get_stats()

        return {
            "success": True,
            "healing_status": status,
        }

    @trace_operation("immune_status")
    async def _handle_immune_status(self, args: dict) -> dict:
        """Get immune system health and statistics."""
        from ..evaluation import get_immune_system

        immune = get_immune_system()
        stats = immune.get_stats()
        health = immune.get_health()

        return {
            "success": True,
            "immune_system": {
                "health": health,
                "statistics": stats,
            },
        }

    @trace_operation("immune_check")
    async def _handle_immune_check(self, args: dict) -> dict:
        """Pre-check a prompt for risks without executing."""
        from ..evaluation import get_immune_system

        prompt = args.get("prompt")
        if not prompt:
            return {"error": "Prompt is required"}

        operation = args.get("operation", "spawn_agent")
        immune = get_immune_system()

        # Get suggestions without actually processing
        suggestions = await immune.get_suggestions(prompt, operation)

        # Also do a pre-spawn check to get full response
        response = await immune.pre_spawn_check(prompt, operation)

        return {
            "success": True,
            "risk_assessment": {
                "risk_score": response.risk_score,
                "should_proceed": response.should_proceed,
                "warnings": response.warnings,
                "guardrails_would_apply": response.guardrails_applied,
                "prompt_would_be_modified": response.original_prompt != response.processed_prompt,
            },
            "suggestions": suggestions,
        }

    @trace_operation("immune_failures")
    async def _handle_immune_failures(self, args: dict) -> dict:
        """List recent failure patterns."""
        from ..evaluation import get_immune_system

        limit = args.get("limit", 10)
        immune = get_immune_system()

        # Get failure store stats
        stats = immune.get_stats()
        failure_stats = stats.get("failure_store", {})

        # Get recent failures from the store
        recent = await immune._failure_store.get_recent_failures(limit=limit)

        return {
            "success": True,
            "failure_patterns": {
                "total_patterns": failure_stats.get("total_patterns", 0),
                "total_occurrences": failure_stats.get("total_occurrences", 0),
                "by_type": failure_stats.get("by_type", {}),
                "by_operation": failure_stats.get("by_operation", {}),
                "recent": [p.to_dict() for p in recent],
            },
        }

    @trace_operation("immune_dashboard")
    async def _handle_immune_dashboard(self, args: dict) -> dict:
        """Get comprehensive immune system dashboard."""
        from ..evaluation import get_immune_system
        from ..evaluation.immune_system import create_dashboard

        format_type = args.get("format", "markdown")
        immune = get_immune_system()
        dashboard = create_dashboard(immune)

        if format_type == "json":
            return {
                "success": True,
                "format": "json",
                "report": dashboard.format_as_json(),
            }
        else:
            return {
                "success": True,
                "format": "markdown",
                "report": dashboard.format_as_markdown(),
            }

    @trace_operation("immune_sync")
    async def _handle_immune_sync(self, args: dict) -> dict:
        """Synchronize immune system with Graphiti."""
        from ..evaluation import get_immune_system

        immune = get_immune_system()

        try:
            result = await immune.sync_with_graphiti()
            return {
                "success": True,
                "sync_result": result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    @trace_operation("alert_list")
    async def _handle_alert_list(self, args: dict) -> dict:
        """List recent alerts from the alerting system."""
        from ..evaluation import AlertManager, HighRiskThreshold, NewPatternDetected

        # Get or create alert manager (singleton pattern)
        if not hasattr(self, '_alert_manager'):
            self._alert_manager = AlertManager(
                rules=[HighRiskThreshold(), NewPatternDetected()]
            )

        limit = args.get("limit", 10)
        severity = args.get("severity")

        if severity:
            alerts = self._alert_manager.get_active_alerts(severity=severity)
        else:
            alerts = self._alert_manager.get_recent_alerts(limit=limit)

        return {
            "success": True,
            "alerts": alerts,
            "stats": self._alert_manager.get_stats(),
        }

    @trace_operation("alert_clear")
    async def _handle_alert_clear(self, args: dict) -> dict:
        """Clear all active alerts."""
        if not hasattr(self, '_alert_manager'):
            return {"success": True, "cleared": 0}

        cleared = self._alert_manager.clear_active_alerts()
        return {
            "success": True,
            "cleared": cleared,
        }

    @trace_operation("predict_risk")
    async def _handle_predict_risk(self, args: dict) -> dict:
        """Predict failure risk for a prompt using ML model."""
        from ..evaluation import FailurePredictor

        prompt = args["prompt"]
        tool = args.get("tool", "spawn_agent")

        # Get or create predictor (singleton pattern)
        if not hasattr(self, '_predictor'):
            self._predictor = FailurePredictor()

        result = self._predictor.predict(prompt, tool)

        return {
            "success": True,
            "prediction": result.to_dict(),
            "model_active": self._predictor.is_active,
        }

    # =========================================================================
    # Federation Tools (Phase 9)
    # =========================================================================

    @trace_operation("federation_status")
    async def _handle_federation_status(self, args: dict) -> dict:
        """Get federation status across portfolio projects."""
        from ..evaluation.immune_system import (
            get_registry_manager,
            PatternFederation,
            get_decay_system,
        )

        # Get or create registry (lazy singleton)
        if not hasattr(self, '_registry'):
            self._registry = get_registry_manager()

        # Get or create federation
        if not hasattr(self, '_federation'):
            self._federation = PatternFederation(
                graphiti_client=None,  # Will be set if available
                local_group_id="project_task_orchestrator",
            )

        include_projects = args.get("include_projects", False)

        result = {
            "success": True,
            "registry": self._registry.get_stats(),
            "federation": self._federation.get_stats(),
            "decay": get_decay_system().get_stats(),
        }

        if include_projects:
            result["projects"] = [
                p.to_dict() for p in self._registry.projects.values()
            ]

        return result

    @trace_operation("federation_subscribe")
    async def _handle_federation_subscribe(self, args: dict) -> dict:
        """Subscribe to another project's patterns."""
        from ..evaluation.immune_system import (
            get_registry_manager,
            PatternFederation,
        )

        project_id = args["project_id"]

        # Get or create registry
        if not hasattr(self, '_registry'):
            self._registry = get_registry_manager()

        # Get or create federation
        if not hasattr(self, '_federation'):
            self._federation = PatternFederation(
                graphiti_client=None,
                local_group_id="project_task_orchestrator",
            )

        # Check if project exists
        project = await self._registry.get_project(project_id)
        if not project:
            return {
                "success": False,
                "error": f"Project '{project_id}' not found in registry",
                "available_projects": list(self._registry.projects.keys()),
            }

        # Subscribe via federation
        result = await self._federation.subscribe_to_project(project.group_id)

        # Update local project subscriptions
        local_project = await self._registry.get_project("task-orchestrator")
        if local_project and project_id not in local_project.subscriptions:
            local_project.subscriptions.append(project_id)

        return {
            "success": True,
            "subscribed_to": project_id,
            "group_id": project.group_id,
            "total_subscriptions": len(self._federation.subscriptions),
        }

    @trace_operation("federation_search")
    async def _handle_federation_search(self, args: dict) -> dict:
        """Search patterns across federated projects."""
        from ..evaluation.immune_system import PatternFederation

        query = args["query"]
        limit = args.get("limit", 10)

        # Get or create federation
        if not hasattr(self, '_federation'):
            self._federation = PatternFederation(
                graphiti_client=None,
                local_group_id="project_task_orchestrator",
            )

        try:
            results = await self._federation.search_global_patterns(query, limit)

            return {
                "success": True,
                "query": query,
                "results_count": len(results),
                "results": [
                    {
                        "pattern_id": r.pattern.id,
                        "operation": r.pattern.operation,
                        "failure_type": r.pattern.failure_type,
                        "relevance_score": r.relevance_score,
                        "source_project": r.source_project,
                        "match_reason": r.match_reason,
                    }
                    for r in results
                ],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    @trace_operation("federation_decay")
    async def _handle_federation_decay(self, args: dict) -> dict:
        """Evaluate pattern decay and staleness."""
        from ..evaluation.immune_system import (
            get_decay_system,
            get_immune_system,
        )

        action = args.get("action", "status")
        decay = get_decay_system()

        if action == "status":
            return {
                "success": True,
                "action": "status",
                "stats": decay.get_stats(),
            }

        elif action == "evaluate":
            # Get patterns from immune system
            immune = get_immune_system()
            store_stats = immune._failure_store.get_stats()

            # Get all patterns for evaluation
            patterns = immune._failure_store.get_all_patterns()
            pattern_dicts = [
                {
                    "id": p.id,
                    "decay_metadata": p.context.get("decay_metadata") if p.context else None,
                }
                for p in patterns
            ]

            # Run batch evaluation
            eval_result = decay.batch_evaluate(pattern_dicts)

            return {
                "success": True,
                "action": "evaluate",
                "total_patterns": store_stats["total_patterns"],
                "evaluation": eval_result,
            }

        elif action == "prune_candidates":
            immune = get_immune_system()
            patterns = immune._failure_store.get_all_patterns()

            prune_candidates = []
            for p in patterns:
                metadata = p.context.get("decay_metadata") if p.context else None
                if decay.should_prune(p.id, metadata):
                    prune_candidates.append({
                        "id": p.id,
                        "operation": p.operation,
                        "failure_type": p.failure_type,
                        "current_score": decay.get_current_relevance(p.id, metadata),
                    })

            return {
                "success": True,
                "action": "prune_candidates",
                "count": len(prune_candidates),
                "candidates": prune_candidates,
            }

        return {
            "success": False,
            "error": f"Unknown action: {action}",
        }

    @trace_operation("sync_status")
    async def _handle_sync_status(self, args: dict) -> dict:
        """Get live sync health status for federated projects."""
        from ..evaluation.immune_system.live_sync import SyncHealthMonitor

        # Get or create sync monitor singleton
        if not hasattr(self, '_sync_monitor'):
            self._sync_monitor = SyncHealthMonitor()

        project_id = args.get("project_id")

        if project_id:
            status = self._sync_monitor.get_project_status(project_id)
            if status:
                return {
                    "success": True,
                    "project": status,
                }
            else:
                return {
                    "success": False,
                    "error": f"Project {project_id} not found in sync monitor",
                }

        return {
            "success": True,
            "dashboard": self._sync_monitor.get_dashboard_metrics(),
        }

    @trace_operation("sync_trigger")
    async def _handle_sync_trigger(self, args: dict) -> dict:
        """Trigger a manual sync cycle for federated patterns."""
        from ..evaluation.immune_system.live_sync import SyncEngine
        from ..evaluation.immune_system import get_registry_manager

        direction = args.get("direction", "both")
        project_id = args.get("project_id")

        # Get or create sync engine singleton
        if not hasattr(self, '_sync_engine'):
            self._sync_engine = SyncEngine(
                project_id="task-orchestrator",
                store=None,  # No store configured yet
                transport=None,  # No transport configured yet
            )
            # Register known projects
            registry = get_registry_manager()
            for pid, proj in registry.projects.items():
                if pid != "task-orchestrator":
                    self._sync_engine.register_peer(
                        pid,
                        is_subscriber=True,
                        is_subscription=True
                    )

        result = {
            "success": True,
            "direction": direction,
            "timestamp": __import__('time').time(),
        }

        if direction in ("pull", "both"):
            if project_id:
                if project_id in self._sync_engine._sync_states:
                    result["pull"] = {project_id: "Sync not configured (no transport)"}
                else:
                    result["pull"] = {project_id: "Project not registered"}
            else:
                result["pull"] = self._sync_engine.trigger_pull_sync()

        if direction in ("push", "both"):
            if project_id:
                if project_id in self._sync_engine._sync_states:
                    result["push"] = {project_id: "Sync not configured (no transport)"}
                else:
                    result["push"] = {project_id: "Project not registered"}
            else:
                result["push"] = self._sync_engine.trigger_push_sync()

        return result

    @trace_operation("sync_alerts")
    async def _handle_sync_alerts(self, args: dict) -> dict:
        """Get sync-related alerts for federation health issues."""
        from ..evaluation.immune_system.live_sync import SyncHealthMonitor, SyncStatus

        # Get or create sync monitor singleton
        if not hasattr(self, '_sync_monitor'):
            self._sync_monitor = SyncHealthMonitor()

        severity_filter = args.get("severity")

        alerts = self._sync_monitor.check_health_and_alert()

        # Filter by severity if specified
        if severity_filter:
            try:
                target_status = SyncStatus(severity_filter)
                alerts = [a for a in alerts if a.severity == target_status]
            except ValueError:
                pass  # Invalid severity, return all

        return {
            "success": True,
            "alert_count": len(alerts),
            "alerts": [
                {
                    "project_id": a.project_id,
                    "severity": a.severity.value,
                    "message": a.message,
                    "timestamp": a.timestamp,
                }
                for a in alerts
            ],
        }

    @trace_operation("spawn_agent")
    async def _handle_spawn_agent(self, args: dict) -> dict:
        """Spawn a Gemini agent to execute a code task."""
        import time
        from ..evaluation import (
            Trial, GraderPipeline, NonEmptyGrader, LengthGrader,
            score_trial, get_exporter, get_immune_system
        )

        # Check circuit breaker
        can_proceed, retry_after = self._gemini_breaker.is_available()
        if not can_proceed:
            return {
                "error": f"Gemini service circuit breaker is open. Retry after {retry_after:.1f}s",
                "circuit_breaker_open": True,
                "retry_after_seconds": retry_after,
            }

        # Check budget
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.llm:
            return {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}

        model = args.get("model", "gemini-3-flash-preview")
        original_prompt = args["prompt"]
        system_prompt = args.get("system_prompt", "You are an expert code assistant. Provide clear, working code solutions.")
        max_tokens = args.get("max_tokens", 8192)

        # Immune System: Pre-spawn check
        immune = get_immune_system()
        immune_response = await immune.pre_spawn_check(original_prompt, "spawn_agent")

        # Check if immune system blocked the request
        if not immune_response.should_proceed:
            return {
                "success": False,
                "error": "Request blocked by Immune System due to high failure risk",
                "immune_blocked": True,
                "risk_score": immune_response.risk_score,
                "warnings": immune_response.warnings,
            }

        # Use processed prompt (may have guardrails added)
        prompt = immune_response.processed_prompt

        # Create Trial for evaluation
        trial = Trial(
            operation="spawn_agent",
            input_prompt=original_prompt,
            model=model,
            circuit_breaker_state=self._gemini_breaker._state.value,
        )

        start_time = time.time()

        try:
            response = await self.coordinator.llm.generate(
                prompt,  # Use processed prompt with guardrails
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )

            # Record timing and output
            trial.latency_ms = (time.time() - start_time) * 1000
            trial.output = response.content
            trial.cost_usd = response.usage.get("estimated_cost_usd", 0) if response.usage else 0

            # Run evaluation pipeline (non-blocking)
            pipeline = GraderPipeline([
                NonEmptyGrader(),
                LengthGrader(min_length=10, max_length=100000),
            ])
            grader_results = await pipeline.run(response.content, {"prompt": original_prompt})

            for result in grader_results:
                trial.add_grader_result(result)

            # Push scores to Langfuse (async, non-blocking)
            try:
                await score_trial(trial)
            except Exception:
                pass  # Don't fail the request if scoring fails

            # Add to training data export buffer
            try:
                exporter = get_exporter()
                exporter.add_trial(trial)
            except Exception:
                pass  # Don't fail the request if export buffering fails

            # Immune System: Record failure if evaluation failed
            if not trial.pass_fail:
                try:
                    await immune.record_failure(
                        operation="spawn_agent",
                        prompt=original_prompt,
                        output=response.content,
                        grader_results=[r.to_dict() for r in trial.grader_results],
                        context={"model": model, "cost_usd": trial.cost_usd},
                    )
                except Exception:
                    pass  # Don't fail the request if immune recording fails

            # Record success
            self._gemini_breaker.record_success()

            return {
                "success": True,
                "response": response.content,
                "model": response.model,
                "usage": response.usage,
                "evaluation": {
                    "passed": trial.pass_fail,
                    "scores": [r.to_dict() for r in trial.grader_results],
                },
                "immune": {
                    "risk_score": immune_response.risk_score,
                    "guardrails_applied": immune_response.guardrails_applied,
                    "prompt_modified": immune_response.original_prompt != immune_response.processed_prompt,
                },
            }
        except Exception as e:
            self._gemini_breaker.record_failure(e)
            return {"success": False, "error": str(e)}

    @trace_operation("spawn_parallel_agents")
    async def _handle_spawn_parallel_agents(self, args: dict) -> dict:
        """Spawn multiple Gemini agents in parallel with evaluation."""
        import time
        from ..evaluation import (
            Trial, GraderPipeline, NonEmptyGrader, LengthGrader,
            score_trial, get_exporter, get_immune_system
        )

        # Check circuit breaker
        can_proceed, retry_after = self._gemini_breaker.is_available()
        if not can_proceed:
            return {
                "error": f"Gemini service circuit breaker is open. Retry after {retry_after:.1f}s",
                "circuit_breaker_open": True,
                "retry_after_seconds": retry_after,
            }

        # Check budget
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.llm:
            return {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}

        prompts = args["prompts"]
        model = args.get("model", "gemini-3-flash-preview")
        system_prompt = args.get("system_prompt", "You are an expert code assistant. Provide clear, working code solutions.")
        max_tokens = args.get("max_tokens", 8192)

        # Get immune system instance for all agents
        immune = get_immune_system()

        async def run_single_agent(original_prompt: str, agent_id: int) -> dict:
            # Immune System: Pre-spawn check
            immune_response = await immune.pre_spawn_check(original_prompt, "spawn_parallel_agent")

            # Check if immune system blocked this agent
            if not immune_response.should_proceed:
                return {
                    "agent_id": agent_id,
                    "success": False,
                    "immune_blocked": True,
                    "risk_score": immune_response.risk_score,
                    "warnings": immune_response.warnings,
                    "evaluation": {"passed": False, "scores": []},
                }

            # Use processed prompt (may have guardrails added)
            prompt = immune_response.processed_prompt

            # Create Trial for this agent
            trial = Trial(
                operation="spawn_parallel_agent",
                input_prompt=original_prompt,
                model=model,
                circuit_breaker_state=self._gemini_breaker._state.value,
            )
            trial.metadata = {"agent_id": agent_id, "mode": "parallel"}

            start_time = time.time()

            try:
                response = await self.coordinator.llm.generate(
                    prompt,  # Use processed prompt with guardrails
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )

                # Record timing and output
                trial.latency_ms = (time.time() - start_time) * 1000
                trial.output = response.content
                trial.cost_usd = response.usage.get("estimated_cost_usd", 0) if response.usage else 0

                # Run evaluation pipeline
                pipeline = GraderPipeline([
                    NonEmptyGrader(),
                    LengthGrader(min_length=10, max_length=100000),
                ])
                grader_results = await pipeline.run(response.content, {"prompt": original_prompt})

                for result in grader_results:
                    trial.add_grader_result(result)

                # Push scores to Langfuse (non-blocking)
                try:
                    await score_trial(trial)
                except Exception:
                    pass

                # Add to training data export buffer
                try:
                    exporter = get_exporter()
                    exporter.add_trial(trial)
                except Exception:
                    pass

                # Immune System: Record failure if evaluation failed
                if not trial.pass_fail:
                    try:
                        await immune.record_failure(
                            operation="spawn_parallel_agent",
                            prompt=original_prompt,
                            output=response.content,
                            grader_results=[r.to_dict() for r in trial.grader_results],
                            context={"model": model, "agent_id": agent_id},
                        )
                    except Exception:
                        pass

                return {
                    "agent_id": agent_id,
                    "success": True,
                    "response": response.content,
                    "model": response.model,
                    "usage": response.usage,
                    "evaluation": {
                        "passed": trial.pass_fail,
                        "scores": [r.to_dict() for r in trial.grader_results],
                    },
                    "immune": {
                        "risk_score": immune_response.risk_score,
                        "guardrails_applied": immune_response.guardrails_applied,
                        "prompt_modified": immune_response.original_prompt != immune_response.processed_prompt,
                    },
                }
            except Exception as e:
                trial.latency_ms = (time.time() - start_time) * 1000
                return {
                    "agent_id": agent_id,
                    "success": False,
                    "error": str(e),
                    "evaluation": {"passed": False, "scores": []},
                }

        # Run all agents in parallel
        tasks = [run_single_agent(prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

        # Count successes/failures
        successes = sum(1 for r in results if r.get("success"))
        failures = len(results) - successes
        eval_passed = sum(1 for r in results if r.get("evaluation", {}).get("passed", False))

        if successes > 0:
            self._gemini_breaker.record_success()
        if failures > 0:
            self._gemini_breaker.record_failure(Exception(f"{failures} agents failed"))

        # Track semantic failures if evaluations failed
        if eval_passed < successes:
            self._gemini_breaker.record_semantic_failure("parallel_eval_failed")

        return {
            "success": failures == 0,
            "total_agents": len(prompts),
            "succeeded": successes,
            "failed": failures,
            "evaluations_passed": eval_passed,
            "results": results,
        }


    # =========================================================================
    # Archetype Agent Tools (Yoinked from Anti-gravity)
    # =========================================================================

    @trace_operation("spawn_archetype_agent")
    async def _handle_spawn_archetype_agent(self, args: dict) -> dict:
        """Spawn an agent with a specific archetype role."""
        import time
        from ..evaluation import (
            Trial, GraderPipeline, NonEmptyGrader, LengthGrader,
            score_trial, get_exporter, get_immune_system
        )

        # Check circuit breaker
        can_proceed, retry_after = self._gemini_breaker.is_available()
        if not can_proceed:
            return {
                "error": f"Gemini service circuit breaker is open. Retry after {retry_after:.1f}s",
                "circuit_breaker_open": True,
                "retry_after_seconds": retry_after,
            }

        # Check budget
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}

        if not self.coordinator.llm:
            return {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}

        # Parse archetype
        archetype_name = args.get("archetype", "builder").lower()
        archetype = self._archetype_registry.get_archetype_by_name(archetype_name)
        if not archetype:
            return {
                "error": f"Unknown archetype: {archetype_name}",
                "valid_archetypes": ["architect", "builder", "qc", "researcher"],
            }

        model = args.get("model", "gemini-3-flash-preview")
        original_prompt = args["prompt"]
        max_tokens = args.get("max_tokens", 8192)
        inject_audit = args.get("inject_audit", True)

        # Get archetype-specific configuration
        archetype_config = self._archetype_registry.get_archetype_config(archetype)
        system_prompt = archetype_config.system_prompt
        temperature = archetype_config.temperature

        # Inject audit history if requested
        if inject_audit:
            if not self._audit_workflow:
                self._audit_workflow = AuditWorkflow()
            system_prompt = self._audit_workflow.inject_to_prompt(system_prompt)

        # Immune System: Pre-spawn check
        immune = get_immune_system()
        immune_response = await immune.pre_spawn_check(original_prompt, "spawn_archetype_agent")

        if not immune_response.should_proceed:
            return {
                "success": False,
                "error": "Request blocked by Immune System due to high failure risk",
                "immune_blocked": True,
                "risk_score": immune_response.risk_score,
                "warnings": immune_response.warnings,
            }

        prompt = immune_response.processed_prompt

        # Publish agent start event to inbox
        start_event = AgentEvent(
            event_type=EventType.AGENT_START,
            agent_name=f"{archetype_name}_agent",
            data={
                "archetype": archetype_name,
                "model": model,
                "prompt_preview": prompt[:200],
            },
            source="spawn_archetype_agent",
        )
        await self._universal_inbox.publish(start_event)

        # Create Trial for evaluation
        trial = Trial(
            operation="spawn_archetype_agent",
            input_prompt=original_prompt,
            model=model,
            circuit_breaker_state=self._gemini_breaker._state.value,
        )
        trial.metadata = {"archetype": archetype_name}

        start_time = time.time()

        try:
            response = await self.coordinator.llm.generate(
                prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Record timing and output
            trial.latency_ms = (time.time() - start_time) * 1000
            trial.output = response.content
            trial.cost_usd = response.usage.get("estimated_cost_usd", 0) if response.usage else 0

            # Run evaluation pipeline
            pipeline = GraderPipeline([
                NonEmptyGrader(),
                LengthGrader(min_length=10, max_length=100000),
            ])
            grader_results = await pipeline.run(response.content, {"prompt": original_prompt})

            for result in grader_results:
                trial.add_grader_result(result)

            # Push scores to Langfuse
            try:
                await score_trial(trial)
            except Exception:
                pass

            # Add to training data export
            try:
                exporter = get_exporter()
                exporter.add_trial(trial)
            except Exception:
                pass

            # Publish agent end event
            end_event = AgentEvent(
                event_type=EventType.AGENT_END,
                agent_name=f"{archetype_name}_agent",
                data={
                    "archetype": archetype_name,
                    "success": True,
                    "latency_ms": trial.latency_ms,
                },
                source="spawn_archetype_agent",
            )
            await self._universal_inbox.publish(end_event)

            # Record success
            self._gemini_breaker.record_success()

            return {
                "success": True,
                "archetype": archetype_name,
                "response": response.content,
                "model": response.model,
                "usage": response.usage,
                "archetype_config": {
                    "temperature": temperature,
                    "category": archetype_config.category,
                    "tool_count": len(archetype_config.tools),
                    "readonly": self._archetype_registry.is_readonly(archetype),
                },
                "evaluation": {
                    "passed": trial.pass_fail,
                    "scores": [r.to_dict() for r in trial.grader_results],
                },
                "immune": {
                    "risk_score": immune_response.risk_score,
                    "guardrails_applied": immune_response.guardrails_applied,
                },
            }
        except Exception as e:
            self._gemini_breaker.record_failure(e)

            # Publish error event
            error_event = AgentEvent(
                event_type=EventType.ERROR,
                agent_name=f"{archetype_name}_agent",
                data={"error": str(e), "archetype": archetype_name},
                source="spawn_archetype_agent",
            )
            await self._universal_inbox.publish(error_event)

            return {"success": False, "error": str(e)}

    @trace_operation("inbox_status")
    async def _handle_inbox_status(self, args: dict) -> dict:
        """Get universal inbox status including pending approvals."""
        risk_level_str = args.get("risk_level")
        agent_name = args.get("agent_name")
        include_history = args.get("include_history", False)
        history_limit = args.get("history_limit", 20)

        # Parse risk level if provided
        risk_level = None
        if risk_level_str:
            try:
                risk_level = ActionRiskLevel(risk_level_str)
            except ValueError:
                pass

        # Get pending approvals
        pending = self._universal_inbox.get_pending_approvals(
            risk_level=risk_level,
            agent_name=agent_name,
        )

        result = {
            "success": True,
            "pending_approvals": [a.to_dict() for a in pending],
            "pending_count": len(pending),
            "by_risk_level": {
                "LOW": len([a for a in pending if a.risk_level == ActionRiskLevel.LOW]),
                "MEDIUM": len([a for a in pending if a.risk_level == ActionRiskLevel.MEDIUM]),
                "HIGH": len([a for a in pending if a.risk_level == ActionRiskLevel.HIGH]),
                "CRITICAL": len([a for a in pending if a.risk_level == ActionRiskLevel.CRITICAL]),
            },
        }

        # Include event history if requested
        if include_history:
            result["event_history"] = self._universal_inbox.get_event_history(
                agent_name=agent_name,
                limit=history_limit,
            )

        return result

    @trace_operation("approve_action")
    async def _handle_approve_action(self, args: dict) -> dict:
        """Approve or reject a pending action."""
        action_id = args["action_id"]
        should_approve = args.get("approve", True)
        reason = args.get("reason", "")
        approved_by = args.get("approved_by", "system")

        try:
            if should_approve:
                action = await self._universal_inbox.approve(
                    action_id=action_id,
                    approved_by=approved_by,
                )
                return {
                    "success": True,
                    "action_id": action_id,
                    "status": "approved",
                    "approved_by": approved_by,
                    "execution_result": action.execution_result,
                }
            else:
                if not reason:
                    return {
                        "success": False,
                        "error": "Reason is required when rejecting an action",
                    }
                action = await self._universal_inbox.reject(
                    action_id=action_id,
                    reason=reason,
                    rejected_by=approved_by,
                )
                return {
                    "success": True,
                    "action_id": action_id,
                    "status": "rejected",
                    "rejected_by": approved_by,
                    "reason": reason,
                }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    @trace_operation("audit_status")
    async def _handle_audit_status(self, args: dict) -> dict:
        """Get audit workflow status."""
        project_root = args.get("project_root")
        query_topic = args.get("query_topic")
        query_error_type = args.get("query_error_type")

        # Initialize or re-initialize audit workflow if project root changed
        if not self._audit_workflow or (project_root and str(self._audit_workflow.project_root) != project_root):
            self._audit_workflow = AuditWorkflow(project_root=project_root)

        result = {
            "success": True,
            "summary": self._audit_workflow.get_summary(),
            "audit_file": str(self._audit_workflow.audit_file),
            "file_exists": self._audit_workflow.audit_file.exists(),
        }

        # Add query results if requested
        if query_topic:
            result["topic_matches"] = self._audit_workflow.query_decisions(query_topic)

        if query_error_type:
            result["error_matches"] = self._audit_workflow.query_errors(query_error_type)

        return result

    @trace_operation("audit_append")
    async def _handle_audit_append(self, args: dict) -> dict:
        """Append a new entry to the audit log."""
        entry_type = args["entry_type"]
        content = args["content"]
        title = args.get("title")
        project_root = args.get("project_root")
        metadata = args.get("metadata", {})

        # Initialize audit workflow if needed
        if not self._audit_workflow or (project_root and str(self._audit_workflow.project_root) != project_root):
            self._audit_workflow = AuditWorkflow(project_root=project_root)

        try:
            self._audit_workflow.append_entry(
                entry_type=entry_type,
                content=content,
                title=title,
                metadata=metadata,
            )
            return {
                "success": True,
                "entry_type": entry_type,
                "title": title or f"Entry {len(self._audit_workflow.audit_data.get(entry_type + 's', []))}",
                "audit_file": str(self._audit_workflow.audit_file),
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    @trace_operation("archetype_info")
    async def _handle_archetype_info(self, args: dict) -> dict:
        """Get information about agent archetypes."""
        archetype_name = args.get("archetype")

        if archetype_name:
            archetype = self._archetype_registry.get_archetype_by_name(archetype_name)
            if not archetype:
                return {
                    "success": False,
                    "error": f"Unknown archetype: {archetype_name}",
                    "valid_archetypes": ["architect", "builder", "qc", "researcher"],
                }

            config = self._archetype_registry.get_archetype_config(archetype)
            return {
                "success": True,
                "archetype": archetype_name,
                "description": config.description,
                "category": config.category,
                "temperature": config.temperature,
                "tool_count": len(config.tools),
                "tools": config.tools,
                "readonly": self._archetype_registry.is_readonly(archetype),
                "system_prompt_preview": config.system_prompt[:500] + "...",
            }

        # Return all archetypes
        return {
            "success": True,
            "archetypes": self._archetype_registry.get_summary(),
        }


async def run_mcp_server():
    """Run as stdio MCP server."""
    server = TaskOrchestratorMCP()

    # Read from stdin, write to stdout
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break

            request = json.loads(line.strip())
            method = request.get("method", "")
            req_id = request.get("id")

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "task-orchestrator",
                            "version": "1.0.0",
                        },
                    },
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": server.get_tools()},
                }
            elif method == "tools/call":
                params = request.get("params", {})
                result = await server.handle_tool_call(
                    params.get("name", ""),
                    params.get("arguments", {}),
                )
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                }
            elif method.startswith("notifications/"):
                # Notifications don't get responses
                continue
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }

            print(json.dumps(response), flush=True)

        except json.JSONDecodeError:
            continue
        except Exception as e:
            if req_id:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(e)},
                }
                print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    asyncio.run(run_mcp_server())
