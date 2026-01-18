"""Core plugin providing free tier tools for Task Orchestrator.

This plugin is MIT licensed and always available. It includes:
- Agent spawning (spawn_agent, spawn_parallel_agents)
- Immune system (immune_status, immune_check, immune_failures, immune_dashboard)
- Task management (tasks_list, tasks_add, tasks_complete, tasks_analyze)
- Self-healing (healing_status)
- Cost tracking (cost_summary, cost_set_budget)
- Validation (validate_code, run_with_error_capture)
- Human-in-the-loop (inbox_status, approve_action)
"""
from typing import Any, Callable, Dict, List

from .base import PluginInterface, PluginTier


# Tool definitions for the free tier
CORE_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    # Agent tools
    {
        "name": "spawn_agent",
        "description": "Spawn a Gemini agent to execute a code task. Returns the agent's response.",
        "parameters": {
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
        "parameters": {
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
    # Immune system tools
    {
        "name": "immune_status",
        "description": "Get immune system health and statistics including failure patterns and guardrail effectiveness.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "immune_check",
        "description": "Pre-check a prompt for risks without executing it. Returns risk score and suggestions.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The prompt to evaluate"},
                "operation": {
                    "type": "string",
                    "default": "spawn_agent",
                    "description": "Operation type (spawn_agent, spawn_parallel_agent)",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "immune_failures",
        "description": "List recent failure patterns stored in the immune system memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum patterns to return (default: 10)",
                },
            },
        },
    },
    {
        "name": "immune_dashboard",
        "description": "Get a comprehensive dashboard report of the immune system including health metrics, failure trends, and top patterns.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "default": 7,
                    "description": "Number of days for trend analysis (default: 7)",
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "json"],
                    "default": "markdown",
                    "description": "Output format (default: markdown)",
                },
            },
        },
    },
    # Task management tools
    {
        "name": "tasks_list",
        "description": "List tasks sorted by priority. Returns pending, scheduled, and in-progress tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["all", "pending", "scheduled", "in_progress", "completed"],
                    "description": "Filter by status (default: all active)",
                },
                "limit": {"type": "integer", "description": "Max tasks to return (default: 10)"},
            },
        },
    },
    {
        "name": "tasks_add",
        "description": "Create a new task in the system.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Task title"},
                "description": {"type": "string", "description": "Task details"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Priority level (default: medium)",
                },
                "due_date": {"type": "string", "description": "Due date in ISO format (optional)"},
                "estimated_minutes": {"type": "integer", "description": "Estimated time in minutes (default: 30)"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                "auto_schedule": {"type": "boolean", "description": "Auto-schedule on calendar (default: false)"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "tasks_complete",
        "description": "Mark a task as completed.",
        "parameters": {
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
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to analyze"},
            },
            "required": ["task_id"],
        },
    },
    # Self-healing tools
    {
        "name": "healing_status",
        "description": "Get self-healing system status including circuit breakers and retry state.",
        "parameters": {"type": "object", "properties": {}},
    },
    # Cost tracking tools
    {
        "name": "cost_summary",
        "description": "View API cost summary across all providers (Gemini, OpenAI, etc.).",
        "parameters": {
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
        "parameters": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": ["google_gemini", "openai", "graphiti", "google_gmail", "google_calendar"],
                    "description": "Provider to configure",
                },
                "daily_limit": {"type": "number", "description": "Daily budget in USD"},
                "monthly_limit": {"type": "number", "description": "Monthly budget in USD"},
            },
            "required": ["provider"],
        },
    },
    # Validation tools
    {
        "name": "validate_code",
        "description": "Validate code syntax and style before showing to user.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to validate"},
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "json"],
                    "description": "Programming language",
                },
                "run_linter": {"type": "boolean", "description": "Also run linter (default: true)"},
            },
            "required": ["code", "language"],
        },
    },
    {
        "name": "run_with_error_capture",
        "description": "Run a command and capture any errors with stack trace analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run"},
                "working_dir": {"type": "string", "description": "Working directory (optional)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
            },
            "required": ["command"],
        },
    },
    # Human-in-the-loop tools
    {
        "name": "inbox_status",
        "description": "Get universal inbox status including pending approvals and recent events.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_history": {"type": "boolean", "description": "Include recent event history (default: false)"},
                "history_limit": {"type": "integer", "description": "Max history events to return (default: 20)"},
                "agent_name": {"type": "string", "description": "Filter by agent name (optional)"},
                "risk_level": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    "description": "Filter pending approvals by risk level (optional)",
                },
            },
        },
    },
    {
        "name": "approve_action",
        "description": "Approve or reject a pending action in the approval queue.",
        "parameters": {
            "type": "object",
            "properties": {
                "action_id": {"type": "string", "description": "ID of the action to approve/reject"},
                "approve": {"type": "boolean", "description": "True to approve, False to reject"},
                "reason": {"type": "string", "description": "Reason for rejection (required if rejecting)"},
                "approved_by": {"type": "string", "description": "User approving/rejecting (default: system)"},
            },
            "required": ["action_id", "approve"],
        },
    },
    # Dynamic tool loading
    {
        "name": "request_tool",
        "description": "Load additional tool categories dynamically. Use when you need tools beyond the core set. Categories: task, agent, immune, federation, sync, workflow, cost.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["task", "agent", "immune", "federation", "sync", "workflow", "cost"],
                    "description": "Tool category to load",
                },
                "reason": {"type": "string", "description": "Why you need these tools (for audit trail)"},
            },
            "required": ["category"],
        },
    },
]


class CorePlugin(PluginInterface):
    """
    Core plugin providing free tier tools.

    This plugin is always available (MIT license) and provides
    essential functionality for safe AI agent execution.
    """

    name = "core"
    version = "1.0.0"
    tier = PluginTier.FREE
    description = "Essential tools for safe AI agent execution"

    def __init__(self, server_instance=None):
        """
        Initialize the core plugin.

        Args:
            server_instance: Reference to the TaskOrchestratorMCP server
                           for accessing shared state (optional for testing)
        """
        self._server = server_instance

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return core tool definitions."""
        return CORE_TOOL_DEFINITIONS.copy()

    def get_tool_handlers(self) -> Dict[str, Callable]:
        """
        Return handlers for core tools.

        Returns dict mapping tool names to handler functions.
        Handlers are bound to the server instance for state access.
        """
        if self._server is None:
            # Return empty handlers for testing
            return {}

        return {
            "spawn_agent": self._server._handle_spawn_agent,
            "spawn_parallel_agents": self._server._handle_spawn_parallel_agents,
            "immune_status": self._server._handle_immune_status,
            "immune_check": self._server._handle_immune_check,
            "immune_failures": self._server._handle_immune_failures,
            "immune_dashboard": self._server._handle_immune_dashboard,
            "tasks_list": self._server._handle_tasks_list,
            "tasks_add": self._server._handle_tasks_add,
            "tasks_complete": self._server._handle_tasks_complete,
            "tasks_analyze": self._server._handle_tasks_analyze,
            "healing_status": self._server._handle_healing_status,
            "cost_summary": self._server._handle_cost_summary,
            "cost_set_budget": self._server._handle_cost_set_budget,
            "validate_code": self._server._handle_validate_code,
            "run_with_error_capture": self._server._handle_run_with_error_capture,
            "inbox_status": self._server._handle_inbox_status,
            "approve_action": self._server._handle_approve_action,
            "request_tool": self._server._handle_request_tool,
        }

    def on_load(self) -> None:
        """Called when plugin is loaded."""
        pass  # Core plugin has no special initialization

    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass  # Core plugin should never be unloaded
