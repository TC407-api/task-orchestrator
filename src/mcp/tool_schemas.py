"""MCP tool inputSchema definitions for Task Orchestrator.

Extracted from server.py to keep that file thin.
"""
from .content_tools import CONTENT_TOOLS
from .research_tools import RESEARCH_TOOLS
from .learning_tools import LEARNING_TOOLS


def get_all_tools() -> list[dict]:
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
        # Batch 2: Terminal, Validation, Workflows, Background Tasks
        {
            "name": "run_with_error_capture",
            "description": "Run a command and capture any errors with stack trace analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run"},
                    "working_dir": {"type": "string", "description": "Working directory (optional)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "validate_code",
            "description": "Validate code syntax and style before showing to user.",
            "inputSchema": {
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
            "name": "trigger_workflow",
            "description": "Execute an @Workflow trigger (e.g., @Refactor, @TestGen, @Debug).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "string",
                        "enum": ["refactor", "testgen", "debug", "review", "docs"],
                        "description": "Workflow to trigger",
                    },
                    "prompt": {"type": "string", "description": "User prompt to process"},
                    "target_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target files for the workflow (optional)",
                    },
                },
                "required": ["workflow", "prompt"],
            },
        },
        {
            "name": "list_workflows",
            "description": "List available @Workflow triggers.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "schedule_task",
            "description": "Schedule a background task.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Task name"},
                    "command": {"type": "string", "description": "Command to run"},
                    "schedule_type": {
                        "type": "string",
                        "enum": ["one_time", "recurring", "deferred"],
                        "description": "Schedule type",
                    },
                    "run_at": {"type": "string", "description": "ISO datetime for one_time tasks"},
                    "interval_seconds": {"type": "integer", "description": "Interval for recurring tasks"},
                },
                "required": ["name", "command", "schedule_type"],
            },
        },
        {
            "name": "list_scheduled_tasks",
            "description": "List scheduled background tasks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "pending", "running", "completed"],
                        "description": "Filter by status",
                    },
                },
            },
        },
        {
            "name": "cancel_scheduled_task",
            "description": "Cancel a scheduled background task.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to cancel"},
                },
                "required": ["task_id"],
            },
        },
        # Dynamic tool loading (Phase 10)
        {
            "name": "request_tool",
            "description": "Load additional tool categories dynamically. Use when you need tools beyond the core set. Categories: task, agent, immune, federation, sync, workflow, cost.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["task", "agent", "immune", "federation", "sync", "workflow", "cost"],
                        "description": "Tool category to load",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you need these tools (for audit trail)",
                    },
                },
                "required": ["category"],
            },
        },
    ] + CONTENT_TOOLS + RESEARCH_TOOLS + LEARNING_TOOLS  # Content + Research + Learning automation tools (Phase 11-13)
