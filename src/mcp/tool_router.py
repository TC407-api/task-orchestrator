"""Dynamic tool routing for MCP server.

Manages tool categories and lazy loading to reduce context window usage.
"""
from enum import Enum
from typing import Dict, List, Any, Optional, Set


class ToolCategory(str, Enum):
    """Categories of MCP tools for lazy loading."""
    CORE = "core"           # Always available: tasks_list, tasks_add, spawn_agent, healing_status, request_tool
    TASK = "task"           # Task management: tasks_sync_email, tasks_schedule, tasks_complete, etc.
    AGENT = "agent"         # Agent tools: spawn_parallel_agents, spawn_archetype_agent, archetype_info
    IMMUNE = "immune"       # Immune system: immune_status, immune_check, immune_failures, immune_dashboard
    FEDERATION = "federation"  # Federation: federation_status, federation_subscribe, federation_search
    SYNC = "sync"           # Sync tools: sync_status, sync_trigger, sync_alerts
    WORKFLOW = "workflow"   # Workflow tools: trigger_workflow, list_workflows, validate_code
    COST = "cost"           # Cost tools: cost_summary, cost_set_budget
    LEARNING = "learning"   # Learning tools: learning_workflow


# Core tools always available (5 tools + request_tool = 6 tools max)
CORE_TOOLS = [
    "tasks_list",
    "tasks_add",
    "spawn_agent",
    "healing_status",
    "request_tool",
]

# Tool category mappings
TOOL_CATEGORIES: Dict[ToolCategory, List[str]] = {
    ToolCategory.CORE: CORE_TOOLS,
    ToolCategory.TASK: [
        "tasks_sync_email",
        "tasks_schedule",
        "tasks_complete",
        "tasks_analyze",
        "tasks_briefing",
    ],
    ToolCategory.AGENT: [
        "spawn_parallel_agents",
        "spawn_archetype_agent",
        "archetype_info",
        "inbox_status",
        "approve_action",
        "audit_status",
        "audit_append",
    ],
    ToolCategory.IMMUNE: [
        "immune_status",
        "immune_check",
        "immune_failures",
        "immune_dashboard",
        "immune_sync",
        "alert_list",
        "alert_clear",
        "predict_risk",
    ],
    ToolCategory.FEDERATION: [
        "federation_status",
        "federation_subscribe",
        "federation_search",
        "federation_decay",
    ],
    ToolCategory.SYNC: [
        "sync_status",
        "sync_trigger",
        "sync_alerts",
    ],
    ToolCategory.WORKFLOW: [
        "run_with_error_capture",
        "validate_code",
        "trigger_workflow",
        "list_workflows",
        "schedule_task",
        "list_scheduled_tasks",
        "cancel_scheduled_task",
    ],
    ToolCategory.COST: [
        "cost_summary",
        "cost_set_budget",
    ],
    ToolCategory.LEARNING: [
        "learning_workflow",
    ],
}


class ToolRouter:
    """
    Routes tool requests and manages dynamic tool loading.

    In dynamic mode, only CORE tools are exposed. Other tools must be
    explicitly requested via request_tool(), which loads an entire category.
    """

    def __init__(self, all_tools: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the router.

        Args:
            all_tools: Complete list of all tool definitions
        """
        self._all_tools = all_tools or []
        self._loaded_categories: Set[ToolCategory] = {ToolCategory.CORE}
        self._dynamic_mode = False

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get currently available tools based on mode."""
        if not self._dynamic_mode:
            # Full mode: return all tools
            return self._all_tools

        # Dynamic mode: return only tools from loaded categories
        allowed_tool_names = set()
        for category in self._loaded_categories:
            allowed_tool_names.update(TOOL_CATEGORIES.get(category, []))

        return [
            tool for tool in self._all_tools
            if tool.get("name") in allowed_tool_names
        ]

    def request_tool(self, category: str) -> List[Dict[str, Any]]:
        """Load a tool category and return the tools."""
        # Convert string to enum
        try:
            cat_enum = ToolCategory(category.lower())
        except ValueError:
            return []

        # Add category to loaded set
        self._loaded_categories.add(cat_enum)

        # Get tool names for this category
        tool_names = TOOL_CATEGORIES.get(cat_enum, [])

        # Return tool definitions from this category
        return [
            tool for tool in self._all_tools
            if tool.get("name") in tool_names
        ]

    def should_switch_to_dynamic(self, remaining_pct: float) -> bool:
        """Determine if we should switch to dynamic mode."""
        # Switch to dynamic when remaining context is 10% or less
        return remaining_pct <= 0.10

    def set_dynamic_mode(self, enabled: bool) -> None:
        """Enable or disable dynamic mode."""
        self._dynamic_mode = enabled

    def is_dynamic_mode(self) -> bool:
        """Check if dynamic mode is enabled."""
        return self._dynamic_mode

    def get_loaded_categories(self) -> Set[ToolCategory]:
        """Get set of currently loaded categories."""
        return self._loaded_categories.copy()

    def get_category_for_tool(self, tool_name: str) -> Optional[ToolCategory]:
        """Find which category a tool belongs to."""
        for category, tools in TOOL_CATEGORIES.items():
            if tool_name in tools:
                return category
        return None
