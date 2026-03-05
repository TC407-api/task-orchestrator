"""MCP Server for Task Orchestrator integration with Claude Code."""
import asyncio
import json
import logging
import sys

logger = logging.getLogger(__name__)
from typing import Optional

from ..core.cost_tracker import get_cost_tracker
from ..agents.coordinator import CoordinatorAgent
from ..agents.archetype_registry import get_archetype_registry
from ..agents.inbox import UniversalInbox
from ..agents.audit_workflow import AuditWorkflow
from ..agents.shadow_validator import ShadowValidator
from ..agents.workflows import WorkflowRegistry
from ..agents.background_tasks import BackgroundTaskScheduler
from ..observability import trace_operation
from ..self_healing import CircuitBreaker, get_healing_status
from .tool_router import ToolRouter
from .context_tracker import ContextTracker
from .content_tools import CONTENT_TOOLS, ContentToolsHandler
from .research_tools import RESEARCH_TOOLS, ResearchToolHandler
from .learning_tools import LEARNING_TOOLS, LearningToolHandler
from .tool_schemas import get_all_tools
from .task_handlers import TaskHandlers
from .cost_handlers import CostHandlers
from .immune_handlers import ImmuneHandlers
from .federation_handlers import FederationHandlers
from .workflow_handlers import WorkflowHandlers
from .archetype_handlers import ArchetypeHandlers
from .agent_runner import AgentRunnerHandlers


class TaskOrchestratorMCP(
    TaskHandlers,
    CostHandlers,
    ImmuneHandlers,
    FederationHandlers,
    WorkflowHandlers,
    ArchetypeHandlers,
    AgentRunnerHandlers,
):
    """
    MCP Server exposing task orchestrator functionality.

    Thin router - all handler logic lives in focused handler modules:
      - task_handlers.py: tasks_list, tasks_add, tasks_sync_email, tasks_schedule,
                          tasks_complete, tasks_analyze, tasks_briefing
      - cost_handlers.py: cost_summary, cost_set_budget, healing_status
      - immune_handlers.py: immune_*, alert_*, predict_risk
      - federation_handlers.py: federation_*, sync_*
      - workflow_handlers.py: run_with_error_capture, validate_code,
                             trigger_workflow, list_workflows, schedule_task,
                             list_scheduled_tasks, cancel_scheduled_task
      - archetype_handlers.py: spawn_archetype_agent, inbox_status,
                               approve_action, audit_*, archetype_info
      - agent_runner.py: spawn_agent, spawn_parallel_agents
    """

    def __init__(self):
        self.coordinator: Optional[CoordinatorAgent] = None
        self.cost_tracker = get_cost_tracker()
        self._initialized = False

        # Circuit breakers for external services
        self._gmail_breaker = CircuitBreaker.get("gmail_service")
        self._calendar_breaker = CircuitBreaker.get("calendar_service")
        self._gemini_breaker = CircuitBreaker.get("gemini_service")

        # Archetype-based components
        self._archetype_registry = get_archetype_registry()
        self._universal_inbox = UniversalInbox()
        self._audit_workflow: Optional[AuditWorkflow] = None

        # Batch 2 components
        self._terminal_listener = None
        self._shadow_validator = ShadowValidator()
        self._workflow_registry = WorkflowRegistry()
        self._background_scheduler: Optional[BackgroundTaskScheduler] = None

        # Federation components (Phase 9)
        self._federation = None
        self._graphiti_client = None
        self._registry = None

        # Lazy-initialized singletons (immune system)
        self._alert_manager = None
        self._predictor = None
        self._sync_monitor = None
        self._sync_engine = None

        # Dynamic tool loading (Phase 10)
        self._context_tracker = ContextTracker()
        self._tool_router: Optional[ToolRouter] = None

        # Content / Research / Learning automation handlers
        self._content_tools_handler: Optional[ContentToolsHandler] = None
        self._research_tools_handler: Optional[ResearchToolHandler] = None
        self._learning_tools_handler: Optional[LearningToolHandler] = None

    async def _get_federation(self):
        """Get or create the PatternFederation with Graphiti client."""
        if self._federation is not None:
            return self._federation
        from ..evaluation.immune_system import PatternFederation, get_registry_manager
        from ..evaluation.immune_system.graphiti_client import create_graphiti_client
        if self._registry is None:
            self._registry = get_registry_manager()
        if self._graphiti_client is None:
            self._graphiti_client = create_graphiti_client()
        self._federation = PatternFederation(
            graphiti_client=self._graphiti_client,
            local_group_id="project_task_orchestrator",
        )
        return self._federation

    async def initialize(self):
        """Initialize the coordinator with agents."""
        if self._initialized:
            return
        from ..llm import GeminiProvider, ModelRouter
        from ..core.cost_tracker import Provider
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            raise RuntimeError(f"Cannot initialize: {msg}")
        try:
            provider = GeminiProvider()
            router = ModelRouter({"google": provider})
            self.coordinator = CoordinatorAgent(llm_router=router)
            self._initialized = True
        except ValueError:
            self.coordinator = CoordinatorAgent()
            self._initialized = True

    async def _get_content_tools_handler(self) -> ContentToolsHandler:
        """Get or create the ContentToolsHandler."""
        if self._content_tools_handler is None:
            if self._background_scheduler is None:
                self._background_scheduler = BackgroundTaskScheduler(inbox=self._universal_inbox)
                await self._background_scheduler.start()
            self._content_tools_handler = ContentToolsHandler(
                scheduler=self._background_scheduler,
                inbox=self._universal_inbox,
            )
        return self._content_tools_handler

    async def _get_research_tools_handler(self) -> ResearchToolHandler:
        """Get or create the ResearchToolHandler."""
        if self._research_tools_handler is None:
            if self._background_scheduler is None:
                self._background_scheduler = BackgroundTaskScheduler(inbox=self._universal_inbox)
                await self._background_scheduler.start()
            self._research_tools_handler = ResearchToolHandler(
                background_scheduler=self._background_scheduler,
            )
        return self._research_tools_handler

    def _get_learning_tools_handler(self) -> LearningToolHandler:
        """Get or create the LearningToolHandler."""
        if self._learning_tools_handler is None:
            self._learning_tools_handler = LearningToolHandler(server=self)
        return self._learning_tools_handler

    def get_tools(self) -> list[dict]:
        """Return MCP tool definitions."""
        return get_all_tools()

    async def handle_tool_call(self, name: str, arguments: dict):
        """Handle an MCP tool call - thin router delegating to handler mixins."""
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
            "sync_status": self._handle_sync_status,
            "sync_trigger": self._handle_sync_trigger,
            "sync_alerts": self._handle_sync_alerts,
            "spawn_archetype_agent": self._handle_spawn_archetype_agent,
            "inbox_status": self._handle_inbox_status,
            "approve_action": self._handle_approve_action,
            "audit_status": self._handle_audit_status,
            "audit_append": self._handle_audit_append,
            "archetype_info": self._handle_archetype_info,
            "run_with_error_capture": self._handle_run_with_error_capture,
            "validate_code": self._handle_validate_code,
            "trigger_workflow": self._handle_trigger_workflow,
            "list_workflows": self._handle_list_workflows,
            "schedule_task": self._handle_schedule_task,
            "list_scheduled_tasks": self._handle_list_scheduled_tasks,
            "cancel_scheduled_task": self._handle_cancel_scheduled_task,
            "request_tool": self._handle_request_tool,
        }

        content_handler_names = {
            "content_generate": "handle_content_generate",
            "content_schedule": "handle_content_schedule",
            "content_publish": "handle_content_publish",
            "content_publish_all": "handle_content_publish_all",
            "content_status": "handle_content_status",
            "content_list_campaigns": "handle_content_list_campaigns",
            "content_cancel": "handle_content_cancel",
        }
        if name in content_handler_names:
            try:
                handler = await self._get_content_tools_handler()
                return await getattr(handler, content_handler_names[name])(arguments)
            except Exception as e:
                logger.error("Content handler failed", exc_info=True)
                return {"error": "Operation failed"}

        research_tool_names = {
            "research_run", "research_add_topic", "research_remove_topic",
            "research_list_topics", "research_get_context", "research_schedule",
            "research_status", "research_search_past",
        }
        if name in research_tool_names:
            try:
                handler = await self._get_research_tools_handler()
                return await handler.handle_tool(name, arguments)
            except Exception as e:
                logger.error("Research handler failed", exc_info=True)
                return {"error": "Operation failed"}

        if name == "learning_workflow":
            try:
                return await self._get_learning_tools_handler().handle_tool(name, arguments)
            except Exception as e:
                logger.error("Learning handler failed", exc_info=True)
                return {"error": "Operation failed"}

        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}
        try:
            return await handler(arguments)
        except Exception as e:
            logger.error("Tool handler failed", exc_info=True)
            return {"error": "Operation failed"}

    @trace_operation("request_tool")
    async def _handle_request_tool(self, args: dict) -> dict:
        """Load a tool category dynamically."""
        category = args.get("category", "")
        reason = args.get("reason", "No reason provided")
        if self._tool_router is None:
            self._tool_router = ToolRouter(all_tools=self.get_tools())
        try:
            loaded_tools = self._tool_router.request_tool(category)
            tool_names = [t.get("name", "unknown") for t in loaded_tools]
            return {
                "success": True,
                "category": category,
                "tools_loaded": tool_names,
                "count": len(tool_names),
                "reason_logged": reason,
                "loaded_categories": list(cat.value for cat in self._tool_router.get_loaded_categories()),
            }
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid category: {category}",
                "available_categories": ["task", "agent", "immune", "federation", "sync", "workflow", "cost"],
            }


async def run_mcp_server():
    """Run as stdio MCP server."""
    server = TaskOrchestratorMCP()

    while True:
        try:
            line = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
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
                        "serverInfo": {"name": "task-orchestrator", "version": "1.0.0"},
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
                    params.get("name", ""), params.get("arguments", {})
                )
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                }
            elif method.startswith("notifications/"):
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
                print(
                    json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(e)}}),
                    flush=True,
                )


if __name__ == "__main__":
    asyncio.run(run_mcp_server())
