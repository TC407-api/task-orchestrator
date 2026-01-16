"""
Research MCP Tools - Tool definitions for the Auto Research Scheduler.

Provides tools for managing research topics, running research, and
accessing research context.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Tool definitions for MCP server
RESEARCH_TOOLS = [
    {
        "name": "research_run",
        "description": "Run research on topics immediately. Searches for each topic via Firecrawl, summarizes with AI, and stores in Graphiti.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific topics to research. If empty, uses all registered topics.",
                },
                "project": {
                    "type": "string",
                    "description": "Filter topics by project name (optional).",
                },
                "max_topics": {
                    "type": "integer",
                    "description": "Maximum number of topics to research this run (default: 10).",
                },
            },
        },
    },
    {
        "name": "research_add_topic",
        "description": "Add a topic to the research registry for daily automated research.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Research topic string (e.g., 'AI agents autonomous systems 2026').",
                },
                "project": {
                    "type": "string",
                    "description": "Project name to associate with. Use 'global' for all projects.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "research_remove_topic",
        "description": "Remove a topic from the research registry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to remove.",
                },
                "project": {
                    "type": "string",
                    "description": "Project to remove from. Use 'global' for global topics.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "research_list_topics",
        "description": "List all registered research topics across all projects.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Filter by project name (optional).",
                },
            },
        },
    },
    {
        "name": "research_get_context",
        "description": "Get the latest research context for injection into Claude sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Specific date (YYYY-MM-DD) or 'latest' for most recent (default: latest).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 2000).",
                },
            },
        },
    },
    {
        "name": "research_schedule",
        "description": "Schedule or update daily research runs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "Time to run daily research in HH:MM format (default: 06:00).",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone (default: America/New_York).",
                },
                "max_searches": {
                    "type": "integer",
                    "description": "Maximum topics to research per run (default: 10).",
                },
            },
        },
    },
    {
        "name": "research_status",
        "description": "Get status of the research scheduler including last run, scheduled tasks, and topic counts.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "research_search_past",
        "description": "Search past research findings in Graphiti knowledge graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for past research.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10).",
                },
            },
            "required": ["query"],
        },
    },
]


class ResearchToolHandler:
    """
    Handles execution of research MCP tools.

    Integrates with ResearchScheduler and related components.
    """

    def __init__(self, background_scheduler=None):
        """
        Initialize tool handler.

        Args:
            background_scheduler: BackgroundTaskScheduler instance for scheduling.
        """
        self.background_scheduler = background_scheduler
        self._scheduler = None
        self._primer = None

    @property
    def scheduler(self):
        """Lazy load ResearchScheduler."""
        if self._scheduler is None:
            from ..research.scheduler import ResearchScheduler
            self._scheduler = ResearchScheduler()
        return self._scheduler

    @property
    def primer(self):
        """Lazy load ContextPrimer."""
        if self._primer is None:
            from ..research.primer import ContextPrimer
            self._primer = ContextPrimer()
        return self._primer

    async def handle_tool(self, name: str, arguments: dict) -> Any:
        """
        Handle a research tool call.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        handlers = {
            "research_run": self._handle_run,
            "research_add_topic": self._handle_add_topic,
            "research_remove_topic": self._handle_remove_topic,
            "research_list_topics": self._handle_list_topics,
            "research_get_context": self._handle_get_context,
            "research_schedule": self._handle_schedule,
            "research_status": self._handle_status,
            "research_search_past": self._handle_search_past,
        }

        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown research tool: {name}"}

        return await handler(arguments)

    async def _handle_run(self, args: dict) -> dict:
        """Handle research_run tool."""
        topics = args.get("topics", [])
        project = args.get("project")
        max_topics = args.get("max_topics")

        result = await self.scheduler.run_now(
            topics=topics if topics else None,
            project=project,
            max_topics=max_topics,
        )
        return result

    async def _handle_add_topic(self, args: dict) -> dict:
        """Handle research_add_topic tool."""
        topic = args["topic"]
        project = args.get("project", "global")

        added = self.scheduler.registry.add_topic(topic, project)

        return {
            "success": added,
            "topic": topic,
            "project": project,
            "message": f"Added topic '{topic}'" if added else f"Topic '{topic}' already exists",
        }

    async def _handle_remove_topic(self, args: dict) -> dict:
        """Handle research_remove_topic tool."""
        topic = args["topic"]
        project = args.get("project", "global")

        removed = self.scheduler.registry.remove_topic(topic, project)

        return {
            "success": removed,
            "topic": topic,
            "project": project,
            "message": f"Removed topic '{topic}'" if removed else f"Topic '{topic}' not found",
        }

    async def _handle_list_topics(self, args: dict) -> dict:
        """Handle research_list_topics tool."""
        project = args.get("project")

        registry = self.scheduler.registry

        if project:
            topics = registry.get_topics_for_project(project)
            return {
                "project": project,
                "topics": topics,
                "count": len(topics),
            }

        return {
            "global_topics": registry.global_topics,
            "project_topics": registry.project_topics,
            "projects": registry.list_projects(),
            "total_topics": len(registry.all_topics),
        }

    async def _handle_get_context(self, args: dict) -> dict:
        """Handle research_get_context tool."""
        date = args.get("date", "latest")
        max_chars = args.get("max_chars", 2000)

        if date == "latest":
            content = self.primer.get_latest_context()
        else:
            content = self.primer.get_context_for_date(date)

        if not content:
            return {
                "found": False,
                "message": "No research context available",
                "available_dates": self.primer.list_available_dates(5),
            }

        # Truncate if needed
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n*[Truncated]*"

        return {
            "found": True,
            "date": date,
            "content": content,
            "length": len(content),
        }

    async def _handle_schedule(self, args: dict) -> dict:
        """Handle research_schedule tool."""
        time = args.get("time", "06:00")
        timezone = args.get("timezone", "America/New_York")
        max_searches = args.get("max_searches", 10)

        # Update registry settings
        registry = self.scheduler.registry
        registry.schedule.run_time = time
        registry.schedule.timezone = timezone
        registry.schedule.max_searches_per_run = max_searches
        registry.save()

        # Schedule if background scheduler available
        if self.background_scheduler:
            task_id = await self.scheduler.schedule_daily_run(
                self.background_scheduler,
                run_time=time,
            )
            return {
                "success": True,
                "task_id": task_id,
                "run_time": time,
                "timezone": timezone,
                "max_searches": max_searches,
            }

        return {
            "success": True,
            "message": "Schedule settings saved (scheduler not active)",
            "run_time": time,
            "timezone": timezone,
            "max_searches": max_searches,
        }

    async def _handle_status(self, args: dict) -> dict:
        """Handle research_status tool."""
        return self.scheduler.get_status()

    async def _handle_search_past(self, args: dict) -> dict:
        """Handle research_search_past tool."""
        query = args["query"]
        max_results = args.get("max_results", 10)

        from ..research.summarizer import ResearchSummarizer
        summarizer = ResearchSummarizer()

        results = await summarizer.search_past_research(query, max_results=max_results)

        return {
            "query": query,
            "results": results,
            "count": len(results),
        }
