"""MCP Server for Task Orchestrator integration with Claude Code."""
import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Optional

from ..core.cost_tracker import Provider, get_cost_tracker
from ..agents.coordinator import CoordinatorAgent, TaskStatus
from ..agents.email_agent import TaskPriority
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

    @trace_operation("spawn_agent")
    async def _handle_spawn_agent(self, args: dict) -> dict:
        """Spawn a Gemini agent to execute a code task."""
        import time
        from ..evaluation import (
            Trial, GraderPipeline, NonEmptyGrader, LengthGrader,
            score_trial, get_exporter
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
        prompt = args["prompt"]
        system_prompt = args.get("system_prompt", "You are an expert code assistant. Provide clear, working code solutions.")
        max_tokens = args.get("max_tokens", 8192)

        # Create Trial for evaluation
        trial = Trial(
            operation="spawn_agent",
            input_prompt=prompt,
            model=model,
            circuit_breaker_state=self._gemini_breaker._state.value,
        )

        start_time = time.time()

        try:
            response = await self.coordinator.llm.generate(
                prompt,
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
            grader_results = await pipeline.run(response.content, {"prompt": prompt})

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
            score_trial, get_exporter
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

        async def run_single_agent(prompt: str, agent_id: int) -> dict:
            # Create Trial for this agent
            trial = Trial(
                operation="spawn_parallel_agent",
                input_prompt=prompt,
                model=model,
                circuit_breaker_state=self._gemini_breaker._state.value,
            )
            trial.metadata = {"agent_id": agent_id, "mode": "parallel"}

            start_time = time.time()

            try:
                response = await self.coordinator.llm.generate(
                    prompt,
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
                grader_results = await pipeline.run(response.content, {"prompt": prompt})

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
