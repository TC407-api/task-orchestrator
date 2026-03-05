"""Workflow, terminal, and background-task handler methods for MCP server."""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
from ..agents.background_tasks import BackgroundTaskScheduler, ScheduledTask, TaskScheduleType
from ..agents.terminal_loop import TerminalListener
from ..agents.workflows import WorkflowRegistry, WorkflowTrigger
from ..observability import trace_operation

ALLOWED_COMMANDS = {
    "npm", "npx", "node", "python", "python3", "pytest", "pip",
    "git", "echo", "ls", "dir", "cat", "head", "tail", "grep",
    "curl", "wget", "docker", "docker-compose",
}


class WorkflowHandlers:
    def _get_terminal_listener(self):
        if self._terminal_listener is None:
            self._terminal_listener = TerminalListener(inbox=self._universal_inbox)
        return self._terminal_listener

    async def _get_background_scheduler(self) -> BackgroundTaskScheduler:
        if self._background_scheduler is None:
            self._background_scheduler = BackgroundTaskScheduler(inbox=self._universal_inbox)
            await self._background_scheduler.start()
        return self._background_scheduler

    @trace_operation("run_with_error_capture")
    async def _handle_run_with_error_capture(self, args: dict) -> dict:
        command = args["command"]
        working_dir = args.get("working_dir")
        timeout = args.get("timeout", 60)
        listener = self._get_terminal_listener()
        try:
            result = await listener.run_command(command=command, working_dir=working_dir, timeout=timeout)
            return {"success": result.exit_code == 0, "exit_code": result.exit_code, "stdout": result.stdout[:5000] if result.stdout else "", "stderr": result.stderr[:5000] if result.stderr else "", "error_detected": result.error is not None, "error": result.error.to_dict() if result.error else None, "fix_proposals": [p.to_dict() for p in result.fix_proposals] if result.fix_proposals else []}
        except Exception as e:
            logger.error("run_with_error_capture handler failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}

    @trace_operation("validate_code")
    async def _handle_validate_code(self, args: dict) -> dict:
        try:
            result = await self._shadow_validator.validate(code=args["code"], language=args["language"], run_linter=args.get("run_linter", True))
            return {"success": True, "valid": result.valid, "errors": result.errors, "warnings": result.warnings, "suggestions": result.suggestions}
        except Exception as e:
            logger.error("validate_code handler failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}

    @trace_operation("trigger_workflow")
    async def _handle_trigger_workflow(self, args: dict) -> dict:
        workflow_name = args["workflow"]
        prompt = args["prompt"]
        try:
            trigger_name = "@" + workflow_name.capitalize()
            if workflow_name == "testgen":
                trigger_name = "@TestGen"
            workflow = self._workflow_registry.get(trigger_name)
            if not workflow:
                return {"success": False, "error": "Unknown workflow: " + workflow_name, "available": [wf.trigger for wf in self._workflow_registry.list_workflows()]}
            trigger = WorkflowTrigger(registry=self._workflow_registry)
            trigger_prompt = trigger_name + " " + prompt if trigger_name not in prompt else prompt
            result = trigger.process_prompt(prompt=trigger_prompt, base_path=".", include_context=True)
            return {"success": True, "workflow": workflow_name, "archetype": workflow.archetype, "expanded_prompt": result["processed_prompt"], "system_prompt_addition": workflow.system_prompt_addition, "triggers_detected": result["triggers"], "context_injected": result["context_injected"]}
        except Exception as e:
            logger.error("trigger_workflow handler failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}

    @trace_operation("list_workflows")
    async def _handle_list_workflows(self, args: dict) -> dict:
        workflows = self._workflow_registry.list_workflows()
        return {"success": True, "workflows": {wf.trigger: {"trigger": wf.trigger, "archetype": wf.archetype, "max_context_tokens": wf.max_context_tokens, "priority": wf.priority} for wf in workflows}}
    @trace_operation("schedule_task")
    async def _handle_schedule_task(self, args: dict) -> dict:
        import shlex, subprocess
        name = args["name"]
        command = args["command"]
        schedule_type_str = args["schedule_type"]
        run_at = args.get("run_at")
        interval_seconds = args.get("interval_seconds")
        scheduler = await self._get_background_scheduler()
        try:
            schedule_type = TaskScheduleType(schedule_type_str.upper())
            try:
                cmd_parts = shlex.split(command)
            except ValueError as e:
                return {"success": False, "error": "Invalid command syntax: " + str(e)}
            if not cmd_parts:
                return {"success": False, "error": "Empty command"}
            base_cmd = cmd_parts[0].lower().split("/")[-1].split("\\")[-1]
            if base_cmd not in ALLOWED_COMMANDS:
                return {"success": False, "error": "Command not in allowed list: " + base_cmd}
            async def run_command():
                result = subprocess.run(cmd_parts, shell=False, capture_output=True, text=True, timeout=300)
                return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
            task = ScheduledTask(name=name, func=run_command, schedule_type=schedule_type, run_at=datetime.fromisoformat(run_at) if run_at else None, interval_seconds=interval_seconds)
            task_id = await scheduler.schedule_task(task)
            return {"success": True, "task_id": task_id, "name": name, "schedule_type": schedule_type_str}
        except Exception as e:
            logger.error("schedule_task failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}

    @trace_operation("list_scheduled_tasks")
    async def _handle_list_scheduled_tasks(self, args: dict) -> dict:
        if not self._background_scheduler:
            return {"success": True, "tasks": [], "message": "No scheduler initialized"}
        status_filter = args.get("status")
        tasks = self._background_scheduler.list_tasks(status=status_filter)
        return {"success": True, "tasks": [t.to_dict() if hasattr(t, "to_dict") else str(t) for t in tasks]}

    @trace_operation("cancel_scheduled_task")
    async def _handle_cancel_scheduled_task(self, args: dict) -> dict:
        task_id = args["task_id"]
        if not self._background_scheduler:
            return {"success": False, "error": "No scheduler initialized"}
        try:
            await self._background_scheduler.cancel_task(task_id)
            return {"success": True, "task_id": task_id, "status": "cancelled"}
        except Exception as e:
            logger.error("cancel_scheduled_task failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}
