"""Archetype agent, inbox, and audit handler methods for MCP server."""
import logging
import time
from ..agents.audit_workflow import AuditWorkflow
from ..agents.inbox import AgentEvent, EventType, ActionRiskLevel
from ..evaluation import (
    Trial, GraderPipeline, NonEmptyGrader, LengthGrader,
    score_trial, get_exporter, get_immune_system,
)
from ..observability import trace_operation

logger = logging.getLogger(__name__)


class ArchetypeHandlers:
    """Mixin providing archetype agent, inbox, and audit handler methods."""

    def _ensure_audit_workflow(self, project_root=None):
        """Initialize or re-initialize audit workflow."""
        if not self._audit_workflow or (
            project_root and str(self._audit_workflow.project_root) != project_root
        ):
            self._audit_workflow = AuditWorkflow(project_root=project_root)

    @trace_operation("spawn_archetype_agent")
    async def _handle_spawn_archetype_agent(self, args: dict) -> dict:
        """Spawn an agent with a specific archetype role."""
        from ..core.cost_tracker import Provider
        can_proceed, retry_after = self._gemini_breaker.is_available()
        if not can_proceed:
            return {
                "error": f"Gemini service circuit breaker is open. Retry after {retry_after:.1f}s",
                "circuit_breaker_open": True,
                "retry_after_seconds": retry_after,
            }
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return {"error": msg, "budget_exceeded": True}
        if not self.coordinator.llm:
            return {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}
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
        archetype_config = self._archetype_registry.get_archetype_config(archetype)
        system_prompt = archetype_config.system_prompt
        temperature = archetype_config.temperature
        if inject_audit:
            self._ensure_audit_workflow()
            system_prompt = self._audit_workflow.inject_to_prompt(system_prompt)
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
        start_event = AgentEvent(
            event_type=EventType.AGENT_START,
            agent_name=f"{archetype_name}_agent",
            data={"archetype": archetype_name, "model": model, "prompt_preview": prompt[:200]},
            source="spawn_archetype_agent",
        )
        await self._universal_inbox.publish(start_event)
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
                prompt, model=model, system_prompt=system_prompt,
                max_tokens=max_tokens, temperature=temperature,
            )
            trial.latency_ms = (time.time() - start_time) * 1000
            trial.output = response.content
            trial.cost_usd = response.usage.get("estimated_cost_usd", 0) if response.usage else 0
            pipeline = GraderPipeline([NonEmptyGrader(), LengthGrader(min_length=10, max_length=100000)])
            grader_results = await pipeline.run(response.content, {"prompt": original_prompt})
            for result in grader_results:
                trial.add_grader_result(result)
            try:
                await score_trial(trial)
            except Exception:
                logger.warning("Failed to score trial", exc_info=True)
            try:
                exporter = get_exporter()
                exporter.add_trial(trial)
            except Exception:
                logger.warning("Failed to export trial", exc_info=True)
            end_event = AgentEvent(
                event_type=EventType.AGENT_END,
                agent_name=f"{archetype_name}_agent",
                data={"archetype": archetype_name, "success": True, "latency_ms": trial.latency_ms},
                source="spawn_archetype_agent",
            )
            await self._universal_inbox.publish(end_event)
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
            error_event = AgentEvent(
                event_type=EventType.ERROR,
                agent_name=f"{archetype_name}_agent",
                data={"error": "Agent execution failed", "archetype": archetype_name},
                source="spawn_archetype_agent",
            )
            await self._universal_inbox.publish(error_event)
            logger.error("Archetype agent execution failed", exc_info=True)
            return {"success": False, "error": "Agent execution failed"}

    @trace_operation("inbox_status")
    async def _handle_inbox_status(self, args: dict) -> dict:
        """Get universal inbox status including pending approvals."""
        risk_level_str = args.get("risk_level")
        agent_name = args.get("agent_name")
        include_history = args.get("include_history", False)
        history_limit = args.get("history_limit", 20)
        risk_level = None
        if risk_level_str:
            try:
                risk_level = ActionRiskLevel(risk_level_str)
            except ValueError:
                pass
        pending = self._universal_inbox.get_pending_approvals(risk_level=risk_level, agent_name=agent_name)
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
        if include_history:
            result["event_history"] = self._universal_inbox.get_event_history(
                agent_name=agent_name, limit=history_limit
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
                action = await self._universal_inbox.approve(action_id=action_id, approved_by=approved_by)
                return {
                    "success": True, "action_id": action_id, "status": "approved",
                    "approved_by": approved_by, "execution_result": action.execution_result,
                }
            else:
                if not reason:
                    return {"success": False, "error": "Reason is required when rejecting an action"}
                action = await self._universal_inbox.reject(action_id=action_id, reason=reason, rejected_by=approved_by)
                return {
                    "success": True, "action_id": action_id, "status": "rejected",
                    "rejected_by": approved_by, "reason": reason,
                }
        except ValueError as e:
            logger.error("Action approval/rejection failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}

    @trace_operation("audit_status")
    async def _handle_audit_status(self, args: dict) -> dict:
        """Get audit workflow status."""
        project_root = args.get("project_root")
        query_topic = args.get("query_topic")
        query_error_type = args.get("query_error_type")
        self._ensure_audit_workflow(project_root)
        result = {
            "success": True,
            "summary": self._audit_workflow.get_summary(),
            "audit_file": str(self._audit_workflow.audit_file),
            "file_exists": self._audit_workflow.audit_file.exists(),
        }
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
        self._ensure_audit_workflow(project_root)
        try:
            self._audit_workflow.append_entry(entry_type=entry_type, content=content, title=title, metadata=metadata)
            entry_count = len(self._audit_workflow.audit_data.get(entry_type + "s", []))
            return {
                "success": True,
                "entry_type": entry_type,
                "title": title or f"Entry {entry_count}",
                "audit_file": str(self._audit_workflow.audit_file),
            }
        except ValueError as e:
            logger.error("Audit append failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}

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
        return {"success": True, "archetypes": self._archetype_registry.get_summary()}
