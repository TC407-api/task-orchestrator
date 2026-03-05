"""Unified Gemini agent spawn logic for MCP server."""
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

from ..evaluation import (
    Trial, GraderPipeline, NonEmptyGrader, LengthGrader,
    score_trial, get_exporter, get_immune_system,
)
from ..core.cost_tracker import Provider
from ..observability import trace_operation

_DEFAULT_SYSTEM_PROMPT = """You are an expert code assistant.

CRITICAL - SELF-VALIDATION REQUIRED:
Before claiming any task is complete, validate your work:
1. Run tests (npm test, pytest, go test)
2. Run build (npm run build, cargo build)
3. Run type check (npx tsc --noEmit, mypy)

If ANY validation fails, fix and re-run until all pass.
NEVER say done without showing passing validation output.
NEVER say should work - RUN IT and show the output."""


async def run_single_agent(
    *,
    coordinator,
    gemini_breaker,
    cost_tracker,
    original_prompt: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    operation: str = "spawn_agent",
    agent_id: int = 0,
    extra_metadata: dict | None = None,
) -> dict:
    """Run a single Gemini agent with immune system integration."""
    immune = get_immune_system()
    immune_response = await immune.pre_spawn_check(original_prompt, operation)
    if not immune_response.should_proceed:
        base: dict = {"success": False, "error": "Request blocked by Immune System", "immune_blocked": True, "risk_score": immune_response.risk_score, "warnings": immune_response.warnings, "evaluation": {"passed": False, "scores": []}}
        if agent_id:
            base["agent_id"] = agent_id
        return base
    prompt = immune_response.processed_prompt
    trial = Trial(operation=operation, input_prompt=original_prompt, model=model, circuit_breaker_state=gemini_breaker._state.value)
    trial.metadata = {**(extra_metadata or {})}
    if agent_id:
        trial.metadata["agent_id"] = agent_id
    start_time = time.time()
    try:
        response = await coordinator.llm.generate(prompt, model=model, system_prompt=system_prompt, max_tokens=max_tokens)
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
            get_exporter().add_trial(trial)
        except Exception:
            logger.warning("Failed to export trial", exc_info=True)
        if not trial.pass_fail:
            try:
                await immune.record_failure(operation=operation, prompt=original_prompt, output=response.content, grader_results=[r.to_dict() for r in trial.grader_results], context={"model": model, "agent_id": agent_id})
            except Exception:
                logger.warning("Failed to record immune failure", exc_info=True)
        base = {"success": True, "response": response.content, "model": response.model, "usage": response.usage, "evaluation": {"passed": trial.pass_fail, "scores": [r.to_dict() for r in trial.grader_results]}, "immune": {"risk_score": immune_response.risk_score, "guardrails_applied": immune_response.guardrails_applied, "prompt_modified": immune_response.original_prompt != immune_response.processed_prompt}}
        if agent_id:
            base["agent_id"] = agent_id
        return base
    except Exception:
        trial.latency_ms = (time.time() - start_time) * 1000
        logger.error("Agent execution failed", exc_info=True)
        base = {"success": False, "error": "Agent execution failed", "evaluation": {"passed": False, "scores": []}}
        if agent_id:
            base["agent_id"] = agent_id
        return base


class AgentRunnerHandlers:
    """Mixin providing spawn_agent and spawn_parallel_agents handlers."""

    def _check_agent_preconditions(self):
        """Check circuit breaker and budget. Returns (ok, error_dict_or_None)."""
        can_proceed, retry_after = self._gemini_breaker.is_available()
        if not can_proceed:
            return False, {"error": f"Gemini service circuit breaker is open. Retry after {retry_after:.1f}s", "circuit_breaker_open": True, "retry_after_seconds": retry_after}
        can_proceed, msg = self.cost_tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            return False, {"error": msg, "budget_exceeded": True}
        if not self.coordinator.llm:
            return False, {"error": "LLM not configured. Set GOOGLE_API_KEY in .env"}
        return True, None

    @trace_operation("spawn_agent")
    async def _handle_spawn_agent(self, args: dict) -> dict:
        """Spawn a Gemini agent to execute a code task."""
        ok, err = self._check_agent_preconditions()
        if not ok:
            return err
        model = args.get("model", "gemini-3-flash-preview")
        system_prompt = args.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        max_tokens = args.get("max_tokens", 8192)
        original_prompt = args["prompt"]
        result = await run_single_agent(coordinator=self.coordinator, gemini_breaker=self._gemini_breaker, cost_tracker=self.cost_tracker, original_prompt=original_prompt, model=model, system_prompt=system_prompt, max_tokens=max_tokens, operation="spawn_agent")
        if result.get("success"):
            self._gemini_breaker.record_success()
        else:
            self._gemini_breaker.record_failure(Exception(result.get("error", "agent failed")))
        return result

    @trace_operation("spawn_parallel_agents")
    async def _handle_spawn_parallel_agents(self, args: dict) -> dict:
        """Spawn multiple Gemini agents in parallel with evaluation."""
        ok, err = self._check_agent_preconditions()
        if not ok:
            return err
        prompts = args["prompts"]
        model = args.get("model", "gemini-3-flash-preview")
        system_prompt = args.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        max_tokens = args.get("max_tokens", 8192)
        tasks = [run_single_agent(coordinator=self.coordinator, gemini_breaker=self._gemini_breaker, cost_tracker=self.cost_tracker, original_prompt=p, model=model, system_prompt=system_prompt, max_tokens=max_tokens, operation="spawn_parallel_agent", agent_id=i, extra_metadata={"mode": "parallel"}) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        successes = sum(1 for r in results if r.get("success"))
        failures = len(results) - successes
        eval_passed = sum(1 for r in results if r.get("evaluation", {}).get("passed", False))
        if successes > 0:
            self._gemini_breaker.record_success()
        if failures > 0:
            self._gemini_breaker.record_failure(Exception(f"{failures} agents failed"))
        if eval_passed < successes:
            self._gemini_breaker.record_semantic_failure("parallel_eval_failed")
        return {"success": failures == 0, "total_agents": len(prompts), "succeeded": successes, "failed": failures, "evaluations_passed": eval_passed, "results": list(results)}
