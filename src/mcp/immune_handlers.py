"""Immune system handler methods for MCP server."""
import logging
from ..observability import trace_operation

logger = logging.getLogger(__name__)


class ImmuneHandlers:
    """Mixin providing immune system handler methods."""

    def _get_alert_manager(self):
        """Lazy-initialize alert manager singleton."""
        if self._alert_manager is None:
            from ..evaluation import AlertManager, HighRiskThreshold, NewPatternDetected
            self._alert_manager = AlertManager(
                rules=[HighRiskThreshold(), NewPatternDetected()]
            )
        return self._alert_manager

    def _get_predictor(self):
        """Lazy-initialize failure predictor singleton."""
        if self._predictor is None:
            from ..evaluation import FailurePredictor
            self._predictor = FailurePredictor()
        return self._predictor

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

        suggestions = await immune.get_suggestions(prompt, operation)
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

        stats = immune.get_stats()
        failure_stats = stats.get("failure_store", {})
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
            return {"success": True, "format": "json", "report": dashboard.format_as_json()}
        else:
            return {"success": True, "format": "markdown", "report": dashboard.format_as_markdown()}

    @trace_operation("immune_sync")
    async def _handle_immune_sync(self, args: dict) -> dict:
        """Synchronize immune system with Graphiti."""
        from ..evaluation import get_immune_system

        immune = get_immune_system()

        try:
            result = await immune.sync_with_graphiti()
            return {"success": True, "sync_result": result}
        except Exception as e:
            logger.error("Immune sync failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}

    @trace_operation("alert_list")
    async def _handle_alert_list(self, args: dict) -> dict:
        """List recent alerts from the alerting system."""
        alert_manager = self._get_alert_manager()

        limit = args.get("limit", 10)
        severity = args.get("severity")

        if severity:
            alerts = alert_manager.get_active_alerts(severity=severity)
        else:
            alerts = alert_manager.get_recent_alerts(limit=limit)

        return {
            "success": True,
            "alerts": alerts,
            "stats": alert_manager.get_stats(),
        }

    @trace_operation("alert_clear")
    async def _handle_alert_clear(self, args: dict) -> dict:
        """Clear all active alerts."""
        if self._alert_manager is None:
            return {"success": True, "cleared": 0}

        cleared = self._alert_manager.clear_active_alerts()
        return {"success": True, "cleared": cleared}

    @trace_operation("predict_risk")
    async def _handle_predict_risk(self, args: dict) -> dict:
        """Predict failure risk for a prompt using ML model."""
        prompt = args["prompt"]
        tool = args.get("tool", "spawn_agent")

        predictor = self._get_predictor()
        result = predictor.predict(prompt, tool)

        return {
            "success": True,
            "prediction": result.to_dict(),
            "model_active": predictor.is_active,
        }
