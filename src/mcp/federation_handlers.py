"""Federation and live-sync handler methods for MCP server."""
import logging
import time
from ..observability import trace_operation

logger = logging.getLogger(__name__)


class FederationHandlers:
    def _get_sync_monitor(self):
        if self._sync_monitor is None:
            from ..evaluation.immune_system.live_sync import SyncHealthMonitor
            self._sync_monitor = SyncHealthMonitor()
        return self._sync_monitor

    def _get_sync_engine(self):
        if self._sync_engine is None:
            from ..evaluation.immune_system.live_sync import SyncEngine
            from ..evaluation.immune_system import get_registry_manager
            self._sync_engine = SyncEngine(project_id="task-orchestrator", store=None, transport=None)
            registry = get_registry_manager()
            for pid in registry.projects:
                if pid != "task-orchestrator":
                    self._sync_engine.register_peer(pid, is_subscriber=True, is_subscription=True)
        return self._sync_engine
    @trace_operation("federation_status")
    async def _handle_federation_status(self, args: dict) -> dict:
        from ..evaluation.immune_system import get_decay_system
        federation = await self._get_federation()
        include_projects = args.get("include_projects", False)
        graphiti_status = "connected" if self._graphiti_client else "not configured"
        result = {"success": True, "registry": self._registry.get_stats(), "federation": federation.get_stats(), "decay": get_decay_system().get_stats(), "graphiti_status": graphiti_status}
        if include_projects:
            result["projects"] = [p.to_dict() for p in self._registry.projects.values()]
        return result

    @trace_operation("federation_subscribe")
    async def _handle_federation_subscribe(self, args: dict) -> dict:
        project_id = args["project_id"]
        federation = await self._get_federation()
        project = await self._registry.get_project(project_id)
        if not project:
            return {"success": False, "error": "Project not found: " + project_id, "available_projects": list(self._registry.projects.keys())}
        await federation.subscribe_to_project(project.group_id)
        local_project = await self._registry.get_project("task-orchestrator")
        if local_project and project_id not in local_project.subscriptions:
            local_project.subscriptions.append(project_id)
        return {"success": True, "subscribed_to": project_id, "group_id": project.group_id, "total_subscriptions": len(federation.subscriptions)}
    @trace_operation("federation_search")
    async def _handle_federation_search(self, args: dict) -> dict:
        query = args["query"]
        limit = args.get("limit", 10)
        federation = await self._get_federation()
        try:
            results = await federation.search_global_patterns(query, limit)
            return {"success": True, "query": query, "results_count": len(results),
                "results": [{"pattern_id": r.pattern.id, "operation": r.pattern.operation, "failure_type": r.pattern.failure_type, "relevance_score": r.relevance_score, "source_project": r.source_project, "match_reason": r.match_reason} for r in results]}
        except Exception as e:
            logger.error("Federation search failed", exc_info=True)
            return {"success": False, "error": "Operation failed"}

    @trace_operation("federation_decay")
    async def _handle_federation_decay(self, args: dict) -> dict:
        from ..evaluation.immune_system import get_decay_system, get_immune_system
        action = args.get("action", "status")
        decay = get_decay_system()
        if action == "status":
            return {"success": True, "action": "status", "stats": decay.get_stats()}
        if action == "evaluate":
            immune = get_immune_system()
            store_stats = immune._failure_store.get_stats()
            patterns = immune._failure_store.get_all_patterns()
            pattern_dicts = [{"id": p.id, "decay_metadata": p.context.get("decay_metadata") if p.context else None} for p in patterns]
            return {"success": True, "action": "evaluate", "total_patterns": store_stats["total_patterns"], "evaluation": decay.batch_evaluate(pattern_dicts)}
        if action == "prune_candidates":
            immune = get_immune_system()
            patterns = immune._failure_store.get_all_patterns()
            prune_candidates = []
            for p in patterns:
                metadata = p.context.get("decay_metadata") if p.context else None
                if decay.should_prune(p.id, metadata):
                    prune_candidates.append({"id": p.id, "operation": p.operation, "failure_type": p.failure_type, "current_score": decay.get_current_relevance(p.id, metadata)})
            return {"success": True, "action": "prune_candidates", "count": len(prune_candidates), "candidates": prune_candidates}
        return {"success": False, "error": "Unknown action: " + action}
    @trace_operation("sync_status")
    async def _handle_sync_status(self, args: dict) -> dict:
        sync_monitor = self._get_sync_monitor()
        project_id = args.get("project_id")
        if project_id:
            status = sync_monitor.get_project_status(project_id)
            if status:
                return {"success": True, "project": status}
            return {"success": False, "error": "Project not found: " + project_id}
        return {"success": True, "dashboard": sync_monitor.get_dashboard_metrics()}

    @trace_operation("sync_trigger")
    async def _handle_sync_trigger(self, args: dict) -> dict:
        direction = args.get("direction", "both")
        project_id = args.get("project_id")
        sync_engine = self._get_sync_engine()
        result = {"success": True, "direction": direction, "timestamp": time.time()}
        for sync_dir in ("pull", "push"):
            if direction not in (sync_dir, "both"):
                continue
            if project_id:
                msg = "Sync not configured (no transport)" if project_id in sync_engine._sync_states else "Project not registered"
                result[sync_dir] = {project_id: msg}
            elif sync_dir == "pull":
                result["pull"] = sync_engine.trigger_pull_sync()
            else:
                result["push"] = sync_engine.trigger_push_sync()
        return result

    @trace_operation("sync_alerts")
    async def _handle_sync_alerts(self, args: dict) -> dict:
        from ..evaluation.immune_system.live_sync import SyncStatus
        sync_monitor = self._get_sync_monitor()
        severity_filter = args.get("severity")
        alerts = sync_monitor.check_health_and_alert()
        if severity_filter:
            try:
                target_status = SyncStatus(severity_filter)
                alerts = [a for a in alerts if a.severity == target_status]
            except ValueError:
                pass
        return {"success": True, "alert_count": len(alerts), "alerts": [{"project_id": a.project_id, "severity": a.severity.value, "message": a.message, "timestamp": a.timestamp} for a in alerts]}
