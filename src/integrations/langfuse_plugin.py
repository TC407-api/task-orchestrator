"""
Enhanced Langfuse Plugin for Task Orchestrator.

Provides deep integration with Langfuse for:
- Exporting immune system patterns as datasets
- Pushing dashboard metrics
- Creating evaluation datasets from trials
- Aggregating scores across traces
"""

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langfuse import Langfuse


class LangfusePlugin:
    """
    Deep Langfuse integration for task-orchestrator.

    Extends basic tracing with dataset management, metric pushing,
    and evaluation dataset creation.
    """

    def __init__(self, client: Optional["Langfuse"] = None):
        """
        Initialize the Langfuse plugin.

        Args:
            client: Optional Langfuse client. If not provided,
                   will attempt to create one from environment variables.
        """
        self.client = client
        self._initialized = False

        if self.client is None:
            self._init_client()
        else:
            self._initialized = True

    def _init_client(self) -> None:
        """Initialize Langfuse client from environment variables."""
        try:
            from langfuse import Langfuse

            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")

            if not public_key or not secret_key:
                logger.warning("Langfuse credentials not found. Plugin disabled.")
                return

            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            self._initialized = True
            logger.info("Langfuse plugin initialized successfully.")
        except ImportError:
            logger.warning("langfuse package not installed. Plugin disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse plugin: {e}")

    @property
    def enabled(self) -> bool:
        """Check if the plugin is enabled and ready."""
        return self._initialized and self.client is not None

    async def export_immune_patterns(
        self,
        patterns: List[Dict[str, Any]],
        dataset_name: str = "immune-patterns",
    ) -> Dict[str, Any]:
        """
        Export immune system patterns as a Langfuse dataset.

        Args:
            patterns: List of immune patterns to export
            dataset_name: Name for the Langfuse dataset

        Returns:
            Dict with dataset_id and items_exported count
        """
        if not self.enabled:
            return {"error": "Langfuse plugin not enabled", "items_exported": 0}

        try:
            # Create dataset
            dataset = self.client.create_dataset(
                name=dataset_name,
                description=f"Immune patterns exported at {datetime.now().isoformat()}",
            )

            # Add each pattern as a dataset item
            for pattern in patterns:
                self.client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=pattern,
                    expected_output={"pattern_type": pattern.get("pattern_type", "unknown")},
                    metadata={
                        "pattern_id": pattern.get("pattern_id"),
                        "frequency": pattern.get("frequency", 0),
                        "last_seen": pattern.get("last_seen"),
                    },
                )

            self.client.flush()

            return {
                "dataset_id": dataset.id,
                "items_exported": len(patterns),
                "dataset_name": dataset_name,
            }
        except Exception as e:
            logger.error(f"Error exporting immune patterns: {e}")
            return {"error": str(e), "items_exported": 0}

    async def push_dashboard_metrics(
        self,
        metrics: Dict[str, Any],
    ) -> bool:
        """
        Push dashboard metrics to Langfuse as a trace.

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            trace = self.client.trace(
                name="dashboard-metrics",
                input={"metrics_pushed": list(metrics.keys())},
                output=metrics,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "metric_count": len(metrics),
                },
                tags=["metrics", "dashboard"],
            )

            # Add individual scores for each metric
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.client.score(
                        trace_id=trace.id,
                        name=name,
                        value=float(value),
                    )

            self.client.flush()
            return True
        except Exception as e:
            logger.error(f"Error pushing dashboard metrics: {e}")
            return False

    async def create_evaluation_dataset(
        self,
        name: str,
        trials: List[Any],
        description: str = "",
    ) -> str:
        """
        Create a Langfuse dataset from evaluation trials.

        Args:
            name: Name for the dataset
            trials: List of Trial objects
            description: Optional description

        Returns:
            Dataset ID if successful, empty string otherwise
        """
        if not self.enabled:
            return ""

        try:
            dataset = self.client.create_dataset(
                name=name,
                description=description or f"Evaluation dataset: {name}",
            )

            for trial in trials:
                input_data = {
                    "prompt": getattr(trial, "input_prompt", ""),
                    "operation": getattr(trial, "operation", ""),
                    "model": getattr(trial, "model", ""),
                }

                output_data = {
                    "output": str(getattr(trial, "output", "")),
                    "cost_usd": getattr(trial, "cost_usd", 0),
                }

                grader_results = getattr(trial, "grader_results", [])
                if grader_results:
                    output_data["grader_scores"] = {
                        r.name: r.score for r in grader_results
                    }

                self.client.create_dataset_item(
                    dataset_name=name,
                    input=input_data,
                    expected_output=output_data,
                    metadata={"trial_id": getattr(trial, "id", "unknown")},
                )

            self.client.flush()
            return dataset.id
        except Exception as e:
            logger.error(f"Error creating evaluation dataset: {e}")
            return ""

    async def aggregate_scores(
        self,
        trace_ids: List[str],
    ) -> Dict[str, float]:
        """
        Aggregate grader scores across multiple traces.

        Args:
            trace_ids: List of trace IDs to aggregate

        Returns:
            Dict mapping score name to average value
        """
        if not self.enabled:
            return {}

        try:
            score_sums: Dict[str, float] = {}
            score_counts: Dict[str, int] = {}

            for trace_id in trace_ids:
                try:
                    trace = self.client.get_trace(trace_id)
                    if hasattr(trace, "scores"):
                        for score in trace.scores:
                            name = score.name
                            value = score.value
                            score_sums[name] = score_sums.get(name, 0) + value
                            score_counts[name] = score_counts.get(name, 0) + 1
                except Exception:
                    continue

            # Calculate averages
            return {
                name: score_sums[name] / score_counts[name]
                for name in score_sums
                if score_counts.get(name, 0) > 0
            }
        except Exception as e:
            logger.error(f"Error aggregating scores: {e}")
            return {}

    def flush(self) -> None:
        """Flush pending events to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")


def create_langfuse_plugin(
    client: Optional["Langfuse"] = None,
) -> Optional[LangfusePlugin]:
    """
    Factory function to create a LangfusePlugin instance.

    Args:
        client: Optional pre-configured Langfuse client

    Returns:
        LangfusePlugin instance or None if initialization fails
    """
    try:
        plugin = LangfusePlugin(client=client)
        if plugin.enabled:
            return plugin
        return plugin  # Return even if not enabled for testing
    except Exception as e:
        logger.error(f"Failed to create Langfuse plugin: {e}")
        return None


# Global plugin instance (lazy initialization)
_plugin_instance: Optional[LangfusePlugin] = None


def get_langfuse_plugin() -> Optional[LangfusePlugin]:
    """Get or create the global LangfusePlugin instance."""
    global _plugin_instance
    if _plugin_instance is None:
        _plugin_instance = create_langfuse_plugin()
    return _plugin_instance


async def handle_langfuse_export(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool handler for langfuse_export.

    Exports data to Langfuse for analysis.

    Args:
        args: Tool arguments with 'type' key (patterns, trials, metrics)

    Returns:
        Result dict with success status or error
    """
    export_type = args.get("type", "")
    dataset_name = args.get("dataset_name", f"export-{datetime.now().strftime('%Y%m%d')}")

    plugin = get_langfuse_plugin()
    if plugin is None or not plugin.enabled:
        return {"error": "Langfuse plugin not configured", "success": False}

    try:
        if export_type == "patterns":
            # Get immune patterns from failure store
            try:
                from ..evaluation.immune_system.failure_store import get_failure_store
                store = get_failure_store()
                raw_patterns = store.get_all_patterns() if store else []
                # Convert to dict format
                patterns = [
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type,
                        "signature": p.signature,
                        "frequency": p.frequency,
                        "last_seen": p.last_seen.isoformat() if hasattr(p.last_seen, 'isoformat') else str(p.last_seen),
                    }
                    for p in raw_patterns
                ]
            except (ImportError, AttributeError):
                patterns = []

            result = await plugin.export_immune_patterns(
                patterns=patterns,
                dataset_name=dataset_name
            )
            return {"success": True, **result}

        elif export_type == "trials":
            # Get recent trials from evaluation system
            try:
                from ..evaluation import get_trial_store
                store = get_trial_store()
                trials = store.get_recent_trials(limit=100) if store else []
            except ImportError:
                trials = []

            dataset_id = await plugin.create_evaluation_dataset(
                name=dataset_name,
                trials=trials
            )
            return {"success": True, "dataset_id": dataset_id}

        elif export_type == "metrics":
            # Get metrics from cost tracker
            try:
                from ..core.cost_tracker import get_cost_tracker
                tracker = get_cost_tracker()
                metrics = tracker.get_summary() if tracker else {}
            except ImportError:
                metrics = {}

            result = await plugin.push_dashboard_metrics(metrics)
            return {"success": result}

        else:
            return {"error": f"Invalid export type: {export_type}", "success": False}

    except Exception as e:
        logger.error(f"Error in langfuse_export: {e}")
        return {"error": str(e), "success": False}


__all__ = [
    "LangfusePlugin",
    "create_langfuse_plugin",
    "get_langfuse_plugin",
    "handle_langfuse_export",
]
