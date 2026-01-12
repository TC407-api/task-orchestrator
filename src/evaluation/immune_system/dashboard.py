"""
Immune System Dashboard.

This module provides visualization and reporting capabilities for the Immune System,
aggregating failure patterns, risk scores, and guardrail metrics into human-readable
and machine-parsable formats.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# --- Protocols for Type Safety ---

class FailurePatternProtocol(Protocol):
    """Protocol defining the expected structure of a FailurePattern."""
    id: str
    operation: str
    failure_type: str
    occurrence_count: int
    created_at: datetime
    grader_scores: Dict[str, float]


class ImmuneSystemProtocol(Protocol):
    """Protocol defining the interface required from the ImmuneSystem."""

    def get_health(self) -> Dict[str, Any]:
        ...

    def get_stats(self) -> Dict[str, Any]:
        ...


class FailureStoreProtocol(Protocol):
    """Protocol for the failure store."""

    def get_all_patterns(self) -> List[FailurePatternProtocol]:
        ...

    def get_stats(self) -> Dict[str, Any]:
        ...


# --- Dashboard Implementation ---

class ImmuneDashboard:
    """
    Dashboard for analyzing and visualizing Immune System health and metrics.
    """

    def __init__(
        self,
        immune_system: ImmuneSystemProtocol,
        failure_store: Optional[FailureStoreProtocol] = None,
    ):
        """
        Initialize the dashboard with a reference to the immune system.

        Args:
            immune_system: An instance adhering to ImmuneSystemProtocol.
            failure_store: Optional direct reference to the failure store.
        """
        self.system = immune_system
        self._failure_store = failure_store

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate high-level health metrics.

        Returns:
            Dict containing system health, total patterns, risk rates, etc.
        """
        health = self.system.get_health()
        stats = self.system.get_stats()

        return {
            "status": health.get("status", "unknown"),
            "total_failure_patterns": health.get("total_patterns", 0),
            "total_occurrences": health.get("total_occurrences", 0),
            "checks_performed": health.get("checks_performed", 0),
            "block_rate": round(health.get("block_rate", 0.0) * 100, 2),
            "guardrail_rate": round(health.get("guardrail_rate", 0.0) * 100, 2),
            "graphiti_available": health.get("graphiti_available", False),
            "graphiti_syncs": health.get("graphiti_syncs", 0),
            "generated_at": datetime.now().isoformat(),
        }

    def get_failure_trends(self, days: int = 7) -> Dict[str, int]:
        """
        Calculate failure counts over time periods.

        Args:
            days: Number of past days to analyze.

        Returns:
            Dict mapping date strings (YYYY-MM-DD) to failure counts.
        """
        if not self._failure_store:
            return {}

        patterns = self._failure_store.get_all_patterns()
        cutoff = datetime.now() - timedelta(days=days)

        trends: Dict[str, int] = defaultdict(int)

        for pattern in patterns:
            if pattern.created_at and pattern.created_at >= cutoff:
                date_key = pattern.created_at.strftime('%Y-%m-%d')
                trends[date_key] += pattern.occurrence_count

        # Fill in missing days with 0
        result = {}
        for i in range(days):
            d = (datetime.now() - timedelta(days=days - 1 - i)).strftime('%Y-%m-%d')
            result[d] = trends.get(d, 0)

        return result

    def get_top_patterns(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Identify the most frequent failure patterns.

        Args:
            n: Number of top patterns to return.

        Returns:
            List of pattern dictionaries sorted by occurrence count.
        """
        if not self._failure_store:
            return []

        patterns = self._failure_store.get_all_patterns()

        # Sort by occurrence count (desc), then by created_at (desc)
        sorted_patterns = sorted(
            patterns,
            key=lambda p: (p.occurrence_count, p.created_at or datetime.min),
            reverse=True,
        )

        return [
            {
                "id": p.id,
                "operation": p.operation,
                "failure_type": p.failure_type,
                "occurrences": p.occurrence_count,
                "last_seen": p.created_at.isoformat() if p.created_at else None,
                "avg_score": (
                    sum(p.grader_scores.values()) / len(p.grader_scores)
                    if p.grader_scores else 0.0
                ),
            }
            for p in sorted_patterns[:n]
        ]

    def get_failure_by_type(self) -> Dict[str, int]:
        """
        Get failure counts grouped by failure type.

        Returns:
            Dict mapping failure types to counts.
        """
        stats = self.system.get_stats()
        return stats.get("failure_store", {}).get("by_type", {})

    def get_failure_by_operation(self) -> Dict[str, int]:
        """
        Get failure counts grouped by operation.

        Returns:
            Dict mapping operations to counts.
        """
        stats = self.system.get_stats()
        return stats.get("failure_store", {}).get("by_operation", {})

    def format_as_json(self) -> str:
        """
        Return a comprehensive dashboard report as a JSON string.
        """
        report = {
            "summary": self.get_summary(),
            "trends": self.get_failure_trends(),
            "top_patterns": self.get_top_patterns(),
            "by_type": self.get_failure_by_type(),
            "by_operation": self.get_failure_by_operation(),
        }
        return json.dumps(report, indent=2, default=str)

    def format_as_markdown(self) -> str:
        """
        Return a human-readable dashboard report in Markdown format.
        """
        summary = self.get_summary()
        trends = self.get_failure_trends()
        top_patterns = self.get_top_patterns(n=5)
        by_type = self.get_failure_by_type()

        md_lines = []
        md_lines.append("# Immune System Dashboard")
        md_lines.append(f"**Generated:** {summary['generated_at']}")
        md_lines.append("")

        # 1. Health Summary
        md_lines.append("## System Health")
        md_lines.append("| Metric | Value |")
        md_lines.append("|:---|:---|")
        md_lines.append(f"| Status | {summary['status']} |")
        md_lines.append(f"| Total Patterns | {summary['total_failure_patterns']} |")
        md_lines.append(f"| Total Occurrences | {summary['total_occurrences']} |")
        md_lines.append(f"| Checks Performed | {summary['checks_performed']} |")
        md_lines.append(f"| Block Rate | {summary['block_rate']}% |")
        md_lines.append(f"| Guardrail Rate | {summary['guardrail_rate']}% |")
        md_lines.append(f"| Graphiti Available | {summary['graphiti_available']} |")
        md_lines.append("")

        # 2. Trends
        md_lines.append("## Failure Trends (Last 7 Days)")
        if not any(trends.values()):
            md_lines.append("_No failures recorded in the last 7 days._")
        else:
            md_lines.append("| Date | Count |")
            md_lines.append("|:---|:---|")
            max_count = max(trends.values()) if trends.values() else 1
            for date, count in trends.items():
                bar_len = int((count / max_count) * 10) if max_count > 0 else 0
                bar = "*" * bar_len
                md_lines.append(f"| {date} | {count} {bar} |")
        md_lines.append("")

        # 3. Top Patterns
        md_lines.append("## Top Failure Patterns")
        if not top_patterns:
            md_lines.append("_No failure patterns detected._")
        else:
            md_lines.append("| Operation | Type | Count | Avg Score |")
            md_lines.append("|:---|:---|:---|:---|")
            for p in top_patterns:
                md_lines.append(
                    f"| {p['operation']} | {p['failure_type']} | "
                    f"{p['occurrences']} | {p['avg_score']:.2f} |"
                )
        md_lines.append("")

        # 4. By Type Breakdown
        md_lines.append("## Failures by Type")
        if not by_type:
            md_lines.append("_No data._")
        else:
            md_lines.append("| Type | Count |")
            md_lines.append("|:---|:---|")
            for ftype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                md_lines.append(f"| {ftype} | {count} |")

        return "\n".join(md_lines)


def create_dashboard(immune_system: ImmuneSystemProtocol) -> ImmuneDashboard:
    """
    Factory function to create a dashboard from an immune system.

    Args:
        immune_system: The immune system instance.

    Returns:
        Configured ImmuneDashboard instance.
    """
    # Try to get failure store from immune system if it has one
    failure_store = getattr(immune_system, '_failure_store', None)
    return ImmuneDashboard(immune_system, failure_store)


__all__ = [
    "ImmuneDashboard",
    "create_dashboard",
]
