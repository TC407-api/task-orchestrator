"""
Phase 8.3: Cross-Project Pattern Federation.

This module enables sharing and synchronization of failure patterns
across different projects.

DEPRECATION NOTE (Phase 4):
Federation is being deprecated in favor of Titan Memory's native project filtering.
- Subscriptions are now no-ops (Titan handles automatically)
- Search uses Titan when available, falls back to Graphiti
- Use titan_recall --project global for cross-project patterns
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .failure_store import FailurePattern

if TYPE_CHECKING:
    from .titan_client import TitanClient

logger = logging.getLogger(__name__)


class PatternVisibility(str, Enum):
    """Visibility settings for patterns."""
    PRIVATE = "private"
    SHARED = "shared"


@dataclass
class FederatedPattern:
    """
    A failure pattern with federation metadata.

    Extends FailurePattern with lineage and visibility tracking.
    """
    pattern: FailurePattern
    visibility: PatternVisibility = PatternVisibility.PRIVATE
    version: int = 1
    original_source_project: Optional[str] = None
    original_pattern_id: Optional[str] = None
    derived_from_id: Optional[str] = None


@dataclass
class ScoredPattern:
    """Result wrapper for federated search results."""
    pattern: FailurePattern
    relevance_score: float
    source_project: str
    match_reason: str


class PatternFederation:
    """
    Manages cross-project pattern sharing.

    DEPRECATION: This class is being deprecated in favor of Titan Memory's
    native project filtering. When titan_client is provided, subscriptions
    become no-ops and search uses Titan directly.

    Enables:
    - Subscribing to patterns from other projects (deprecated - use Titan)
    - Publishing local patterns for discovery
    - Searching across subscribed projects (uses Titan when available)
    - Importing patterns with lineage tracking
    """

    def __init__(
        self,
        graphiti_client: Any,
        local_group_id: str,
        subscriptions: Optional[Set[str]] = None,
        titan_client: Optional["TitanClient"] = None,
    ):
        """
        Initialize the federation system.

        Args:
            graphiti_client: The Graphiti client instance (legacy)
            local_group_id: Current project identifier (e.g., 'project_task_orchestrator')
            subscriptions: Initial set of subscribed project IDs (deprecated with Titan)
            titan_client: Optional Titan client (preferred for search)
        """
        self.client = graphiti_client
        self.local_group_id = local_group_id
        self.subscriptions: Set[str] = subscriptions or set()
        self._pattern_visibility: Dict[str, PatternVisibility] = {}

        # Titan support (Phase 4)
        self._titan_client = titan_client
        self._use_titan = titan_client is not None

        backend = "Titan" if self._use_titan else ("Graphiti" if graphiti_client else "local")
        logger.info(f"PatternFederation initialized for {local_group_id} (backend={backend})")

    async def subscribe_to_project(self, target_group_id: str) -> Dict[str, Any]:
        """
        Subscribe to another project's shared patterns.

        DEPRECATED: When using Titan, subscriptions are no-ops.
        Titan's project filtering handles cross-project access automatically.

        Args:
            target_group_id: The group_id of the project to subscribe to

        Returns:
            Result dictionary with status
        """
        if self._use_titan:
            warnings.warn(
                "federation subscriptions are deprecated with Titan backend. "
                "Use titan_recall without project filter to search across projects.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.info(f"[DEPRECATED] Subscribe to {target_group_id} - Titan handles automatically")
            return {
                "success": True,
                "subscribed_to": target_group_id,
                "deprecated": True,
                "message": "Subscriptions are no-ops with Titan. Cross-project search is automatic.",
            }

        if target_group_id == self.local_group_id:
            return {"success": False, "error": "Cannot subscribe to self"}

        self.subscriptions.add(target_group_id)
        logger.info(f"Subscribed to project: {target_group_id}")

        return {
            "success": True,
            "subscribed_to": target_group_id,
            "total_subscriptions": len(self.subscriptions),
        }

    async def unsubscribe_from_project(self, target_group_id: str) -> Dict[str, Any]:
        """
        Unsubscribe from a project's patterns.

        Args:
            target_group_id: The group_id to unsubscribe from

        Returns:
            Result dictionary
        """
        if target_group_id in self.subscriptions:
            self.subscriptions.discard(target_group_id)
            logger.info(f"Unsubscribed from project: {target_group_id}")
            return {"success": True, "unsubscribed_from": target_group_id}
        return {"success": False, "error": "Not subscribed to this project"}

    async def publish_pattern(
        self,
        pattern_id: str,
        visibility: str = "shared",
    ) -> Dict[str, Any]:
        """
        Update a local pattern's visibility.

        Args:
            pattern_id: ID of the pattern to publish
            visibility: 'shared' or 'private'

        Returns:
            Result dictionary
        """
        try:
            vis = PatternVisibility(visibility)
            self._pattern_visibility[pattern_id] = vis
            logger.info(f"Pattern {pattern_id} visibility set to {visibility}")

            return {
                "success": True,
                "pattern_id": pattern_id,
                "visibility": visibility,
            }
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid visibility: {visibility}. Use 'shared' or 'private'",
            }

    async def search_global_patterns(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ScoredPattern]:
        """
        Search for patterns across all subscribed projects + local.

        When Titan is available, uses titan_recall for cross-project search.
        Falls back to Graphiti when Titan is not available.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of ScoredPattern results, sorted by relevance
        """
        # Use Titan when available (Phase 4)
        if self._use_titan and self._titan_client:
            return await self._search_with_titan(query, limit)

        # Legacy: Graphiti-based search
        results: List[ScoredPattern] = []

        # Search local project first
        local_patterns = await self._search_project(
            self.local_group_id,
            query,
            require_shared=False,
        )
        results.extend(local_patterns)

        # Search subscribed projects
        for remote_group in self.subscriptions:
            remote_patterns = await self._search_project(
                remote_group,
                query,
                require_shared=True,
            )
            results.extend(remote_patterns)

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[:limit]

    async def _search_project(
        self,
        group_id: str,
        query: str,
        require_shared: bool = False,
    ) -> List[ScoredPattern]:
        """
        Search a specific project for patterns.

        Args:
            group_id: The project to search
            query: Search query
            require_shared: Only return shared patterns (for remote projects)

        Returns:
            List of ScoredPattern results
        """
        scored_results: List[ScoredPattern] = []

        if not self.client:
            logger.debug("No Graphiti client available for search")
            return scored_results

        try:
            # Search using Graphiti
            facts = await self.client.search_memory_facts(
                query=query,
                group_ids=[group_id],
                max_facts=20,
            )

            if not facts:
                return scored_results

            for fact in facts:
                # Parse the fact data
                try:
                    import json
                    if hasattr(fact, 'fact'):
                        data = json.loads(fact.fact) if isinstance(fact.fact, str) else fact.fact
                    elif isinstance(fact, dict):
                        data = fact
                    else:
                        continue

                    # Check if this is a failure pattern
                    if data.get("type") != "immune_failure_pattern":
                        continue

                    # Check visibility for remote projects
                    visibility = data.get("visibility", "private")
                    if require_shared and visibility != "shared":
                        continue

                    # Reconstruct the pattern
                    pattern = FailurePattern(
                        id=data.get("pattern_id", ""),
                        operation=data.get("operation", "unknown"),
                        failure_type=data.get("failure_type", "unknown"),
                        input_summary=data.get("input_summary", ""),
                        output_summary=data.get("output_summary", ""),
                        grader_scores=data.get("grader_scores", {}),
                        occurrence_count=data.get("occurrence_count", 1),
                        created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
                        context=data.get("context", {}),
                    )

                    # Calculate relevance score
                    score = self._calculate_relevance(
                        pattern,
                        query,
                        is_local=(group_id == self.local_group_id),
                    )

                    scored_results.append(ScoredPattern(
                        pattern=pattern,
                        relevance_score=score,
                        source_project=group_id,
                        match_reason=f"Matched query '{query}' in {group_id}",
                    ))

                except Exception as e:
                    logger.warning(f"Failed to parse pattern from fact: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Search failed for group {group_id}: {e}")

        return scored_results

    async def _search_with_titan(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ScoredPattern]:
        """
        Search for patterns using Titan Memory (Phase 4).

        Searches across all projects using Titan's native search without
        project filtering, which automatically includes all accessible patterns.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of ScoredPattern results
        """
        import json

        scored_results: List[ScoredPattern] = []

        if not self._titan_client:
            return scored_results

        try:
            # Search Titan without project filter = cross-project search
            # Tags filter for failure patterns
            memories = await self._titan_client.search_memories(
                query=f"{query} failure pattern",
                tags=["failure", "immune"],
                limit=limit * 2,  # Get more to filter
            )

            for memory in memories:
                try:
                    content = memory.get("content", "")
                    if not content:
                        continue

                    # Try to parse as JSON (failure pattern format)
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # Not a JSON pattern, skip
                        continue

                    if data.get("type") != "failure_pattern":
                        continue

                    # Reconstruct pattern from Titan data
                    pattern = FailurePattern(
                        id=data.get("id", memory.get("id", "")),
                        operation=data.get("operation", "unknown"),
                        failure_type=data.get("failure_type", "unknown"),
                        input_summary=data.get("input_summary", ""),
                        output_summary=data.get("output_summary", ""),
                        grader_scores=data.get("grader_scores", {}),
                        occurrence_count=data.get("occurrence_count", 1),
                        context=data.get("context", {}),
                    )

                    # Use Titan's score if available, otherwise calculate
                    score = memory.get("score", 0.5)

                    scored_results.append(ScoredPattern(
                        pattern=pattern,
                        relevance_score=score,
                        source_project=data.get("context", {}).get("project", "titan"),
                        match_reason=f"Matched via Titan search: {query}",
                    ))

                except Exception as e:
                    logger.debug(f"Failed to parse Titan memory: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Titan search failed: {e}")

        # Sort by relevance score (descending)
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return scored_results[:limit]

    def _calculate_relevance(
        self,
        pattern: FailurePattern,
        query: str,
        is_local: bool,
    ) -> float:
        """
        Calculate relevance score for a pattern.

        Scoring factors:
        - Base similarity score (from Graphiti)
        - Success rate weighting
        - Local preference
        - Usage/maturity

        Args:
            pattern: The pattern to score
            query: The search query
            is_local: Whether this is from the local project

        Returns:
            Relevance score (0.0-1.0)
        """
        score = 0.0

        # Base score (simulated - in production this comes from vector similarity)
        score += 0.5

        # Local preference bonus
        if is_local:
            score += 0.15

        # Maturity/usage bonus
        if pattern.occurrence_count > 5:
            score += 0.1
        if pattern.occurrence_count > 10:
            score += 0.1

        # Text match bonus (simple keyword matching)
        query_lower = query.lower()
        if query_lower in pattern.input_summary.lower():
            score += 0.1
        if query_lower in pattern.failure_type.lower():
            score += 0.05

        return min(1.0, round(score, 3))

    async def import_pattern(
        self,
        remote_pattern_id: str,
        source_group_id: str,
    ) -> Dict[str, Any]:
        """
        Import a pattern from a remote project.

        Creates a local copy with lineage tracking.

        Args:
            remote_pattern_id: ID of the pattern to import
            source_group_id: The source project's group_id

        Returns:
            Result dictionary
        """
        if source_group_id == self.local_group_id:
            return {
                "success": False,
                "error": "Pattern is already local",
            }

        if not self.client:
            return {
                "success": False,
                "error": "No Graphiti client available",
            }

        try:
            # Search for the specific pattern
            facts = await self.client.search_memory_facts(
                query=f"pattern_id:{remote_pattern_id}",
                group_ids=[source_group_id],
                max_facts=1,
            )

            if not facts:
                return {
                    "success": False,
                    "error": f"Pattern {remote_pattern_id} not found in {source_group_id}",
                }

            # Parse the pattern data
            import json
            fact = facts[0]
            if hasattr(fact, 'fact'):
                data = json.loads(fact.fact) if isinstance(fact.fact, str) else fact.fact
            elif isinstance(fact, dict):
                data = fact
            else:
                return {"success": False, "error": "Invalid pattern format"}

            # Create local copy with lineage
            import uuid
            new_id = str(uuid.uuid4())[:16]

            local_pattern_data = {
                "type": "immune_failure_pattern",
                "pattern_id": new_id,
                "operation": data.get("operation", "unknown"),
                "failure_type": data.get("failure_type", "unknown"),
                "input_summary": data.get("input_summary", ""),
                "output_summary": data.get("output_summary", ""),
                "grader_scores": data.get("grader_scores", {}),
                "occurrence_count": 1,  # Reset count for local copy
                "created_at": datetime.utcnow().isoformat(),
                "context": data.get("context", {}),
                # Lineage tracking
                "original_source_project": data.get("original_source_project") or source_group_id,
                "original_pattern_id": data.get("original_pattern_id") or remote_pattern_id,
                "derived_from_id": remote_pattern_id,
                "visibility": "private",  # Import as private by default
            }

            # Store in local Graphiti
            await self.client.add_memory(
                name=f"failure_pattern_{new_id}",
                episode_body=json.dumps(local_pattern_data),
                group_id=self.local_group_id,
                source="json",
                source_description="immune_failure_pattern_imported",
            )

            logger.info(f"Imported pattern {remote_pattern_id} from {source_group_id} as {new_id}")

            return {
                "success": True,
                "local_pattern_id": new_id,
                "source_project": source_group_id,
                "original_pattern_id": remote_pattern_id,
            }

        except Exception as e:
            logger.error("Failed to import pattern", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_subscriptions(self) -> List[str]:
        """Get list of subscribed projects."""
        return list(self.subscriptions)

    def get_stats(self) -> Dict[str, Any]:
        """Get federation statistics."""
        return {
            "local_group_id": self.local_group_id,
            "subscriptions_count": len(self.subscriptions),
            "subscribed_to": list(self.subscriptions),
            "published_patterns": len([
                v for v in self._pattern_visibility.values()
                if v == PatternVisibility.SHARED
            ]),
        }


__all__ = [
    "PatternFederation",
    "PatternVisibility",
    "FederatedPattern",
    "ScoredPattern",
]
