"""
Cross-Project Learning Integration for Task Orchestrator.

Enables pattern sharing and learning across projects using:
1. Local pattern store (file-based, fast)
2. Graphiti knowledge graph (MCP-based, semantic search)

Usage:
    from .cross_project import CrossProjectLearning

    learning = CrossProjectLearning()

    # Query patterns
    patterns = await learning.query_patterns("circuit breaker resilience")

    # Record pattern usage
    await learning.record_usage(pattern_id, success=True)

    # Extract and store new pattern
    await learning.extract_pattern(
        name="New Pattern",
        description="Description",
        content="Pattern content",
        pattern_type="error_fix",
        tags=["tag1", "tag2"]
    )
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import local pattern store
import sys
sys.path.insert(0, str(Path.home() / ".claude" / "grade5" / "cross-project"))

try:
    from pattern_store import PatternStore, PatternType, Pattern
    _pattern_store_available = True
except ImportError:
    _pattern_store_available = False
    PatternStore = None
    PatternType = None
    Pattern = None


@dataclass
class PatternResult:
    """A pattern result from either local store or Graphiti."""
    id: str
    name: str
    description: str
    content: str
    source: str  # "local" or "graphiti"
    pattern_type: str
    tags: List[str]
    validity_score: float
    group_id: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "source": self.source,
            "pattern_type": self.pattern_type,
            "tags": self.tags,
            "validity_score": self.validity_score,
            "group_id": self.group_id,
            "metadata": self.metadata,
        }


class CrossProjectLearning:
    """
    Cross-project learning system.

    Combines local pattern store with Graphiti knowledge graph
    for comprehensive pattern management.
    """

    def __init__(self, project_id: str = "task-orchestrator"):
        self.project_id = project_id
        self.group_id = f"project_{project_id.replace('-', '_')}"

        # Load namespace configuration
        self.namespaces = self._load_namespaces()

        # Initialize local pattern store
        self._pattern_store = None  # type: ignore[assignment]
        if _pattern_store_available and PatternStore is not None:
            try:
                self._pattern_store = PatternStore()
            except Exception:
                pass

    def _load_namespaces(self) -> Dict[str, Any]:
        """Load project namespace configuration."""
        namespace_path = Path.home() / ".claude" / "grade5" / "namespaces.json"
        if namespace_path.exists():
            try:
                with open(namespace_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"namespaces": [], "global_group_id": "global"}

    def _get_related_group_ids(self) -> List[str]:
        """Get group IDs for this project and related projects."""
        group_ids = [self.group_id, "global"]

        for ns in self.namespaces.get("namespaces", []):
            if ns.get("project_id") == self.project_id:
                # Add related projects
                for related in ns.get("related_projects", []):
                    related_group = f"project_{related.replace('-', '_')}"
                    if related_group not in group_ids:
                        group_ids.append(related_group)
                break

        return group_ids

    async def query_patterns(
        self,
        query: str,
        pattern_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        include_global: bool = True,
        limit: int = 10,
    ) -> List[PatternResult]:
        """
        Query patterns from both local store and Graphiti.

        Args:
            query: Search query
            pattern_type: Filter by pattern type
            tags: Filter by tags
            include_global: Include global patterns
            limit: Maximum results

        Returns:
            List of matching patterns
        """
        results = []
        group_ids = self._get_related_group_ids() if include_global else [self.group_id]

        # Query local pattern store first (fast)
        if self._pattern_store:
            try:
                p_type = None
                if pattern_type and PatternType is not None and hasattr(PatternType, pattern_type.upper()):
                    p_type = PatternType[pattern_type.upper()]  # type: ignore[index]

                local_patterns = self._pattern_store.query(
                    group_ids=group_ids,
                    pattern_type=p_type,
                    tags=tags,
                    limit=limit,
                )

                for p in local_patterns:
                    results.append(PatternResult(
                        id=p.id,
                        name=p.name,
                        description=p.description,
                        content=p.content,
                        source="local",
                        pattern_type=p.pattern_type.value if hasattr(p.pattern_type, 'value') else str(p.pattern_type),
                        tags=p.tags,
                        validity_score=p.validity_score,
                        group_id=p.group_id,
                        metadata=p.metadata,
                    ))
            except Exception:
                # Log error but continue
                pass

        # Note: Graphiti queries would be done through MCP calls
        # The caller can use mcp__graphiti__search_memory_facts directly

        # Sort by validity score
        results.sort(key=lambda r: r.validity_score, reverse=True)

        return results[:limit]

    async def record_usage(
        self,
        pattern_id: str,
        success: bool,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Record usage of a pattern (success or failure).

        This updates the pattern's validity score based on actual usage.
        """
        if not self._pattern_store:
            return False

        try:
            if success:
                return self._pattern_store.record_success(pattern_id)
            else:
                return self._pattern_store.record_failure(pattern_id)
        except Exception:
            return False

    async def extract_pattern(
        self,
        name: str,
        description: str,
        content: str,
        pattern_type: str = "code_pattern",
        tags: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        scope: str = "project",  # "project", "shared", or "global"
    ) -> Optional[PatternResult]:
        """
        Extract and store a new pattern.

        Args:
            name: Pattern name
            description: What this pattern does
            content: The actual pattern (code, config, etc.)
            pattern_type: Type of pattern
            tags: Tags for categorization
            examples: Usage examples
            scope: "project" (this project only), "shared" (related projects), or "global"

        Returns:
            The created pattern, or None if failed
        """
        if not self._pattern_store:
            return None

        try:
            # Determine group_id based on scope
            if scope == "global":
                group_id = "global"
            else:
                group_id = self.group_id

            # Map string type to enum
            p_type = PatternType.CODE_PATTERN if PatternType is not None else None  # type: ignore[union-attr]
            if PatternType is not None and hasattr(PatternType, pattern_type.upper()):
                p_type = PatternType[pattern_type.upper()]  # type: ignore[index]

            pattern = self._pattern_store.add_pattern(
                pattern_type=p_type,
                name=name,
                description=description,
                content=content,
                group_id=group_id,
                tags=tags,
                examples=examples,
                metadata={
                    "source_project": self.project_id,
                    "scope": scope,
                    "extracted_at": datetime.utcnow().isoformat(),
                },
            )

            return PatternResult(
                id=pattern.id,
                name=pattern.name,
                description=pattern.description,
                content=pattern.content,
                source="local",
                pattern_type=pattern.pattern_type.value,
                tags=pattern.tags,
                validity_score=pattern.validity_score,
                group_id=pattern.group_id,
                metadata=pattern.metadata,
            )
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get cross-project learning statistics."""
        stats = {
            "project_id": self.project_id,
            "group_id": self.group_id,
            "related_groups": self._get_related_group_ids(),
            "local_store": None,
        }

        if self._pattern_store:
            try:
                stats["local_store"] = self._pattern_store.get_stats()
            except Exception:
                pass

        return stats


# Convenience functions for direct usage
_learning_instance: Optional[CrossProjectLearning] = None


def get_learning(project_id: str = "task-orchestrator") -> CrossProjectLearning:
    """Get or create the cross-project learning instance."""
    global _learning_instance
    if _learning_instance is None or _learning_instance.project_id != project_id:
        _learning_instance = CrossProjectLearning(project_id)
    return _learning_instance
