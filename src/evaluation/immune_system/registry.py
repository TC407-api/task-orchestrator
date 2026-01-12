"""
Portfolio Project Registry for Cross-Project Federation.

Manages portfolio project discovery and federation state using
a hybrid approach: static namespaces.json + dynamic Graphiti discovery.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

logger = logging.getLogger(__name__)

# Type alias for health status
HealthStatus = Literal["healthy", "degraded", "maintenance", "offline"]

REGISTRY_GROUP_ID = "portfolio-federation-registry"


@dataclass
class PortfolioProject:
    """
    Represents a single project within the federation.

    Attributes:
        project_id: Unique slug for the project (e.g., 'task-orchestrator')
        group_id: Graphiti group_id for knowledge isolation
        description: Human-readable description
        version: Project schema version
        health_status: Current health state
        sync_frequency_seconds: How often to sync (default 5 minutes)
        patterns_shared: Types of patterns this project shares
        subscriptions: List of project_ids this project consumes from
        last_sync: Timestamp of last successful sync
    """
    project_id: str
    group_id: str
    description: str
    version: str = "0.1.0"
    health_status: HealthStatus = "healthy"
    sync_frequency_seconds: int = 300
    patterns_shared: List[str] = field(default_factory=list)
    subscriptions: List[str] = field(default_factory=list)
    last_sync: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, handling datetime objects."""
        data = asdict(self)
        if self.last_sync:
            data['last_sync'] = self.last_sync.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioProject':
        """Factory to create instance from dictionary/JSON."""
        # Handle datetime deserialization
        if data.get('last_sync'):
            try:
                if isinstance(data['last_sync'], str):
                    data['last_sync'] = datetime.fromisoformat(data['last_sync'])
            except ValueError:
                data['last_sync'] = None

        # Filter to known fields only
        known_fields = {
            'project_id', 'group_id', 'description', 'version',
            'health_status', 'sync_frequency_seconds', 'patterns_shared',
            'subscriptions', 'last_sync'
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    @property
    def needs_sync(self) -> bool:
        """Check if sync is required based on frequency."""
        if not self.last_sync:
            return True
        delta = (datetime.now(timezone.utc) - self.last_sync).total_seconds()
        return delta > self.sync_frequency_seconds


@dataclass
class RegistryConfig:
    """Root container for the namespaces configuration."""
    registry_version: str
    projects: List[PortfolioProject] = field(default_factory=list)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'RegistryConfig':
        """Load registry from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            registry_version=data.get('registry_version', '0.0.0'),
            projects=[PortfolioProject.from_dict(p) for p in data.get('projects', [])]
        )

    def save_to_file(self, filepath: str) -> None:
        """Save registry to JSON file."""
        data = {
            'registry_version': self.registry_version,
            'projects': [p.to_dict() for p in self.projects]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class RegistryManager:
    """
    Manages project discovery and federation state.

    Acts as the source of truth for which Graphiti group_ids belong
    to which logical projects.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        graphiti_client: Optional[Any] = None,
    ):
        """
        Initialize the registry manager.

        Args:
            config_path: Path to namespaces.json file
            graphiti_client: Optional Graphiti client for persistence
        """
        self.config_path = config_path
        self.client = graphiti_client
        self.projects: Dict[str, PortfolioProject] = {}

        # Load local config if provided
        if config_path and Path(config_path).exists():
            self._load_local_config()
        else:
            # Initialize with defaults
            self._init_defaults()

    def _init_defaults(self) -> None:
        """Initialize with default portfolio projects."""
        defaults = [
            PortfolioProject(
                project_id="task-orchestrator",
                group_id="project_task_orchestrator",
                description="MCP server for AI agent orchestration",
                patterns_shared=["failure_patterns", "guardrails", "evaluations"],
            ),
            PortfolioProject(
                project_id="construction-connect",
                group_id="project_construction_connect",
                description="Construction workforce management platform",
                patterns_shared=["api_patterns", "business_logic"],
            ),
        ]
        for p in defaults:
            self.projects[p.project_id] = p
        logger.info(f"Initialized with {len(self.projects)} default projects")

    def _load_local_config(self) -> None:
        """Loads the static namespaces.json file."""
        try:
            config = RegistryConfig.load_from_file(self.config_path)
            for p in config.projects:
                self.projects[p.project_id] = p
            logger.info(f"Loaded {len(self.projects)} projects from local config")
        except FileNotFoundError:
            logger.warning("namespaces.json not found, using defaults")
            self._init_defaults()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in namespaces.json: {e}")
            self._init_defaults()

    async def get_project(self, project_id: str) -> Optional[PortfolioProject]:
        """Retrieve a project by ID."""
        return self.projects.get(project_id)

    async def register_project(self, project: PortfolioProject) -> Dict[str, Any]:
        """
        Register a project both in memory and optionally in Graphiti.

        Args:
            project: The project to register

        Returns:
            Result dictionary with success status
        """
        self.projects[project.project_id] = project
        logger.info(f"Registered project {project.project_id}")

        # Persist to Graphiti if available
        if self.client:
            try:
                await self.client.add_memory(
                    name=f"project_registry_{project.project_id}",
                    episode_body=json.dumps({
                        "type": "project_metadata",
                        **project.to_dict()
                    }),
                    group_id=REGISTRY_GROUP_ID,
                    source="json",
                    source_description="portfolio_project_registration",
                )
                logger.info(f"Persisted {project.project_id} to Graphiti")
            except Exception as e:
                logger.error(f"Failed to persist to Graphiti: {e}")

        return {
            "success": True,
            "project_id": project.project_id,
            "group_id": project.group_id,
        }

    async def auto_discover(self) -> Dict[str, Any]:
        """
        Discover projects from Graphiti registry group.

        Returns:
            Dict with discovery statistics
        """
        if not self.client:
            logger.debug("Auto-discovery skipped: no Graphiti client")
            return {"discovered": 0, "skipped": True}

        logger.info("Starting Graphiti auto-discovery...")
        discovered_count = 0

        try:
            # Search for project metadata in registry group
            results = await self.client.search_memory_facts(
                query="project_metadata",
                group_ids=[REGISTRY_GROUP_ID],
                max_facts=50,
            )

            if not results:
                return {"discovered": 0}

            for fact in results:
                try:
                    # Parse the fact data
                    if hasattr(fact, 'fact'):
                        data = json.loads(fact.fact) if isinstance(fact.fact, str) else fact.fact
                    elif isinstance(fact, dict):
                        data = fact
                    else:
                        continue

                    # Validate it's project metadata
                    if data.get("type") != "project_metadata":
                        continue

                    # Reconstruct project
                    project = PortfolioProject.from_dict(data)
                    self.projects[project.project_id] = project
                    discovered_count += 1

                except Exception as e:
                    logger.warning(f"Failed to parse project from fact: {e}")
                    continue

            logger.info(f"Auto-discovery complete. Found {discovered_count} projects")
            return {"discovered": discovered_count, "total": len(self.projects)}

        except Exception as e:
            logger.error(f"Auto-discovery failed: {e}")
            return {"discovered": 0, "error": str(e)}

    async def update_sync_status(self, project_id: str) -> None:
        """Update the last_sync timestamp for a project."""
        if project_id in self.projects:
            self.projects[project_id].last_sync = datetime.now(timezone.utc)
            # Re-persist the update
            await self.register_project(self.projects[project_id])

    def get_projects_needing_sync(self) -> List[PortfolioProject]:
        """Get all projects that need synchronization."""
        return [p for p in self.projects.values() if p.needs_sync]

    def get_subscribed_projects(self, project_id: str) -> List[PortfolioProject]:
        """Get all projects that a given project is subscribed to."""
        project = self.projects.get(project_id)
        if not project:
            return []

        return [
            self.projects[sub_id]
            for sub_id in project.subscriptions
            if sub_id in self.projects
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_projects": len(self.projects),
            "projects": list(self.projects.keys()),
            "needing_sync": len(self.get_projects_needing_sync()),
            "healthy_count": sum(
                1 for p in self.projects.values()
                if p.health_status == "healthy"
            ),
        }


# Module-level singleton
_registry_manager: Optional[RegistryManager] = None


def get_registry_manager(
    config_path: Optional[str] = None,
    graphiti_client: Optional[Any] = None,
) -> RegistryManager:
    """Get or create the global registry manager instance."""
    global _registry_manager
    if _registry_manager is None:
        _registry_manager = RegistryManager(config_path, graphiti_client)
    return _registry_manager


def reset_registry_manager() -> None:
    """Reset the global registry manager (for testing)."""
    global _registry_manager
    _registry_manager = None


__all__ = [
    "PortfolioProject",
    "RegistryConfig",
    "RegistryManager",
    "get_registry_manager",
    "reset_registry_manager",
    "REGISTRY_GROUP_ID",
]
