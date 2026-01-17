"""
Tests for Federation System (Phase 9).

Tests the portfolio registry, pattern decay, and cross-project federation.
"""

import pytest
from datetime import datetime, timezone, timedelta

# Import federation components
from src.evaluation.immune_system.registry import (
    PortfolioProject,
    RegistryManager,
    get_registry_manager,
    reset_registry_manager,
)
from src.evaluation.immune_system.decay import (
    PatternDecaySystem,
    DecayMetadata,
    InteractionOutcome,
    get_decay_system,
    reset_decay_system,
    DEFAULT_HALF_LIFE_HOURS,
    STALENESS_THRESHOLD_DAYS,
)
from src.evaluation.immune_system.federation import (
    PatternFederation,
)


# ============================================================================
# Portfolio Registry Tests
# ============================================================================

class TestPortfolioProject:
    """Tests for PortfolioProject dataclass."""

    def test_create_project(self):
        """Test creating a portfolio project."""
        project = PortfolioProject(
            project_id="test-project",
            group_id="project_test",
            description="Test project",
        )
        assert project.project_id == "test-project"
        assert project.group_id == "project_test"
        assert project.version == "0.1.0"
        assert project.health_status == "healthy"
        assert project.sync_frequency_seconds == 300

    def test_to_dict(self):
        """Test serialization to dict."""
        project = PortfolioProject(
            project_id="test",
            group_id="group_test",
            description="Test",
            last_sync=datetime(2026, 1, 12, 10, 0, 0, tzinfo=timezone.utc),
        )
        d = project.to_dict()
        assert d["project_id"] == "test"
        assert "2026-01-12" in d["last_sync"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "project_id": "restored",
            "group_id": "project_restored",
            "description": "Restored project",
            "version": "1.0.0",
            "last_sync": "2026-01-12T10:00:00+00:00",
        }
        project = PortfolioProject.from_dict(data)
        assert project.project_id == "restored"
        assert project.version == "1.0.0"
        assert project.last_sync is not None

    def test_needs_sync_no_sync(self):
        """Test needs_sync when never synced."""
        project = PortfolioProject(
            project_id="test",
            group_id="test",
            description="Test",
            last_sync=None,
        )
        assert project.needs_sync is True

    def test_needs_sync_recent(self):
        """Test needs_sync when recently synced."""
        project = PortfolioProject(
            project_id="test",
            group_id="test",
            description="Test",
            last_sync=datetime.now(timezone.utc),
            sync_frequency_seconds=300,
        )
        assert project.needs_sync is False

    def test_needs_sync_stale(self):
        """Test needs_sync when sync is overdue."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=400)
        project = PortfolioProject(
            project_id="test",
            group_id="test",
            description="Test",
            last_sync=old_time,
            sync_frequency_seconds=300,
        )
        assert project.needs_sync is True


class TestRegistryManager:
    """Tests for RegistryManager."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_registry_manager()

    def test_init_defaults(self):
        """Test initialization with default projects."""
        manager = RegistryManager()
        assert len(manager.projects) >= 1
        assert "task-orchestrator" in manager.projects

    @pytest.mark.asyncio
    async def test_get_project(self):
        """Test getting a project by ID."""
        manager = RegistryManager()
        project = await manager.get_project("task-orchestrator")
        assert project is not None
        assert project.group_id == "project_task_orchestrator"

    @pytest.mark.asyncio
    async def test_register_project(self):
        """Test registering a new project."""
        manager = RegistryManager()
        new_project = PortfolioProject(
            project_id="new-project",
            group_id="project_new",
            description="New test project",
        )
        result = await manager.register_project(new_project)
        assert result["success"] is True
        assert "new-project" in manager.projects

    def test_get_projects_needing_sync(self):
        """Test getting projects that need sync."""
        manager = RegistryManager()
        # All projects should need sync initially (no last_sync)
        needing_sync = manager.get_projects_needing_sync()
        assert len(needing_sync) > 0

    def test_get_stats(self):
        """Test getting registry statistics."""
        manager = RegistryManager()
        stats = manager.get_stats()
        assert "total_projects" in stats
        assert stats["total_projects"] >= 1

    def test_singleton_pattern(self):
        """Test that get_registry_manager returns singleton."""
        manager1 = get_registry_manager()
        manager2 = get_registry_manager()
        assert manager1 is manager2


# ============================================================================
# Pattern Decay Tests
# ============================================================================

class TestDecayMetadata:
    """Tests for DecayMetadata dataclass."""

    def test_create_default(self):
        """Test creating default metadata."""
        meta = DecayMetadata()
        assert meta.relevance_score == 0.5
        assert meta.usage_count == 0
        assert meta.is_stale is False

    def test_to_dict(self):
        """Test serialization."""
        meta = DecayMetadata(relevance_score=0.8, usage_count=5)
        d = meta.to_dict()
        assert d["relevance_score"] == 0.8
        assert d["usage_count"] == 5
        assert "last_updated" in d

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "relevance_score": 0.7,
            "usage_count": 10,
            "last_updated": "2026-01-12T10:00:00+00:00",
        }
        meta = DecayMetadata.from_dict(data)
        assert meta.relevance_score == 0.7
        assert meta.usage_count == 10


class TestPatternDecaySystem:
    """Tests for PatternDecaySystem."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_decay_system()

    def test_init_defaults(self):
        """Test initialization with default values."""
        decay = PatternDecaySystem()
        assert decay.half_life_hours == DEFAULT_HALF_LIFE_HOURS
        assert decay.staleness_days == STALENESS_THRESHOLD_DAYS

    def test_calculate_decay_recent(self):
        """Test that recent timestamps show minimal decay."""
        decay = PatternDecaySystem(half_life_hours=24.0)
        now = datetime.now(timezone.utc)
        score = decay._calculate_decay(1.0, now)
        assert score > 0.99  # Almost no decay

    def test_calculate_decay_half_life(self):
        """Test decay at exactly one half-life."""
        decay = PatternDecaySystem(half_life_hours=24.0)
        old_time = datetime.now(timezone.utc) - timedelta(hours=24)
        score = decay._calculate_decay(1.0, old_time)
        assert 0.49 < score < 0.51  # Should be ~0.5

    def test_calculate_decay_two_half_lives(self):
        """Test decay at two half-lives."""
        decay = PatternDecaySystem(half_life_hours=24.0)
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        score = decay._calculate_decay(1.0, old_time)
        assert 0.24 < score < 0.26  # Should be ~0.25

    def test_get_current_relevance_default(self):
        """Test getting relevance for unknown pattern."""
        decay = PatternDecaySystem()
        score = decay.get_current_relevance("unknown-pattern")
        assert score == 0.5  # Default initial score

    def test_register_interaction_boost(self):
        """Test that successful interactions boost score."""
        decay = PatternDecaySystem()
        meta = decay.register_interaction(
            "test-pattern",
            InteractionOutcome.CRITICAL_SUCCESS,
        )
        assert meta.relevance_score > 0.5
        assert meta.usage_count == 1
        assert meta.success_count == 1

    def test_register_interaction_penalty(self):
        """Test that failures penalize score."""
        decay = PatternDecaySystem()
        # First register to get a baseline
        decay.register_interaction("test", InteractionOutcome.STANDARD_MATCH)
        # Then apply penalty
        meta = decay.register_interaction("test", InteractionOutcome.FAILURE_PENALTY)
        assert meta.failure_count == 1
        assert meta.relevance_score < 0.55  # Should be reduced

    def test_check_staleness_fresh(self):
        """Test fresh pattern is not stale."""
        decay = PatternDecaySystem(staleness_days=14)
        # Register interaction to set timestamp
        decay.register_interaction("test", InteractionOutcome.STANDARD_MATCH)
        is_stale = decay.check_staleness("test")
        assert is_stale is False

    def test_check_staleness_old(self):
        """Test old pattern is stale."""
        decay = PatternDecaySystem(staleness_days=14)
        old_meta = {
            "relevance_score": 0.5,
            "last_updated": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
        }
        is_stale = decay.check_staleness("old-pattern", old_meta)
        assert is_stale is True

    def test_should_prune(self):
        """Test pruning logic."""
        decay = PatternDecaySystem(staleness_days=14, min_score=0.1)
        old_meta = {
            "relevance_score": 0.05,  # Below threshold
            "last_updated": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
        }
        should_prune = decay.should_prune("old-pattern", old_meta)
        assert should_prune is True

    def test_batch_evaluate(self):
        """Test batch evaluation of patterns."""
        decay = PatternDecaySystem()
        patterns = [
            {"id": "p1", "decay_metadata": {"relevance_score": 0.8}},
            {"id": "p2", "decay_metadata": {"relevance_score": 0.3}},
        ]
        result = decay.batch_evaluate(patterns)
        assert result["evaluated"] == 2
        assert "scores" in result

    def test_get_stats(self):
        """Test getting system statistics."""
        decay = PatternDecaySystem()
        stats = decay.get_stats()
        assert "half_life_hours" in stats
        assert "cached_patterns" in stats

    def test_singleton_pattern(self):
        """Test that get_decay_system returns singleton."""
        decay1 = get_decay_system()
        decay2 = get_decay_system()
        assert decay1 is decay2


# ============================================================================
# Pattern Federation Tests
# ============================================================================

class TestPatternFederation:
    """Tests for PatternFederation."""

    def test_init(self):
        """Test initialization."""
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )
        assert fed.local_group_id == "project_test"
        assert len(fed.subscriptions) == 0

    @pytest.mark.asyncio
    async def test_subscribe_to_project(self):
        """Test subscribing to another project."""
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_local",
        )
        result = await fed.subscribe_to_project("project_remote")
        assert result["success"] is True
        assert "project_remote" in fed.subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_to_self_fails(self):
        """Test that subscribing to self fails."""
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )
        result = await fed.subscribe_to_project("project_test")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from a project."""
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
            subscriptions={"project_remote"},
        )
        result = await fed.unsubscribe_from_project("project_remote")
        assert result["success"] is True
        assert "project_remote" not in fed.subscriptions

    @pytest.mark.asyncio
    async def test_publish_pattern(self):
        """Test publishing a pattern."""
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )
        result = await fed.publish_pattern("pattern-1", visibility="shared")
        assert result["success"] is True
        assert result["visibility"] == "shared"

    @pytest.mark.asyncio
    async def test_search_global_patterns_empty(self):
        """Test searching with no Graphiti client."""
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )
        results = await fed.search_global_patterns("test query")
        assert results == []

    def test_get_subscriptions(self):
        """Test getting subscription list."""
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
            subscriptions={"project_a", "project_b"},
        )
        subs = fed.get_subscriptions()
        assert len(subs) == 2
        assert "project_a" in subs

    def test_get_stats(self):
        """Test getting federation statistics."""
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )
        stats = fed.get_stats()
        assert stats["local_group_id"] == "project_test"
        assert "subscriptions_count" in stats


# ============================================================================
# Integration Tests
# ============================================================================

class TestFederationIntegration:
    """Integration tests for the full federation system."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_registry_manager()
        reset_decay_system()

    @pytest.mark.asyncio
    async def test_registry_federation_integration(self):
        """Test registry and federation working together."""
        # Create registry
        registry = get_registry_manager()

        # Create federation
        fed = PatternFederation(
            graphiti_client=None,
            local_group_id="project_task_orchestrator",
        )

        # Get a project from registry
        project = await registry.get_project("task-orchestrator")
        assert project is not None

        # Subscribe to another project's patterns
        cc_project = await registry.get_project("construction-connect")
        if cc_project:
            result = await fed.subscribe_to_project(cc_project.group_id)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_decay_with_patterns(self):
        """Test decay system with pattern evaluation."""
        decay = get_decay_system()

        # Simulate pattern interactions
        decay.register_interaction("pattern-1", InteractionOutcome.CRITICAL_SUCCESS)
        decay.register_interaction("pattern-2", InteractionOutcome.FAILURE_PENALTY)

        # Verify scores
        score1 = decay.get_current_relevance("pattern-1")
        score2 = decay.get_current_relevance("pattern-2")

        assert score1 > score2  # Success should have higher score

    def test_interaction_outcomes(self):
        """Test all interaction outcome values."""
        assert InteractionOutcome.CRITICAL_SUCCESS.value == 0.20
        assert InteractionOutcome.STANDARD_MATCH.value == 0.05
        assert InteractionOutcome.PARTIAL_MATCH.value == 0.01
        assert InteractionOutcome.FAILURE_PENALTY.value == -0.15
        assert InteractionOutcome.CRITICAL_FAILURE.value == -0.50
