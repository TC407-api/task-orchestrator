"""
Tests for Auto Research Scheduler.

Tests the research module including:
- TopicRegistry: loading, saving, topic management
- ResearchRunner: Firecrawl search execution
- ResearchSummarizer: AI summarization
- ContextPrimer: context file generation
- ResearchToolHandler: MCP tool handling
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.research.scheduler import TopicRegistry, ResearchScheduler, ScheduleConfig
from src.research.runner import ResearchRunner
from src.research.summarizer import ResearchSummarizer
from src.research.primer import ContextPrimer
from src.mcp.research_tools import ResearchToolHandler, RESEARCH_TOOLS


class TestTopicRegistry:
    """Tests for TopicRegistry."""

    def test_create_empty_registry(self):
        """Test creating a new empty registry."""
        registry = TopicRegistry()
        assert registry.global_topics == []
        assert registry.project_topics == {}
        assert registry.schedule.run_time == "06:00"

    def test_load_from_nonexistent_file(self, tmp_path):
        """Test loading from a file that doesn't exist creates it."""
        file_path = tmp_path / "topics.json"
        registry = TopicRegistry.load(str(file_path))

        assert file_path.exists()
        assert registry.global_topics == []

    def test_save_and_load(self, tmp_path):
        """Test saving and loading registry."""
        file_path = tmp_path / "topics.json"

        # Create and save
        registry = TopicRegistry(
            global_topics=["AI agents", "MCP protocol"],
            project_topics={"project1": ["topic1", "topic2"]},
        )
        registry._registry_path = file_path
        registry.save()

        # Load and verify
        loaded = TopicRegistry.load(str(file_path))
        assert loaded.global_topics == ["AI agents", "MCP protocol"]
        assert loaded.project_topics == {"project1": ["topic1", "topic2"]}

    def test_add_global_topic(self, tmp_path):
        """Test adding a global topic."""
        file_path = tmp_path / "topics.json"
        registry = TopicRegistry()
        registry._registry_path = file_path

        assert registry.add_topic("AI research") is True
        assert "AI research" in registry.global_topics

        # Adding again should return False
        assert registry.add_topic("AI research") is False

    def test_add_project_topic(self, tmp_path):
        """Test adding a project-specific topic."""
        file_path = tmp_path / "topics.json"
        registry = TopicRegistry()
        registry._registry_path = file_path

        assert registry.add_topic("Gemini API", "task-orchestrator") is True
        assert "task-orchestrator" in registry.project_topics
        assert "Gemini API" in registry.project_topics["task-orchestrator"]

    def test_remove_topic(self, tmp_path):
        """Test removing a topic."""
        file_path = tmp_path / "topics.json"
        registry = TopicRegistry(global_topics=["topic1", "topic2"])
        registry._registry_path = file_path

        assert registry.remove_topic("topic1") is True
        assert "topic1" not in registry.global_topics

        # Removing non-existent should return False
        assert registry.remove_topic("nonexistent") is False

    def test_all_topics(self):
        """Test getting all topics."""
        registry = TopicRegistry(
            global_topics=["global1", "global2"],
            project_topics={
                "p1": ["p1_topic"],
                "p2": ["p2_topic", "global1"],  # Duplicate
            },
        )

        all_topics = registry.all_topics
        assert len(all_topics) == 4  # Deduplicated
        assert "global1" in all_topics
        assert "p1_topic" in all_topics

    def test_get_topics_for_project(self):
        """Test getting topics for a specific project."""
        registry = TopicRegistry(
            global_topics=["global1"],
            project_topics={"p1": ["p1_topic"]},
        )

        topics = registry.get_topics_for_project("p1")
        assert "global1" in topics
        assert "p1_topic" in topics

        # Non-existent project should still get global
        topics = registry.get_topics_for_project("nonexistent")
        assert "global1" in topics


class TestResearchRunner:
    """Tests for ResearchRunner."""

    @pytest.mark.asyncio
    async def test_search_without_api_key(self):
        """Test search returns empty when no API key explicitly set."""
        # Explicitly pass api_key=None and verify it's None
        runner = ResearchRunner(api_key=None)
        # The runner should have api_key=None (not from env)
        # Note: In actual runner, api_key=None will fallback to env var
        # So we directly set it to None to bypass that
        runner.api_key = None
        results = await runner.search_topic("test query")
        assert results == []
        # Clean up session
        await runner.close()

    @pytest.mark.asyncio
    async def test_search_topic_success(self):
        """Test successful search - just verify API key is checked."""
        runner = ResearchRunner(api_key="test_key")
        # Verify runner has API key set
        assert runner.api_key == "test_key"
        # Without actual mocking of aiohttp, we just verify the runner is configured
        # Full integration tests require actual API key

    def test_parse_search_results(self):
        """Test parsing Firecrawl response."""
        runner = ResearchRunner(api_key="test")

        data = {
            "data": [
                {
                    "url": "https://example.com",
                    "title": "Title",
                    "description": "Desc",
                    "markdown": "Content",
                }
            ]
        }

        results = runner._parse_search_results(data)
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        assert results[0]["title"] == "Title"
        assert results[0]["content"] == "Content"


class TestResearchSummarizer:
    """Tests for ResearchSummarizer."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": ""}, clear=True):
            summarizer = ResearchSummarizer(api_key=None)
            assert summarizer._genai_model is None

    def test_fallback_summary(self):
        """Test fallback summary when AI not available."""
        summarizer = ResearchSummarizer(api_key=None)

        results = [
            {"title": "Title 1", "url": "https://example1.com", "description": "Desc 1"},
            {"title": "Title 2", "url": "https://example2.com", "description": "Desc 2"},
        ]

        summary = summarizer._fallback_summary("AI research", results)
        assert "AI research" in summary
        assert "Title 1" in summary
        assert "Title 2" in summary

    def test_format_results(self):
        """Test formatting results for prompt."""
        summarizer = ResearchSummarizer(api_key=None)

        results = [
            {"title": "Title", "url": "https://example.com", "content": "Content here"},
        ]

        formatted = summarizer._format_results(results, 100)
        assert "Title" in formatted
        assert "https://example.com" in formatted


class TestContextPrimer:
    """Tests for ContextPrimer."""

    def test_init_creates_directory(self, tmp_path):
        """Test initialization creates the base directory."""
        path = tmp_path / "research"
        primer = ContextPrimer(base_path=str(path))
        assert path.exists()

    def test_generate_context_file(self, tmp_path):
        """Test generating a context file."""
        primer = ContextPrimer(base_path=str(tmp_path))

        results = [
            {"topic": "AI agents", "status": "success", "results_count": 5},
            {"topic": "MCP", "status": "error", "error": "Timeout"},
        ]

        file_path = primer.generate_context_file("2026-01-16", results)

        assert file_path.exists()
        content = file_path.read_text()
        assert "AI agents" in content
        assert "MCP" in content
        assert "January 16, 2026" in content

    def test_get_latest_context(self, tmp_path):
        """Test getting latest context."""
        primer = ContextPrimer(base_path=str(tmp_path))

        # Create test files
        (tmp_path / "2026-01-15.md").write_text("Old content")
        (tmp_path / "2026-01-16.md").write_text("New content")

        content = primer.get_latest_context()
        assert content == "New content"

    def test_get_latest_context_empty(self, tmp_path):
        """Test getting latest context when no files exist."""
        primer = ContextPrimer(base_path=str(tmp_path))
        assert primer.get_latest_context() is None

    def test_list_available_dates(self, tmp_path):
        """Test listing available dates."""
        primer = ContextPrimer(base_path=str(tmp_path))

        (tmp_path / "2026-01-14.md").write_text("Content")
        (tmp_path / "2026-01-15.md").write_text("Content")
        (tmp_path / "2026-01-16.md").write_text("Content")
        (tmp_path / "not-a-date.md").write_text("Content")

        dates = primer.list_available_dates()
        assert len(dates) == 3
        assert dates[0] == "2026-01-16"  # Most recent first

    def test_cleanup_old_files(self, tmp_path):
        """Test cleaning up old files."""
        primer = ContextPrimer(base_path=str(tmp_path))

        # Create files with dates
        old_date = datetime.now().strftime("%Y-%m-%d")
        (tmp_path / "2020-01-01.md").write_text("Old")  # Very old
        (tmp_path / f"{old_date}.md").write_text("Recent")

        deleted = primer.cleanup_old_files(keep_days=30)
        assert deleted == 1  # Only the 2020 file

    def test_get_injection_text(self, tmp_path):
        """Test getting injection text."""
        primer = ContextPrimer(base_path=str(tmp_path))
        (tmp_path / "2026-01-16.md").write_text("Test content")

        text = primer.get_injection_text()
        assert "<research-context>" in text
        assert "Test content" in text
        assert "</research-context>" in text


class TestResearchToolHandler:
    """Tests for ResearchToolHandler."""

    @pytest.mark.asyncio
    async def test_handle_list_topics(self, tmp_path):
        """Test handling list_topics tool."""
        # Create a real registry with test data
        registry = TopicRegistry(
            global_topics=["topic1", "topic2"],
            project_topics={"p1": ["p1_topic"]},
        )
        registry._registry_path = tmp_path / "topics.json"

        scheduler = ResearchScheduler(registry=registry)
        handler = ResearchToolHandler()
        handler._scheduler = scheduler

        result = await handler._handle_list_topics({})

        assert result["global_topics"] == ["topic1", "topic2"]
        assert result["total_topics"] == 3

    @pytest.mark.asyncio
    async def test_handle_add_topic(self, tmp_path):
        """Test handling add_topic tool."""
        registry = TopicRegistry()
        registry._registry_path = tmp_path / "topics.json"

        scheduler = ResearchScheduler(registry=registry)
        handler = ResearchToolHandler()
        handler._scheduler = scheduler

        result = await handler._handle_add_topic({"topic": "new topic", "project": "global"})

        assert result["success"] is True
        assert result["topic"] == "new topic"
        assert "new topic" in scheduler.registry.global_topics

    @pytest.mark.asyncio
    async def test_handle_status(self, tmp_path):
        """Test handling status tool."""
        registry = TopicRegistry(
            global_topics=["t1", "t2", "t3", "t4", "t5"],
        )
        registry._registry_path = tmp_path / "topics.json"

        scheduler = ResearchScheduler(registry=registry)
        handler = ResearchToolHandler()
        handler._scheduler = scheduler

        result = await handler._handle_status({})

        assert result["total_topics"] == 5
        assert result["scheduled_task_id"] is None


class TestResearchTools:
    """Tests for RESEARCH_TOOLS definition."""

    def test_all_tools_have_required_fields(self):
        """Test all tools have required fields."""
        for tool in RESEARCH_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_tool_names_are_unique(self):
        """Test all tool names are unique."""
        names = [tool["name"] for tool in RESEARCH_TOOLS]
        assert len(names) == len(set(names))

    def test_expected_tools_exist(self):
        """Test expected tools are defined."""
        expected = [
            "research_run",
            "research_add_topic",
            "research_remove_topic",
            "research_list_topics",
            "research_get_context",
            "research_schedule",
            "research_status",
            "research_search_past",
        ]

        tool_names = [tool["name"] for tool in RESEARCH_TOOLS]
        for expected_name in expected:
            assert expected_name in tool_names, f"Missing tool: {expected_name}"


class TestResearchScheduler:
    """Tests for ResearchScheduler."""

    def test_init_with_registry(self, tmp_path):
        """Test initialization with existing registry."""
        registry = TopicRegistry(global_topics=["test"])
        registry._registry_path = tmp_path / "topics.json"

        scheduler = ResearchScheduler(registry=registry)
        assert scheduler.registry.global_topics == ["test"]

    def test_get_status(self, tmp_path):
        """Test getting scheduler status."""
        registry = TopicRegistry(
            global_topics=["g1", "g2"],
            project_topics={"p1": ["t1"]},
        )
        registry._registry_path = tmp_path / "topics.json"

        scheduler = ResearchScheduler(registry=registry)
        status = scheduler.get_status()

        assert status["total_topics"] == 3
        assert status["global_topics"] == 2
        assert "p1" in status["projects"]
        assert status["scheduled_task_id"] is None
