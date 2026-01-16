"""Tests for learning_tools module."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import json

from .learning_tools import (
    LEARNING_TOOLS,
    LearningToolHandler,
    EXTRACT_PROMPT,
    APPLY_PROMPT,
    RECALL_PROMPT,
    LEARNING_GROUP_ID,
)


class TestLearningToolsDefinition:
    """Test tool definitions."""

    def test_learning_tools_is_list(self):
        """LEARNING_TOOLS should be a list."""
        assert isinstance(LEARNING_TOOLS, list)

    def test_learning_tools_has_one_tool(self):
        """LEARNING_TOOLS should have exactly one tool."""
        assert len(LEARNING_TOOLS) == 1

    def test_learning_workflow_tool_name(self):
        """learning_workflow tool should have correct name."""
        tool = LEARNING_TOOLS[0]
        assert tool["name"] == "learning_workflow"

    def test_learning_workflow_has_input_schema(self):
        """learning_workflow should have inputSchema."""
        tool = LEARNING_TOOLS[0]
        assert "inputSchema" in tool
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_learning_workflow_required_fields(self):
        """learning_workflow should require operation and topic."""
        tool = LEARNING_TOOLS[0]
        required = tool["inputSchema"]["required"]
        assert "operation" in required
        assert "topic" in required

    def test_operation_enum_values(self):
        """operation should have correct enum values."""
        tool = LEARNING_TOOLS[0]
        operation = tool["inputSchema"]["properties"]["operation"]
        assert operation["enum"] == ["extract", "apply", "recall"]


class TestPromptTemplates:
    """Test prompt template structures."""

    def test_extract_prompt_has_system_and_task(self):
        """EXTRACT_PROMPT should have system and task."""
        assert "system" in EXTRACT_PROMPT
        assert "task" in EXTRACT_PROMPT

    def test_apply_prompt_has_system_and_task(self):
        """APPLY_PROMPT should have system and task."""
        assert "system" in APPLY_PROMPT
        assert "task" in APPLY_PROMPT

    def test_recall_prompt_has_system_and_task(self):
        """RECALL_PROMPT should have system and task."""
        assert "system" in RECALL_PROMPT
        assert "task" in RECALL_PROMPT


class TestLearningToolHandler:
    """Test LearningToolHandler."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create mock GraphitiClient."""
        client = MagicMock()
        client.add_memory = AsyncMock(return_value={"uuid": "test-uuid", "status": "ok"})
        client.search_memory_facts = AsyncMock(return_value=[])
        client.search_memory_nodes = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def mock_server(self):
        """Create mock server."""
        server = MagicMock()
        server._handle_spawn_parallel_agents = AsyncMock(return_value={"results": []})
        server._handle_spawn_archetype_agent = AsyncMock(return_value={"response": "{}"})
        return server

    @pytest.fixture
    def handler(self, mock_server, mock_graphiti_client):
        """Create handler with mock server and graphiti client."""
        h = LearningToolHandler(mock_server)
        h._graphiti_client = mock_graphiti_client
        return h

    @pytest.mark.asyncio
    async def test_handle_tool_unknown_tool(self, handler):
        """handle_tool should return error for unknown tool."""
        result = await handler.handle_tool("unknown_tool", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_unknown_operation(self, handler):
        """handle should return error for unknown operation."""
        result = await handler.handle({"operation": "unknown"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_no_portfolio(self, handler, tmp_path):
        """extract should handle missing portfolio."""
        handler._portfolio_path = tmp_path / "nonexistent.json"
        result = await handler._extract({"topic": "test", "projects": []})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_empty_projects(self, handler, tmp_path):
        """extract should handle empty projects."""
        portfolio_path = tmp_path / "portfolio.json"
        portfolio_path.write_text(json.dumps({"products": [], "internalTools": [], "personalProjects": []}))
        handler._portfolio_path = portfolio_path
        result = await handler._extract({"topic": "test", "projects": []})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_with_projects(self, handler, mock_server, tmp_path):
        """extract should spawn parallel agents for projects."""
        portfolio_path = tmp_path / "portfolio.json"
        portfolio = {
            "products": [{"name": "test-project", "stack": ["python"]}],
            "internalTools": [],
            "personalProjects": []
        }
        portfolio_path.write_text(json.dumps(portfolio))
        handler._portfolio_path = portfolio_path

        mock_server._handle_spawn_parallel_agents = AsyncMock(return_value={
            "results": [{"success": True, "response": '{"learnings":[]}'}]
        })

        result = await handler._extract({"topic": "authentication"})
        assert result["operation"] == "extract"
        assert result["topic"] == "authentication"
        assert "count" in result
        mock_server._handle_spawn_parallel_agents.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_calls_architect_agent(self, handler, mock_server):
        """apply should spawn architect agent."""
        result = await handler._apply({"topic": "security"})
        assert result["operation"] == "apply"
        assert result["topic"] == "security"
        mock_server._handle_spawn_archetype_agent.assert_called_once()
        call_args = mock_server._handle_spawn_archetype_agent.call_args[0][0]
        assert call_args["archetype"] == "architect"

    @pytest.mark.asyncio
    async def test_recall_calls_researcher_agent(self, handler, mock_server):
        """recall should spawn researcher agent."""
        result = await handler._recall({"topic": "caching"})
        assert result["operation"] == "recall"
        assert result["topic"] == "caching"
        mock_server._handle_spawn_archetype_agent.assert_called_once()
        call_args = mock_server._handle_spawn_archetype_agent.call_args[0][0]
        assert call_args["archetype"] == "researcher"

    def test_parse_json_valid(self, handler):
        """_parse_json should parse valid JSON."""
        result = handler._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_invalid(self, handler):
        """_parse_json should return empty dict for invalid JSON."""
        result = handler._parse_json("not json")
        assert result == {}

    def test_get_projects_all(self, handler):
        """_get_projects should return all projects when no filter."""
        portfolio = {
            "products": [{"name": "p1"}],
            "internalTools": [{"name": "t1"}],
            "personalProjects": [{"name": "pp1"}]
        }
        result = handler._get_projects(portfolio, [])
        assert len(result) == 3

    def test_get_projects_filtered(self, handler):
        """_get_projects should filter by name."""
        portfolio = {
            "products": [{"name": "p1"}],
            "internalTools": [{"name": "t1"}],
            "personalProjects": [{"name": "pp1"}]
        }
        result = handler._get_projects(portfolio, ["p1", "t1"])
        assert len(result) == 2
        assert all(p["name"] in ["p1", "t1"] for p in result)


class TestGraphitiIntegration:
    """Test Graphiti integration methods."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create mock GraphitiClient."""
        client = MagicMock()
        client.add_memory = AsyncMock(return_value={"uuid": "test-uuid", "status": "ok"})
        client.search_memory_facts = AsyncMock(return_value=[])
        client.search_memory_nodes = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def mock_server(self):
        """Create mock server."""
        server = MagicMock()
        server._handle_spawn_parallel_agents = AsyncMock(return_value={"results": []})
        server._handle_spawn_archetype_agent = AsyncMock(return_value={"response": "{}"})
        return server

    @pytest.fixture
    def handler(self, mock_server, mock_graphiti_client):
        """Create handler with mock server and graphiti client."""
        h = LearningToolHandler(mock_server)
        h._graphiti_client = mock_graphiti_client
        return h

    @pytest.mark.asyncio
    async def test_store_graphiti_success(self, handler, mock_graphiti_client):
        """_store_graphiti should store learnings."""
        learnings = [
            {"project": "test-project", "type": "pattern", "name": "test_pattern"}
        ]
        result = await handler._store_graphiti(learnings, "auth")
        assert result["stored"] == 1
        assert result["total"] == 1
        mock_graphiti_client.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_graphiti_multiple(self, handler, mock_graphiti_client):
        """_store_graphiti should store multiple learnings."""
        learnings = [
            {"project": "p1", "type": "pattern", "name": "pattern1"},
            {"project": "p2", "type": "anti-pattern", "name": "antipattern1"},
        ]
        result = await handler._store_graphiti(learnings, "security")
        assert result["stored"] == 2
        assert result["total"] == 2
        assert mock_graphiti_client.add_memory.call_count == 2

    @pytest.mark.asyncio
    async def test_store_graphiti_handles_error(self, handler, mock_graphiti_client):
        """_store_graphiti should handle errors gracefully."""
        mock_graphiti_client.add_memory = AsyncMock(return_value={"error": "Connection failed"})
        learnings = [{"project": "test", "name": "test"}]
        result = await handler._store_graphiti(learnings, "test")
        assert result["stored"] == 0
        assert result["errors"] is not None

    @pytest.mark.asyncio
    async def test_search_graphiti_empty(self, handler, mock_graphiti_client):
        """_search_graphiti should return empty list when no results."""
        result = await handler._search_graphiti("unknown_topic")
        assert result == []
        mock_graphiti_client.search_memory_facts.assert_called_once()
        mock_graphiti_client.search_memory_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_graphiti_with_facts(self, handler, mock_graphiti_client):
        """_search_graphiti should return parsed facts."""
        mock_graphiti_client.search_memory_facts = AsyncMock(return_value=[
            {"content": '{"type": "learning", "summary": "Test learning"}'},
        ])
        result = await handler._search_graphiti("test_topic")
        assert len(result) >= 1
        assert result[0]["type"] == "learning"

    @pytest.mark.asyncio
    async def test_search_graphiti_with_nodes(self, handler, mock_graphiti_client):
        """_search_graphiti should include matching nodes."""
        mock_graphiti_client.search_memory_nodes = AsyncMock(return_value=[
            {"name": "learning-test-auth", "content": "Auth pattern"},
        ])
        result = await handler._search_graphiti("auth")
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_extract_includes_graphiti_result(self, handler, mock_server, mock_graphiti_client, tmp_path):
        """extract should include graphiti storage result."""
        portfolio_path = tmp_path / "portfolio.json"
        portfolio = {
            "products": [{"name": "test-project", "stack": ["python"]}],
            "internalTools": [],
            "personalProjects": []
        }
        portfolio_path.write_text(json.dumps(portfolio))
        handler._portfolio_path = portfolio_path

        mock_server._handle_spawn_parallel_agents = AsyncMock(return_value={
            "results": [{"success": True, "response": '{"learnings":[{"type":"pattern","name":"test"}]}'}]
        })

        result = await handler._extract({"topic": "auth"})
        assert "graphiti" in result
        assert result["graphiti"]["total"] == 1
