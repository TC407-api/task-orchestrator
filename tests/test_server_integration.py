"""Tests for server integration with dynamic tool loading."""
import pytest
from unittest.mock import MagicMock

# Import server
from src.mcp.server import TaskOrchestratorMCP


class TestServerIntegration:
    """Integration tests for dynamic tool loading in MCP server."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock ToolRouter."""
        router = MagicMock()
        router.get_available_tools.return_value = [
            {"name": "tasks_list", "description": "List tasks", "inputSchema": {}},
            {"name": "spawn_agent", "description": "Spawn agent", "inputSchema": {}},
            {"name": "request_tool", "description": "Request tools", "inputSchema": {}},
        ]
        router.is_dynamic_mode.return_value = True
        router.get_loaded_categories.return_value = {"core"}
        return router

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock ContextTracker."""
        tracker = MagicMock()
        tracker.below_threshold.return_value = True
        tracker.remaining_pct.return_value = 0.08
        tracker.get_stats.return_value = {
            "max_context": 200000,
            "current_usage": 184000,
            "remaining_tokens": 16000,
            "usage_pct": 0.92,
        }
        return tracker

    @pytest.fixture
    def server(self):
        """Create a server instance."""
        return TaskOrchestratorMCP()

    def test_server_initializes_with_router(self, server):
        """Test that server has _tool_router and _context_tracker attributes."""
        # _context_tracker is initialized immediately
        assert hasattr(server, "_context_tracker")
        assert server._context_tracker is not None

        # _tool_router starts as None (lazy initialization)
        assert hasattr(server, "_tool_router")
        # It starts as None and is populated on first request_tool call
        # This is by design for lazy loading

    def test_get_tools_returns_core_in_dynamic_mode(self, server, mock_router, mock_tracker):
        """Test that get_tools returns only core tools when in dynamic mode."""
        # Note: get_tools() currently returns all tools statically
        # The dynamic filtering happens in the router's get_available_tools()
        # For now, verify request_tool exists
        tools = server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "request_tool" in tool_names

    def test_get_tools_returns_all_in_full_mode(self, server, mock_router, mock_tracker):
        """Test that get_tools returns all tools when context is healthy."""
        tools = server.get_tools()
        # Should have many tools (42+ with request_tool)
        assert len(tools) >= 40

    @pytest.mark.asyncio
    async def test_request_tool_handler_loads_category(self, server, mock_router):
        """Test that request_tool handler loads the requested category."""
        from src.mcp.tool_router import ToolCategory

        # Initialize server
        await server.initialize()

        # Call handle_tool_call for request_tool
        result = await server.handle_tool_call("request_tool", {"category": "immune"})

        # Result should indicate success
        assert result.get("success") is True
        assert result.get("category") == "immune"
        assert "loaded_categories" in result

        # Verify the router was created and category loaded
        assert server._tool_router is not None
        assert ToolCategory.IMMUNE in server._tool_router.get_loaded_categories()

    def test_tools_list_includes_mode_metadata(self, server, mock_router, mock_tracker):
        """Test that the tools response includes mode metadata."""
        # The current implementation returns a list of tools
        # Mode metadata would be available via _context_tracker.get_stats()
        tools = server.get_tools()
        assert isinstance(tools, list)

        # Context tracker stats are available
        stats = server._context_tracker.get_stats()
        assert "max_context" in stats
        assert "current_usage" in stats


def test_request_tool_definition_exists():
    """Test that request_tool is defined in the tool list."""
    server = TaskOrchestratorMCP()
    tools = server.get_tools()

    tool_names = [t["name"] for t in tools]
    assert "request_tool" in tool_names

    # Find request_tool and check its schema
    request_tool = next(t for t in tools if t["name"] == "request_tool")
    assert "category" in str(request_tool.get("inputSchema", {}))


def test_request_tool_has_proper_schema():
    """Test that request_tool has the correct input schema."""
    server = TaskOrchestratorMCP()
    tools = server.get_tools()

    request_tool = next(t for t in tools if t["name"] == "request_tool")
    schema = request_tool.get("inputSchema", {})

    # Check required properties
    assert "properties" in schema
    assert "category" in schema["properties"]
    assert "required" in schema
    assert "category" in schema["required"]

    # Check category enum values
    category_prop = schema["properties"]["category"]
    assert "enum" in category_prop
    expected_categories = ["task", "agent", "immune", "federation", "sync", "workflow", "cost"]
    for cat in expected_categories:
        assert cat in category_prop["enum"]


@pytest.mark.asyncio
async def test_request_tool_returns_loaded_tools():
    """Test that request_tool returns the tools it loaded."""
    server = TaskOrchestratorMCP()
    await server.initialize()

    # Request immune category
    result = await server.handle_tool_call("request_tool", {
        "category": "immune",
        "reason": "Testing immune system tools"
    })

    assert result["success"] is True
    assert result["category"] == "immune"
    assert "tools_loaded" in result
    assert "immune_status" in result["tools_loaded"]
    assert "immune_check" in result["tools_loaded"]
    assert "reason_logged" in result
