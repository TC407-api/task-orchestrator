"""Tests for ToolRouter - TDD RED phase."""
import pytest
from src.mcp.tool_router import (
    ToolRouter,
    ToolCategory,
    CORE_TOOLS,
    TOOL_CATEGORIES,
)


def test_router_initializes_with_core_tools():
    """Test that router starts with CORE category loaded."""
    router = ToolRouter()

    # CORE should be in loaded categories by default
    assert ToolCategory.CORE in router.get_loaded_categories()

    # Should only have CORE category loaded initially
    assert len(router.get_loaded_categories()) == 1


def test_router_switches_to_dynamic_at_10_percent():
    """Test dynamic mode triggers at 10% remaining context."""
    router = ToolRouter()

    # At 15% remaining -> should NOT switch
    assert router.should_switch_to_dynamic(0.15) is False

    # At 10% remaining -> should switch
    assert router.should_switch_to_dynamic(0.10) is True

    # At 5% remaining -> definitely should switch
    assert router.should_switch_to_dynamic(0.05) is True


def test_request_tool_loads_category():
    """Test that request_tool loads the entire category."""
    # Create mock tools
    mock_tools = [
        {"name": "tasks_list", "description": "List tasks"},
        {"name": "spawn_agent", "description": "Spawn agent"},
        {"name": "immune_status", "description": "Immune status"},
        {"name": "immune_check", "description": "Check immune"},
    ]

    router = ToolRouter(all_tools=mock_tools)
    router.set_dynamic_mode(True)

    # Request immune category
    loaded_tools = router.request_tool("immune")

    # Should return tools from immune category
    tool_names = [t["name"] for t in loaded_tools]
    assert "immune_status" in tool_names
    assert "immune_check" in tool_names

    # Category should now be loaded
    assert ToolCategory.IMMUNE in router.get_loaded_categories()


def test_get_available_tools_respects_mode():
    """Test that get_available_tools returns appropriate tools based on mode."""
    mock_tools = [
        {"name": "tasks_list", "description": "List tasks"},
        {"name": "tasks_add", "description": "Add task"},
        {"name": "spawn_agent", "description": "Spawn agent"},
        {"name": "healing_status", "description": "Healing status"},
        {"name": "request_tool", "description": "Request tool"},
        {"name": "immune_status", "description": "Immune status"},
        {"name": "cost_summary", "description": "Cost summary"},
    ]

    router = ToolRouter(all_tools=mock_tools)

    # In full mode (dynamic off), should return all tools
    router.set_dynamic_mode(False)
    all_tools = router.get_available_tools()
    assert len(all_tools) == 7

    # In dynamic mode, should only return CORE tools
    router.set_dynamic_mode(True)
    core_tools = router.get_available_tools()
    core_names = [t["name"] for t in core_tools]

    assert len(core_tools) == 5  # Only CORE tools
    for core_tool in CORE_TOOLS:
        assert core_tool in core_names


def test_core_tools_constant_has_expected_tools():
    """Verify CORE_TOOLS contains the expected tools."""
    expected = ["tasks_list", "tasks_add", "spawn_agent", "healing_status", "request_tool"]

    for tool in expected:
        assert tool in CORE_TOOLS


def test_category_for_tool_lookup():
    """Test looking up which category a tool belongs to."""
    router = ToolRouter()

    assert router.get_category_for_tool("tasks_list") == ToolCategory.CORE
    assert router.get_category_for_tool("immune_status") == ToolCategory.IMMUNE
    assert router.get_category_for_tool("sync_trigger") == ToolCategory.SYNC
    assert router.get_category_for_tool("nonexistent_tool") is None
