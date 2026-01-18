"""Tests for the plugin architecture."""
import pytest
from typing import Any, Callable, Dict, List

from src.mcp.plugins import PluginInterface, PluginRegistry, PluginTier
from src.mcp.plugins.core_plugin import CorePlugin, CORE_TOOL_DEFINITIONS
from src.license import LicenseValidator, LicenseInfo, LicenseStatus


class MockPlugin(PluginInterface):
    """Mock plugin for testing."""

    name = "mock-plugin"
    version = "1.0.0"
    tier = PluginTier.FREE
    description = "Mock plugin for tests"

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "mock_tool", "description": "A mock tool", "parameters": {}},
        ]

    def get_tool_handlers(self) -> Dict[str, Callable]:
        return {"mock_tool": self._mock_handler}

    async def _mock_handler(self, args: dict) -> dict:
        return {"result": "mock"}


class MockEnterprisePlugin(PluginInterface):
    """Mock enterprise plugin for testing."""

    name = "mock-enterprise"
    version = "1.0.0"
    tier = PluginTier.ENTERPRISE
    description = "Mock enterprise plugin"

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "enterprise_tool", "description": "Enterprise only", "parameters": {}},
        ]

    def get_tool_handlers(self) -> Dict[str, Callable]:
        return {"enterprise_tool": self._handler}

    async def _handler(self, args: dict) -> dict:
        return {"result": "enterprise"}

    def _check_license_key(self, license_key: str) -> bool:
        # Accept "valid-enterprise-key" for testing
        return license_key == "valid-enterprise-key"


class TestPluginInterface:
    """Tests for PluginInterface base class."""

    def test_free_plugin_validates_without_license(self):
        """Free plugins should validate without a license key."""
        plugin = MockPlugin()
        assert plugin.validate_license(None) is True
        assert plugin.validate_license("any-key") is True

    def test_enterprise_plugin_requires_license(self):
        """Enterprise plugins should require a valid license."""
        plugin = MockEnterprisePlugin()
        assert plugin.validate_license(None) is False
        assert plugin.validate_license("invalid-key") is False
        assert plugin.validate_license("valid-enterprise-key") is True

    def test_plugin_has_required_attributes(self):
        """Plugins should have name, version, tier, description."""
        plugin = MockPlugin()
        assert plugin.name == "mock-plugin"
        assert plugin.version == "1.0.0"
        assert plugin.tier == PluginTier.FREE
        assert plugin.description == "Mock plugin for tests"


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_free_plugin(self):
        """Free plugins should register without license."""
        registry = PluginRegistry()
        plugin = MockPlugin()

        result = registry.register(plugin)

        assert result is True
        assert registry.get_plugin("mock-plugin") is plugin

    def test_register_enterprise_without_license_fails(self):
        """Enterprise plugins should fail without license."""
        registry = PluginRegistry()
        plugin = MockEnterprisePlugin()

        result = registry.register(plugin)

        assert result is False
        assert registry.get_plugin("mock-enterprise") is None

    def test_register_enterprise_with_valid_license(self):
        """Enterprise plugins should register with valid license."""
        registry = PluginRegistry()
        registry.set_license_key("valid-enterprise-key")
        plugin = MockEnterprisePlugin()

        result = registry.register(plugin)

        assert result is True
        assert registry.get_plugin("mock-enterprise") is plugin

    def test_get_all_tools(self):
        """Should aggregate tools from all plugins."""
        registry = PluginRegistry()
        registry.register(MockPlugin())

        tools = registry.get_all_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "mock_tool"

    def test_get_tool_handler(self):
        """Should return handler for registered tool."""
        registry = PluginRegistry()
        registry.register(MockPlugin())

        handler = registry.get_tool_handler("mock_tool")

        assert handler is not None

    def test_get_tool_handler_unknown_tool(self):
        """Should return None for unknown tool."""
        registry = PluginRegistry()

        handler = registry.get_tool_handler("unknown_tool")

        assert handler is None

    def test_unregister_plugin(self):
        """Should unregister plugin and remove tools."""
        registry = PluginRegistry()
        registry.register(MockPlugin())

        result = registry.unregister("mock-plugin")

        assert result is True
        assert registry.get_plugin("mock-plugin") is None
        assert registry.get_tool_handler("mock_tool") is None

    def test_get_stats(self):
        """Should return registry statistics."""
        registry = PluginRegistry()
        registry.register(MockPlugin())

        stats = registry.get_stats()

        assert stats["total_plugins"] == 1
        assert stats["free_plugins"] == 1
        assert stats["enterprise_plugins"] == 0
        assert stats["total_tools"] == 1


class TestCorePlugin:
    """Tests for CorePlugin."""

    def test_core_plugin_is_free_tier(self):
        """Core plugin should be free tier."""
        plugin = CorePlugin()
        assert plugin.tier == PluginTier.FREE
        assert plugin.name == "core"

    def test_core_plugin_has_tools(self):
        """Core plugin should have tool definitions."""
        plugin = CorePlugin()
        tools = plugin.get_tools()

        assert len(tools) > 0
        tool_names = [t["name"] for t in tools]
        assert "spawn_agent" in tool_names
        assert "immune_status" in tool_names
        assert "healing_status" in tool_names

    def test_core_plugin_validates_without_license(self):
        """Core plugin should validate without license."""
        plugin = CorePlugin()
        assert plugin.validate_license(None) is True


class TestLicenseValidator:
    """Tests for LicenseValidator."""

    def test_validate_none_returns_not_found(self):
        """Should return NOT_FOUND for None key."""
        validator = LicenseValidator()

        info = validator.validate(None)

        assert info.status == LicenseStatus.NOT_FOUND
        assert info.is_valid is False

    def test_development_key_all_features(self):
        """Dev key 'dev-all-features' should grant all access."""
        validator = LicenseValidator()

        info = validator.validate("dev-all-features")

        assert info.status == LicenseStatus.VALID
        assert info.is_valid is True
        assert info.has_feature("federation") is True
        assert info.has_feature("content") is True
        assert info.has_feature("any-feature") is True

    def test_development_key_specific_features(self):
        """Dev keys should parse feature names."""
        validator = LicenseValidator()

        info = validator.validate("dev-federation-content")

        assert info.status == LicenseStatus.VALID
        assert info.has_feature("federation") is True
        assert info.has_feature("content") is True
        assert info.has_feature("sync") is False

    def test_invalid_key_format(self):
        """Invalid keys should return INVALID status."""
        validator = LicenseValidator()

        info = validator.validate("not-a-valid-key")

        assert info.status == LicenseStatus.INVALID
        assert info.is_valid is False


class TestPluginIntegration:
    """Integration tests for plugin system."""

    def test_full_registration_flow(self):
        """Test registering multiple plugins."""
        registry = PluginRegistry()

        # Register core plugin
        core = CorePlugin()
        assert registry.register(core) is True

        # Register mock free plugin
        mock = MockPlugin()
        assert registry.register(mock) is True

        # Try enterprise without license
        enterprise = MockEnterprisePlugin()
        assert registry.register(enterprise) is False

        # Set license and try again
        registry.set_license_key("valid-enterprise-key")
        enterprise2 = MockEnterprisePlugin()
        assert registry.register(enterprise2) is True

        # Verify all tools are available
        tools = registry.get_all_tools()
        tool_names = [t["name"] for t in tools]
        assert "spawn_agent" in tool_names  # from core
        assert "mock_tool" in tool_names  # from mock
        assert "enterprise_tool" in tool_names  # from enterprise

    def test_tool_definitions_are_valid(self):
        """Verify core tool definitions have required fields."""
        for tool in CORE_TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing description"
            assert "parameters" in tool, f"Tool {tool.get('name')} missing parameters"
            assert isinstance(tool["parameters"], dict), f"Tool {tool.get('name')} parameters should be dict"
