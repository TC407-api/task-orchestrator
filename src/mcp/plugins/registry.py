"""Plugin registry for Task Orchestrator MCP server."""
from typing import Any, Callable, Dict, List, Optional

from .base import PluginInterface, PluginTier


class PluginRegistry:
    """
    Manages plugins for the MCP server.

    Handles plugin registration, license validation, and tool aggregation.
    Free plugins are always loaded; enterprise plugins require a valid
    license key set via TASK_ORCHESTRATOR_LICENSE environment variable.

    Usage:
        registry = PluginRegistry()
        registry.register(CorePlugin())

        # Set license key for enterprise plugins
        registry.set_license_key(os.getenv("TASK_ORCHESTRATOR_LICENSE"))

        # Try to register enterprise plugins
        try:
            from task_orchestrator_enterprise import FederationPlugin
            registry.register(FederationPlugin())
        except ImportError:
            pass  # Enterprise not installed

        # Get all available tools
        tools = registry.get_all_tools()
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._plugins: Dict[str, PluginInterface] = {}
        self._license_key: Optional[str] = None
        self._tool_to_plugin: Dict[str, str] = {}  # tool_name -> plugin_name

    def set_license_key(self, key: Optional[str]) -> None:
        """
        Set the enterprise license key.

        This should be called before registering enterprise plugins.

        Args:
            key: License key string or None
        """
        self._license_key = key

    def register(self, plugin: PluginInterface) -> bool:
        """
        Register a plugin with the registry.

        Free plugins are always registered.
        Enterprise plugins require a valid license key.

        Args:
            plugin: Plugin instance to register

        Returns:
            True if plugin was registered, False if license validation failed
        """
        # Check license for enterprise plugins
        if not plugin.validate_license(self._license_key):
            return False

        # Check for duplicate plugin names
        if plugin.name in self._plugins:
            # Allow re-registration (useful for testing/hot reload)
            self.unregister(plugin.name)

        # Register the plugin
        self._plugins[plugin.name] = plugin

        # Map tool names to this plugin
        for tool in plugin.get_tools():
            tool_name = tool.get("name", "")
            if tool_name:
                self._tool_to_plugin[tool_name] = plugin.name

        # Call plugin's on_load hook
        plugin.on_load()

        return True

    def unregister(self, plugin_name: str) -> bool:
        """
        Unregister a plugin by name.

        Args:
            plugin_name: Name of the plugin to unregister

        Returns:
            True if plugin was found and unregistered
        """
        if plugin_name not in self._plugins:
            return False

        plugin = self._plugins[plugin_name]

        # Remove tool mappings
        tools_to_remove = [
            name for name, p in self._tool_to_plugin.items()
            if p == plugin_name
        ]
        for tool_name in tools_to_remove:
            del self._tool_to_plugin[tool_name]

        # Call plugin's on_unload hook
        plugin.on_unload()

        # Remove plugin
        del self._plugins[plugin_name]

        return True

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_all_plugins(self) -> List[PluginInterface]:
        """Get all registered plugins."""
        return list(self._plugins.values())

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions from all registered plugins.

        Returns:
            List of all tool definitions
        """
        tools = []
        for plugin in self._plugins.values():
            tools.extend(plugin.get_tools())
        return tools

    def get_tool_handler(self, tool_name: str) -> Optional[Callable]:
        """
        Get the handler function for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Handler function or None if tool not found
        """
        plugin_name = self._tool_to_plugin.get(tool_name)
        if plugin_name is None:
            return None

        plugin = self._plugins.get(plugin_name)
        if plugin is None:
            return None

        handlers = plugin.get_tool_handlers()
        return handlers.get(tool_name)

    def get_plugin_for_tool(self, tool_name: str) -> Optional[PluginInterface]:
        """Get the plugin that provides a specific tool."""
        plugin_name = self._tool_to_plugin.get(tool_name)
        if plugin_name is None:
            return None
        return self._plugins.get(plugin_name)

    def list_enterprise_features(self) -> List[str]:
        """
        List enterprise features that would be available with a license.

        Returns:
            List of enterprise plugin names (whether loaded or not)
        """
        # Known enterprise plugins
        return [
            "federation",
            "sync",
            "content",
            "research",
            "learning",
        ]

    def is_enterprise_licensed(self) -> bool:
        """Check if a valid enterprise license is set."""
        if self._license_key is None:
            return False

        # Check if any enterprise plugins are loaded
        for plugin in self._plugins.values():
            if plugin.tier == PluginTier.ENTERPRISE:
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        free_count = sum(
            1 for p in self._plugins.values()
            if p.tier == PluginTier.FREE
        )
        enterprise_count = sum(
            1 for p in self._plugins.values()
            if p.tier == PluginTier.ENTERPRISE
        )

        return {
            "total_plugins": len(self._plugins),
            "free_plugins": free_count,
            "enterprise_plugins": enterprise_count,
            "total_tools": len(self._tool_to_plugin),
            "is_licensed": self.is_enterprise_licensed(),
        }
