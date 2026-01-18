"""Base plugin interface for Task Orchestrator MCP server."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class PluginTier(str, Enum):
    """License tier for plugins."""
    FREE = "free"           # MIT license, always available
    ENTERPRISE = "enterprise"  # Requires license key


class PluginInterface(ABC):
    """
    Base class for all Task Orchestrator plugins.

    Plugins provide tools that can be loaded into the MCP server.
    Free plugins are always available; enterprise plugins require
    a valid license key.

    Attributes:
        name: Unique identifier for this plugin
        version: Semantic version string
        tier: License tier (FREE or ENTERPRISE)
        description: Human-readable description
    """

    name: str = "unnamed-plugin"
    version: str = "1.0.0"
    tier: PluginTier = PluginTier.FREE
    description: str = ""

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Return tool definitions for this plugin.

        Each tool definition should be a dict with:
        - name: Tool name (str)
        - description: What the tool does (str)
        - parameters: JSON schema for parameters (dict)

        Returns:
            List of tool definition dicts
        """
        pass

    @abstractmethod
    def get_tool_handlers(self) -> Dict[str, Callable]:
        """
        Return handlers for each tool.

        Returns:
            Dict mapping tool names to async handler functions.
            Each handler takes (args: dict) and returns dict.
        """
        pass

    def on_load(self) -> None:
        """Called when plugin is loaded into the registry."""
        pass

    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass

    def validate_license(self, license_key: Optional[str]) -> bool:
        """
        Check if the provided license key is valid for this plugin.

        Free plugins always return True.
        Enterprise plugins check the license key.

        Args:
            license_key: The license key to validate (may be None)

        Returns:
            True if plugin can be used, False otherwise
        """
        if self.tier == PluginTier.FREE:
            return True

        # Enterprise plugins require license validation
        if license_key is None:
            return False

        return self._check_license_key(license_key)

    def _check_license_key(self, license_key: str) -> bool:
        """
        Validate an enterprise license key.

        Override this in enterprise plugins with actual validation.
        Default implementation always returns False (no valid license).

        Args:
            license_key: The license key string

        Returns:
            True if license is valid for this plugin
        """
        return False
