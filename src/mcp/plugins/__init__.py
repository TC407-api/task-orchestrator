"""Plugin architecture for Task Orchestrator MCP server.

Enables open-core model with free (MIT) and enterprise (licensed) tiers.
"""
from .base import PluginInterface, PluginTier
from .registry import PluginRegistry

__all__ = [
    "PluginInterface",
    "PluginTier",
    "PluginRegistry",
]
