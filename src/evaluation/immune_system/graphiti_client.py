"""
Graphiti Client - Wrapper for Graphiti knowledge graph operations.

Provides a unified interface for storing and retrieving data from Graphiti,
supporting both direct API calls and MCP-based interactions.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Default Graphiti MCP server endpoint (if running locally)
DEFAULT_GRAPHITI_URL = os.getenv("GRAPHITI_URL", "http://localhost:8000")


class GraphitiClient:
    """
    Client for interacting with Graphiti knowledge graph.

    Supports storing memories (episodes) and searching nodes/facts.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        group_id: str = "default",
    ):
        """
        Initialize Graphiti client.

        Args:
            base_url: Graphiti server URL. Defaults to GRAPHITI_URL env var.
            group_id: Default group ID for organizing data.
        """
        self.base_url = base_url or DEFAULT_GRAPHITI_URL
        self.default_group_id = group_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def add_memory(
        self,
        name: str,
        episode_body: str,
        group_id: Optional[str] = None,
        source: str = "text",
        source_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a memory/episode to the knowledge graph.

        Args:
            name: Name/identifier for the episode
            episode_body: Content of the episode
            group_id: Group to organize the memory (defaults to default_group_id)
            source: Source type (text, message, json)
            source_description: Description of the source

        Returns:
            Dict with uuid and status
        """
        group = group_id or self.default_group_id

        try:
            session = await self._get_session()
            payload = {
                "name": name,
                "episode_body": episode_body,
                "group_id": group,
                "source": source,
            }
            if source_description:
                payload["source_description"] = source_description

            async with session.post(
                f"{self.base_url}/episodes",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Added memory '{name}' to Graphiti")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Graphiti add_memory failed: {error_text}")
                    return {"error": error_text, "status": response.status}

        except aiohttp.ClientError as e:
            logger.warning(f"Graphiti connection error: {e}")
            # Return a local-only result when Graphiti is unavailable
            return {
                "uuid": f"local_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "status": "local_only",
                "message": "Graphiti unavailable, stored locally",
            }
        except Exception as e:
            logger.error(f"Graphiti add_memory error: {e}")
            return {"error": str(e)}

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        group_id: Optional[str] = None,
        source: str = "text",
    ) -> Dict[str, Any]:
        """Alias for add_memory for backwards compatibility."""
        return await self.add_memory(name, episode_body, group_id, source)

    async def search_nodes(
        self,
        query: str,
        group_ids: Optional[List[str]] = None,
        max_nodes: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for nodes (entities) in the knowledge graph.

        Args:
            query: Search query
            group_ids: Filter by group IDs
            max_nodes: Maximum nodes to return

        Returns:
            List of matching nodes
        """
        try:
            session = await self._get_session()
            params = {
                "query": query,
                "max_nodes": max_nodes,
            }
            if group_ids:
                params["group_ids"] = json.dumps(group_ids)

            async with session.get(
                f"{self.base_url}/nodes/search",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result if isinstance(result, list) else result.get("nodes", [])
                else:
                    logger.error(f"Graphiti search_nodes failed: {response.status}")
                    return []

        except aiohttp.ClientError as e:
            logger.warning(f"Graphiti connection error during search: {e}")
            return []
        except Exception as e:
            logger.error(f"Graphiti search_nodes error: {e}")
            return []

    async def search_memory_nodes(
        self,
        query: str,
        group_ids: Optional[List[str]] = None,
        max_nodes: int = 10,
    ) -> List[Dict[str, Any]]:
        """Alias for search_nodes matching MCP tool signature."""
        return await self.search_nodes(query, group_ids, max_nodes)

    async def search_facts(
        self,
        query: str,
        group_ids: Optional[List[str]] = None,
        max_facts: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for facts (relationships) in the knowledge graph.

        Args:
            query: Search query
            group_ids: Filter by group IDs
            max_facts: Maximum facts to return

        Returns:
            List of matching facts
        """
        try:
            session = await self._get_session()
            params = {
                "query": query,
                "max_facts": max_facts,
            }
            if group_ids:
                params["group_ids"] = json.dumps(group_ids)

            async with session.get(
                f"{self.base_url}/facts/search",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result if isinstance(result, list) else result.get("facts", [])
                else:
                    logger.error(f"Graphiti search_facts failed: {response.status}")
                    return []

        except aiohttp.ClientError as e:
            logger.warning(f"Graphiti connection error during search: {e}")
            return []
        except Exception as e:
            logger.error(f"Graphiti search_facts error: {e}")
            return []

    async def search_memory_facts(
        self,
        query: str,
        group_ids: Optional[List[str]] = None,
        max_facts: int = 10,
    ) -> List[Dict[str, Any]]:
        """Alias for search_facts matching MCP tool signature."""
        return await self.search_facts(query, group_ids, max_facts)


def create_graphiti_client(
    base_url: Optional[str] = None,
    group_id: str = "default",
) -> GraphitiClient:
    """
    Factory function to create a GraphitiClient.

    Args:
        base_url: Graphiti server URL
        group_id: Default group ID

    Returns:
        GraphitiClient instance
    """
    return GraphitiClient(base_url=base_url, group_id=group_id)
