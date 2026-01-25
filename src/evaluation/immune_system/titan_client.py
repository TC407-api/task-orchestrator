"""
Titan Memory Client - Wrapper for Titan Memory operations.

Provides a unified interface for storing and retrieving failure patterns
from Titan Memory, replacing the Graphiti backend for the immune system.
"""

import asyncio
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Titan CLI path
TITAN_CLI = os.path.expanduser("~/.claude/titan-memory/dist/cli/index.js")
NODE_PATH = "node"

# Group/project ID for immune system failures
IMMUNE_PROJECT_ID = "task-orchestrator-immune"

# Map layer numbers to CLI names
LAYER_NAMES = {
    2: "factual",
    3: "longterm",
    4: "semantic",
    5: "episodic",
}


class TitanClient:
    """
    Client for interacting with Titan Memory system.

    Uses the Titan CLI for storage operations, providing:
    - Failure pattern storage with tags
    - Semantic search with tag filtering
    - Utility tracking via feedback signals
    """

    def __init__(
        self,
        project_id: str = IMMUNE_PROJECT_ID,
        cli_path: Optional[str] = None,
    ):
        """
        Initialize Titan client.

        Args:
            project_id: Default project ID for organizing immune system data.
            cli_path: Path to Titan CLI. Defaults to ~/.claude/titan-memory/dist/cli/index.js
        """
        self.project_id = project_id
        self.cli_path = cli_path or TITAN_CLI
        self._available = os.path.exists(self.cli_path)

        if not self._available:
            logger.warning(f"Titan CLI not found at {self.cli_path}")

    def _run_cli(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """
        Run Titan CLI command synchronously.

        Args:
            args: CLI arguments
            timeout: Command timeout in seconds

        Returns:
            Dict with command result
        """
        if not self._available:
            return {"error": "Titan CLI not available", "stored": False}

        cmd = [NODE_PATH, self.cli_path] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
            )

            output = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode != 0:
                logger.error(f"Titan CLI error: {stderr}")
                return {"error": stderr, "stored": False}

            # Parse output
            if "Memory added:" in output:
                # Extract memory ID
                parts = output.split("Memory added:")
                if len(parts) > 1:
                    memory_id = parts[1].strip().split()[0]
                    return {"id": memory_id, "stored": True, "output": output}

            if "Found" in output and "memories" in output:
                # Parse search results
                return self._parse_search_output(output)

            if "Feedback recorded" in output:
                return {"success": True, "output": output}

            return {"output": output, "stored": True}

        except subprocess.TimeoutExpired:
            logger.error(f"Titan CLI timeout after {timeout}s")
            return {"error": "timeout", "stored": False}
        except Exception as e:
            logger.error(f"Titan CLI error: {e}")
            return {"error": str(e), "stored": False}

    def _parse_search_output(self, output: str) -> Dict[str, Any]:
        """Parse search output from Titan CLI."""
        results: List[Dict[str, Any]] = []
        lines = output.split("\n")

        # Look for memory entries in output
        current_memory: Dict[str, Any] = {}
        for line in lines:
            line = line.strip()
            if line.startswith("ID:"):
                if current_memory:
                    results.append(current_memory)
                current_memory = {"id": line.split("ID:")[1].strip()}
            elif line.startswith("Content:"):
                current_memory["content"] = line.split("Content:", 1)[1].strip()
            elif line.startswith("Score:"):
                try:
                    current_memory["score"] = float(line.split("Score:")[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Tags:"):
                current_memory["tags"] = line.split("Tags:")[1].strip()

        if current_memory:
            results.append(current_memory)

        return {"memories": results, "count": len(results)}

    async def add_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        layer: int = 5,
    ) -> Dict[str, Any]:
        """
        Add a memory/failure pattern to Titan.

        Args:
            content: Content of the memory
            tags: Tags for categorization
            project_id: Project ID (defaults to immune project)
            layer: Memory layer (5 = episodic, best for failures)

        Returns:
            Dict with memory ID and status
        """
        project = project_id or self.project_id

        args = ["add", content]

        if tags:
            args.extend(["--tags", ",".join(tags)])

        args.extend(["--project", project])

        # Convert layer number to name if needed
        layer_name = LAYER_NAMES.get(layer, "episodic")
        args.extend(["--layer", layer_name])

        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self._run_cli(args))

        return result

    async def search_memories(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for memories/failure patterns.

        Args:
            query: Search query
            tags: Filter by tags
            project_id: Filter by project
            limit: Maximum results

        Returns:
            List of matching memories
        """
        args = ["recall", query]

        if tags:
            args.extend(["--tags", ",".join(tags)])

        if project_id:
            args.extend(["--project", project_id])

        args.extend(["--limit", str(limit)])

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self._run_cli(args))

        return result.get("memories", [])

    async def record_feedback(
        self,
        memory_id: str,
        signal: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record feedback on a memory's utility.

        Args:
            memory_id: ID of the memory
            signal: "helpful" or "harmful"
            context: Optional context for the feedback
            session_id: Session ID for idempotency

        Returns:
            Dict with feedback result
        """
        args = ["feedback", memory_id, "--signal", signal]

        if context:
            args.extend(["--context", context])

        if session_id:
            args.extend(["--session", session_id])

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self._run_cli(args))

        return result

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory dict or None
        """
        args = ["get", memory_id, "--json"]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self._run_cli(args))

        if "error" in result:
            return None

        return result

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted successfully
        """
        args = ["delete", memory_id, "--yes"]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self._run_cli(args))

        return "error" not in result

    def is_available(self) -> bool:
        """Check if Titan CLI is available."""
        return self._available


def create_titan_client(
    project_id: str = IMMUNE_PROJECT_ID,
) -> TitanClient:
    """
    Factory function to create a TitanClient.

    Args:
        project_id: Project ID for organizing data

    Returns:
        TitanClient instance
    """
    return TitanClient(project_id=project_id)


__all__ = [
    "TitanClient",
    "create_titan_client",
    "IMMUNE_PROJECT_ID",
]
