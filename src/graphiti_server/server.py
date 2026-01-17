"""FastAPI REST server for local Graphiti-compatible API."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .storage import LocalGraphitiStorage

logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path(os.getenv("GRAPHITI_DB_PATH", str(Path.home() / ".claude" / "graphiti_local.db")))

# Initialize storage
storage = LocalGraphitiStorage(DB_PATH)

# FastAPI app
app = FastAPI(
    title="Local Graphiti Server",
    description="Local Graphiti-compatible REST API with SQLite storage",
    version="1.0.0",
)

# CORS middleware - SECURITY: Restrict to known origins
ALLOWED_ORIGINS = os.getenv(
    "GRAPHITI_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


# Request/Response Models
class EpisodeRequest(BaseModel):
    """Request model for adding an episode."""
    name: str
    episode_body: str
    group_id: str = "default"
    source: str = "text"
    source_description: Optional[str] = None


class EpisodeResponse(BaseModel):
    """Response model for episode creation."""
    uuid: str
    name: str
    nodes_created: int
    facts_created: int
    created_at: str


class NodeResponse(BaseModel):
    """Response model for a node."""
    uuid: str
    name: str
    summary: Optional[str]
    labels: str
    group_id: str
    episode_uuid: Optional[str]
    created_at: str


class FactResponse(BaseModel):
    """Response model for a fact."""
    uuid: str
    subject: str
    predicate: str
    object: str
    group_id: str
    episode_uuid: Optional[str]
    created_at: str
    episode_body: Optional[str] = None


class StatsResponse(BaseModel):
    """Response model for storage statistics."""
    episodes: int
    nodes: int
    facts: int
    unsynced: int


class SyncRequest(BaseModel):
    """Request model for sync operation."""
    max_episodes: int = 50


class SyncResponse(BaseModel):
    """Response model for sync operation."""
    synced: int
    failed: int
    errors: List[str]


# Endpoints
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "graphiti-local"}


@app.post("/episodes", response_model=EpisodeResponse)
async def add_episode(request: EpisodeRequest) -> Dict[str, Any]:
    """
    Add an episode and extract entities/facts.

    This is the main endpoint for storing memories/learnings.
    """
    try:
        result = storage.add_episode(
            name=request.name,
            body=request.episode_body,
            group_id=request.group_id,
            source=request.source,
            source_description=request.source_description,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to add episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nodes/search")
async def search_nodes(
    query: str = Query(..., description="Search query"),
    group_ids: Optional[str] = Query(None, description="Comma-separated group IDs"),
    max_nodes: int = Query(10, description="Maximum nodes to return"),
) -> List[Dict[str, Any]]:
    """
    Search for nodes matching the query.

    Searches node names and summaries.
    """
    try:
        parsed_group_ids = None
        if group_ids:
            # Handle JSON array or comma-separated
            if group_ids.startswith("["):
                import json
                parsed_group_ids = json.loads(group_ids)
            else:
                parsed_group_ids = [g.strip() for g in group_ids.split(",")]

        return storage.search_nodes(
            query=query,
            group_ids=parsed_group_ids,
            max_nodes=max_nodes,
        )
    except Exception as e:
        logger.error(f"Failed to search nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/facts/search")
async def search_facts(
    query: str = Query(..., description="Search query"),
    group_ids: Optional[str] = Query(None, description="Comma-separated group IDs"),
    max_facts: int = Query(10, description="Maximum facts to return"),
) -> List[Dict[str, Any]]:
    """
    Search for facts matching the query.

    Searches fact subjects, predicates, and objects.
    Returns facts with their associated episode bodies.
    """
    try:
        parsed_group_ids = None
        if group_ids:
            if group_ids.startswith("["):
                import json
                parsed_group_ids = json.loads(group_ids)
            else:
                parsed_group_ids = [g.strip() for g in group_ids.split(",")]

        return storage.search_facts(
            query=query,
            group_ids=parsed_group_ids,
            max_facts=max_facts,
        )
    except Exception as e:
        logger.error(f"Failed to search facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> Dict[str, int]:
    """Get storage statistics."""
    return storage.get_stats()


@app.post("/sync", response_model=SyncResponse)
async def trigger_sync(request: SyncRequest = SyncRequest()) -> Dict[str, Any]:
    """
    Trigger sync to MCP Graphiti backup.

    Syncs unsynced episodes to the remote Graphiti MCP server.
    """
    from .sync import GraphitiSyncService

    try:
        sync_service = GraphitiSyncService(storage)
        result = await sync_service.sync_to_mcp(max_episodes=request.max_episodes)
        return result
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return {"synced": 0, "failed": 0, "errors": [str(e)]}


@app.get("/episodes/unsynced")
async def get_unsynced_episodes(
    limit: int = Query(100, description="Maximum episodes to return"),
) -> List[Dict[str, Any]]:
    """Get episodes that haven't been synced to MCP Graphiti."""
    return storage.get_unsynced_episodes(limit=limit)


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the server using uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
