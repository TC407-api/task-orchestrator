"""Local storage backend for Graphiti-compatible server."""
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".claude" / "graphiti_local.db"

# SECURITY: Limit input sizes to prevent DoS/resource exhaustion
MAX_GROUP_IDS = 100
MAX_QUERY_LENGTH = 1000
MAX_RESULTS = 1000


class LocalGraphitiStorage:
    """
    SQLite-based local storage for Graphiti episodes and entities.

    Provides fast local access with optional sync to remote Graphiti MCP.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize storage with SQLite database."""
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    uuid TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    body TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    source TEXT DEFAULT 'text',
                    source_description TEXT,
                    created_at TEXT NOT NULL,
                    synced_at TEXT,
                    synced_to_mcp INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    uuid TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    summary TEXT,
                    labels TEXT DEFAULT '["Entity"]',
                    group_id TEXT NOT NULL,
                    episode_uuid TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_uuid) REFERENCES episodes(uuid)
                );

                CREATE TABLE IF NOT EXISTS facts (
                    uuid TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    episode_uuid TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_uuid) REFERENCES episodes(uuid)
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_group ON episodes(group_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_group ON nodes(group_id);
                CREATE INDEX IF NOT EXISTS idx_facts_group ON facts(group_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
                CREATE INDEX IF NOT EXISTS idx_episodes_synced ON episodes(synced_to_mcp);
            """)
            conn.commit()

    def add_episode(
        self,
        name: str,
        body: str,
        group_id: str,
        source: str = "text",
        source_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add an episode and extract entities/facts from it.

        Returns:
            Dict with uuid and created entities
        """
        episode_uuid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Insert episode
            conn.execute(
                """INSERT INTO episodes
                   (uuid, name, body, group_id, source, source_description, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (episode_uuid, name, body, group_id, source, source_description, now)
            )

            # Extract and store entities from body
            nodes = self._extract_nodes(body, group_id, episode_uuid, now, conn)
            facts = self._extract_facts(body, group_id, episode_uuid, now, conn)

            conn.commit()

        logger.info(f"Added episode {episode_uuid}: {len(nodes)} nodes, {len(facts)} facts")

        return {
            "uuid": episode_uuid,
            "name": name,
            "nodes_created": len(nodes),
            "facts_created": len(facts),
            "created_at": now,
        }

    def _extract_nodes(
        self,
        body: str,
        group_id: str,
        episode_uuid: str,
        created_at: str,
        conn: sqlite3.Connection,
    ) -> List[Dict]:
        """Extract entity nodes from episode body."""
        nodes = []

        try:
            # Try to parse as JSON to extract structured data
            data = json.loads(body)

            # Extract key fields as nodes
            if isinstance(data, dict):
                # Topic node
                if data.get("topic"):
                    node = self._create_node(
                        name=data["topic"],
                        summary=f"Learning topic: {data.get('summary', data['topic'])}",
                        group_id=group_id,
                        episode_uuid=episode_uuid,
                        created_at=created_at,
                        conn=conn,
                    )
                    nodes.append(node)

                # Project node
                if data.get("project"):
                    node = self._create_node(
                        name=data["project"],
                        summary="Source project for learning",
                        group_id=group_id,
                        episode_uuid=episode_uuid,
                        created_at=created_at,
                        conn=conn,
                    )
                    nodes.append(node)

                # Pattern name node
                if data.get("name"):
                    node = self._create_node(
                        name=data["name"],
                        summary=data.get("summary", f"Pattern: {data['name']}"),
                        group_id=group_id,
                        episode_uuid=episode_uuid,
                        created_at=created_at,
                        conn=conn,
                    )
                    nodes.append(node)

                # Tag nodes
                for tag in data.get("tags", []):
                    node = self._create_node(
                        name=tag,
                        summary=f"Tag: {tag}",
                        group_id=group_id,
                        episode_uuid=episode_uuid,
                        created_at=created_at,
                        conn=conn,
                    )
                    nodes.append(node)

        except json.JSONDecodeError:
            # For plain text, create a single node from the name
            pass

        return nodes

    def _create_node(
        self,
        name: str,
        summary: str,
        group_id: str,
        episode_uuid: str,
        created_at: str,
        conn: sqlite3.Connection,
    ) -> Dict:
        """Create a node in the database."""
        node_uuid = str(uuid.uuid4())
        conn.execute(
            """INSERT OR IGNORE INTO nodes
               (uuid, name, summary, group_id, episode_uuid, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (node_uuid, name, summary, group_id, episode_uuid, created_at)
        )
        return {"uuid": node_uuid, "name": name, "summary": summary}

    def _extract_facts(
        self,
        body: str,
        group_id: str,
        episode_uuid: str,
        created_at: str,
        conn: sqlite3.Connection,
    ) -> List[Dict]:
        """Extract facts (relationships) from episode body."""
        facts = []

        try:
            data = json.loads(body)

            if isinstance(data, dict):
                # Create fact: project -> has_pattern -> pattern_name
                if data.get("project") and data.get("name"):
                    fact = self._create_fact(
                        subject=data["project"],
                        predicate="has_pattern",
                        obj=data["name"],
                        group_id=group_id,
                        episode_uuid=episode_uuid,
                        created_at=created_at,
                        conn=conn,
                    )
                    facts.append(fact)

                # Create fact: pattern -> relates_to -> topic
                if data.get("name") and data.get("topic"):
                    fact = self._create_fact(
                        subject=data["name"],
                        predicate="relates_to",
                        obj=data["topic"],
                        group_id=group_id,
                        episode_uuid=episode_uuid,
                        created_at=created_at,
                        conn=conn,
                    )
                    facts.append(fact)

                # Create fact: pattern -> has_confidence -> confidence_value
                if data.get("name") and data.get("confidence"):
                    fact = self._create_fact(
                        subject=data["name"],
                        predicate="has_confidence",
                        obj=str(data["confidence"]),
                        group_id=group_id,
                        episode_uuid=episode_uuid,
                        created_at=created_at,
                        conn=conn,
                    )
                    facts.append(fact)

        except json.JSONDecodeError:
            pass

        return facts

    def _create_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        group_id: str,
        episode_uuid: str,
        created_at: str,
        conn: sqlite3.Connection,
    ) -> Dict:
        """Create a fact in the database."""
        fact_uuid = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO facts
               (uuid, subject, predicate, object, group_id, episode_uuid, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (fact_uuid, subject, predicate, obj, group_id, episode_uuid, created_at)
        )
        return {"uuid": fact_uuid, "subject": subject, "predicate": predicate, "object": obj}

    def search_nodes(
        self,
        query: str,
        group_ids: Optional[List[str]] = None,
        max_nodes: int = 10,
    ) -> List[Dict]:
        """Search for nodes matching query."""
        # SECURITY: Validate inputs to prevent DoS
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]
        if group_ids and len(group_ids) > MAX_GROUP_IDS:
            raise ValueError(f"Too many group_ids: {len(group_ids)} > {MAX_GROUP_IDS}")
        max_nodes = min(max_nodes, MAX_RESULTS)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if group_ids:
                placeholders = ",".join("?" * len(group_ids))
                sql = f"""
                    SELECT * FROM nodes
                    WHERE group_id IN ({placeholders})
                    AND (name LIKE ? OR summary LIKE ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                params = group_ids + [f"%{query}%", f"%{query}%", max_nodes]
            else:
                sql = """
                    SELECT * FROM nodes
                    WHERE name LIKE ? OR summary LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                params = [f"%{query}%", f"%{query}%", max_nodes]

            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def search_facts(
        self,
        query: str,
        group_ids: Optional[List[str]] = None,
        max_facts: int = 10,
    ) -> List[Dict]:
        """Search for facts matching query."""
        # SECURITY: Validate inputs to prevent DoS
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]
        if group_ids and len(group_ids) > MAX_GROUP_IDS:
            raise ValueError(f"Too many group_ids: {len(group_ids)} > {MAX_GROUP_IDS}")
        max_facts = min(max_facts, MAX_RESULTS)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if group_ids:
                placeholders = ",".join("?" * len(group_ids))
                sql = f"""
                    SELECT f.*, e.body as episode_body FROM facts f
                    LEFT JOIN episodes e ON f.episode_uuid = e.uuid
                    WHERE f.group_id IN ({placeholders})
                    AND (f.subject LIKE ? OR f.predicate LIKE ? OR f.object LIKE ?)
                    ORDER BY f.created_at DESC
                    LIMIT ?
                """
                params = group_ids + [f"%{query}%", f"%{query}%", f"%{query}%", max_facts]
            else:
                sql = """
                    SELECT f.*, e.body as episode_body FROM facts f
                    LEFT JOIN episodes e ON f.episode_uuid = e.uuid
                    WHERE f.subject LIKE ? OR f.predicate LIKE ? OR f.object LIKE ?
                    ORDER BY f.created_at DESC
                    LIMIT ?
                """
                params = [f"%{query}%", f"%{query}%", f"%{query}%", max_facts]

            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def get_unsynced_episodes(self, limit: int = 100) -> List[Dict]:
        """Get episodes that haven't been synced to MCP Graphiti."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM episodes
                   WHERE synced_to_mcp = 0
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_synced(self, episode_uuid: str) -> None:
        """Mark an episode as synced to MCP Graphiti."""
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE episodes SET synced_to_mcp = 1, synced_at = ? WHERE uuid = ?""",
                (now, episode_uuid)
            )
            conn.commit()

    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            unsynced = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE synced_to_mcp = 0"
            ).fetchone()[0]

        return {
            "episodes": episodes,
            "nodes": nodes,
            "facts": facts,
            "unsynced": unsynced,
        }
