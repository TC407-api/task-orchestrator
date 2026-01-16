"""Learning MCP Tools - Automated cross-project learning workflow."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .server import TaskOrchestratorMCP
    from ..evaluation.immune_system.graphiti_client import GraphitiClient

logger = logging.getLogger(__name__)

# Graphiti group ID for learning data
LEARNING_GROUP_ID = "cross_project_learning"

LEARNING_TOOLS = [
    {
        "name": "learning_workflow",
        "description": "Automated learning: extract patterns from projects, apply to current context, or recall from knowledge graph. Uses Gemini agents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["extract", "apply", "recall"],
                    "description": "extract=from projects, apply=to current, recall=query graph"
                },
                "topic": {
                    "type": "string",
                    "description": "Topic (e.g., 'authentication', 'error-handling')"
                },
                "projects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Project filter (empty=all)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results for recall (default: 10)"
                }
            },
            "required": ["operation", "topic"]
        }
    }
]

# JSON prompts for agents
EXTRACT_PROMPT = {
    "system": (
        "Pattern extraction specialist. Output ONLY valid JSON: "
        '{"learnings":[{"type":"pattern|anti-pattern","name":"snake_case",'
        '"summary":"<100chars","context":"when to use","implementation":"how",'
        '"tags":[],"confidence":0.0-1.0}]}'
    ),
    "task": "Extract '{topic}' patterns from {project_name} ({stack}). Max 5 learnings."
}

APPLY_PROMPT = {
    "system": (
        "Architecture advisor. Output ONLY valid JSON: "
        '{"recommendations":[{"source":"project","pattern":"name",'
        '"applicability":"HIGH|MED|LOW","steps":[]}],"anti_patterns":[],'
        '"synthesis":"summary"}'
    ),
    "task": "Apply '{topic}' learnings to current project. Learnings: {learnings}"
}

RECALL_PROMPT = {
    "system": (
        "Knowledge retrieval. Output ONLY valid JSON: "
        '{"summary":"answer","sources":[{"project":"name","relevance":0.0-1.0,'
        '"content":"insight"}],"related":[]}'
    ),
    "task": "Synthesize knowledge about '{topic}'. Facts: {facts}, Nodes: {nodes}"
}


class LearningToolHandler:
    """Handles automated learning workflow."""

    def __init__(self, server: "TaskOrchestratorMCP"):
        self.server = server
        self._portfolio_path = Path.home() / ".claude" / "portfolio.json"
        self._graphiti_client: Optional["GraphitiClient"] = None

    def _get_graphiti_client(self) -> "GraphitiClient":
        """Lazy-load Graphiti client."""
        if self._graphiti_client is None:
            from ..evaluation.immune_system.graphiti_client import GraphitiClient
            self._graphiti_client = GraphitiClient(group_id=LEARNING_GROUP_ID)
        return self._graphiti_client

    async def handle_tool(self, name: str, args: dict) -> dict:
        """Route to the appropriate handler method."""
        if name != "learning_workflow":
            return {"error": f"Unknown learning tool: {name}"}
        return await self.handle(args)

    async def handle(self, args: dict) -> dict:
        """Handle the learning_workflow tool call."""
        op = args.get("operation")
        if op == "extract":
            return await self._extract(args)
        elif op == "apply":
            return await self._apply(args)
        elif op == "recall":
            return await self._recall(args)
        return {"error": f"Unknown operation: {op}"}

    async def _extract(self, args: dict) -> dict:
        """Extract patterns from projects."""
        topic = args["topic"]
        projects = args.get("projects", [])

        # Load portfolio, filter projects
        portfolio = self._load_portfolio()
        if not portfolio:
            return {"error": "Could not load portfolio.json"}

        targets = self._get_projects(portfolio, projects)
        if not targets:
            return {"error": "No projects found matching filter"}

        # Build prompts for each project
        prompts = [
            EXTRACT_PROMPT["task"].format(
                topic=topic,
                project_name=p["name"],
                stack=",".join(p.get("stack", []))
            ) for p in targets
        ]

        # Spawn parallel Gemini agents
        result = await self.server._handle_spawn_parallel_agents({
            "prompts": prompts,
            "model": "gemini-3-flash-preview",
            "system_prompt": EXTRACT_PROMPT["system"]
        })

        # Parse results, store in Graphiti, update portfolio
        learnings = self._parse_extractions(result, topic, targets)
        storage_result = await self._store_graphiti(learnings, topic)
        self._update_portfolio(targets)

        return {
            "operation": "extract",
            "topic": topic,
            "count": len(learnings),
            "learnings": learnings,
            "graphiti": storage_result,
        }

    async def _apply(self, args: dict) -> dict:
        """Apply learnings to current context."""
        topic = args["topic"]

        # Search Graphiti for learnings
        facts = await self._search_graphiti(topic)

        # Spawn architect agent
        prompt = APPLY_PROMPT["task"].format(
            topic=topic,
            learnings=json.dumps(facts[:20])
        )
        result = await self.server._handle_spawn_archetype_agent({
            "archetype": "architect",
            "prompt": prompt
        })

        return {
            "operation": "apply",
            "topic": topic,
            "result": self._parse_json(result.get("response", "{}"))
        }

    async def _recall(self, args: dict) -> dict:
        """Recall learnings from knowledge graph."""
        topic = args["topic"]
        max_results = args.get("max_results", 10)

        # Search both facts and nodes
        facts = await self._search_graphiti(topic)

        # Spawn researcher agent for synthesis
        prompt = RECALL_PROMPT["task"].format(
            topic=topic,
            facts=json.dumps(facts[:10]),
            nodes="[]"
        )
        result = await self.server._handle_spawn_archetype_agent({
            "archetype": "researcher",
            "prompt": prompt
        })

        return {
            "operation": "recall",
            "topic": topic,
            "result": self._parse_json(result.get("response", "{}")),
            "raw": facts[:max_results]
        }

    # Helper methods
    def _load_portfolio(self) -> dict:
        """Load portfolio.json safely."""
        try:
            if self._portfolio_path.exists():
                return json.loads(self._portfolio_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
        return {}

    def _get_projects(self, portfolio: dict, filter_names: list) -> list:
        """Get projects from portfolio, optionally filtering by name."""
        all_projects = (
            portfolio.get("products", []) +
            portfolio.get("internalTools", []) +
            portfolio.get("personalProjects", [])
        )
        if filter_names:
            return [p for p in all_projects if p.get("name") in filter_names]
        return all_projects

    def _parse_extractions(self, result: dict, topic: str, targets: list) -> list:
        """Parse extraction results from parallel agents."""
        learnings = []
        for i, r in enumerate(result.get("results", [])):
            if r.get("success"):
                parsed = self._parse_json(r.get("response", "{}"))
                for learning in parsed.get("learnings", []):
                    learning["project"] = targets[i]["name"] if i < len(targets) else "unknown"
                    learning["topic"] = topic
                    learnings.append(learning)
        return learnings

    def _parse_json(self, text: str) -> dict:
        """Parse JSON safely."""
        try:
            return json.loads(text)
        except Exception:
            return {}

    async def _store_graphiti(self, learnings: list, topic: str) -> Dict[str, Any]:
        """Store learnings in Graphiti knowledge graph."""
        client = self._get_graphiti_client()
        stored_count = 0
        errors = []

        for i, learning in enumerate(learnings):
            project = learning.get('project', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            name = f"learning-{project}-{topic}-{timestamp}-{i}"

            # Create structured episode body
            episode_body = json.dumps({
                "type": "learning",
                "topic": topic,
                "project": project,
                "learning_type": learning.get("type", "pattern"),
                "name": learning.get("name", "unnamed"),
                "summary": learning.get("summary", ""),
                "context": learning.get("context", ""),
                "implementation": learning.get("implementation", ""),
                "tags": learning.get("tags", []),
                "confidence": learning.get("confidence", 0.5),
                "extracted_at": datetime.now().isoformat(),
            })

            try:
                result = await client.add_memory(
                    name=name,
                    episode_body=episode_body,
                    group_id=LEARNING_GROUP_ID,
                    source="json",
                    source_description=f"learning_extraction_{topic}",
                )

                if "error" not in result:
                    stored_count += 1
                    logger.info(f"Stored learning in Graphiti: {name}")
                else:
                    errors.append(f"{name}: {result.get('error')}")
                    logger.warning(f"Failed to store learning {name}: {result}")

            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                logger.error(f"Error storing learning {name}: {e}")

        return {
            "stored": stored_count,
            "total": len(learnings),
            "errors": errors if errors else None,
        }

    async def _search_graphiti(self, topic: str) -> list:
        """Search Graphiti for relevant facts and nodes."""
        client = self._get_graphiti_client()
        results = []

        try:
            # Search for facts related to the topic
            facts = await client.search_memory_facts(
                query=f"learning {topic}",
                group_ids=[LEARNING_GROUP_ID],
                max_facts=20,
            )

            # Parse and normalize results
            for fact in facts:
                try:
                    # Facts might have structured content
                    content = fact.get("content") or fact.get("fact") or str(fact)
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            results.append(parsed)
                        except json.JSONDecodeError:
                            results.append({"content": content, "raw": True})
                    else:
                        results.append(content)
                except Exception as e:
                    logger.debug(f"Error parsing fact: {e}")
                    results.append(fact)

            # Also search nodes for additional context
            nodes = await client.search_memory_nodes(
                query=f"learning {topic}",
                group_ids=[LEARNING_GROUP_ID],
                max_nodes=10,
            )

            for node in nodes:
                try:
                    name = node.get("name", "")
                    if "learning" in name.lower() or topic.lower() in name.lower():
                        results.append({
                            "type": "node",
                            "name": name,
                            "content": node.get("content", node.get("summary", "")),
                        })
                except Exception as e:
                    logger.debug(f"Error parsing node: {e}")

            logger.info(f"Found {len(results)} results for topic '{topic}'")

        except Exception as e:
            logger.error(f"Graphiti search error for '{topic}': {e}")

        return results

    def _update_portfolio(self, targets: list) -> None:
        """Update portfolio with last learning extraction date."""
        try:
            portfolio = self._load_portfolio()
            if not portfolio:
                return

            today = datetime.now().strftime("%Y-%m-%d")
            target_names = {t["name"] for t in targets}

            for section in ["products", "internalTools", "personalProjects"]:
                for project in portfolio.get(section, []):
                    if project.get("name") in target_names:
                        project["lastLearningExtracted"] = today

            self._portfolio_path.write_text(json.dumps(portfolio, indent=2))
        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")
