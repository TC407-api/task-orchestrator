"""
Research Summarizer - AI summarization and Graphiti storage.

Uses Gemini to summarize research findings and stores them in the
Graphiti knowledge graph for later retrieval.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import google.generativeai as genai  # type: ignore[import]
from dotenv import load_dotenv

load_dotenv(".env.local")

logger = logging.getLogger(__name__)

# Summarization prompt template
SUMMARIZE_PROMPT = """Summarize the following research findings for the topic: "{topic}"

Research Results:
{results}

Provide a concise summary (3-5 bullet points) covering:
1. Key findings and insights
2. Notable trends or developments
3. Actionable information

Format as markdown bullet points. Be specific and include relevant details.
"""


class ResearchSummarizer:
    """
    Summarizes research and stores in Graphiti.

    Uses Gemini for AI-powered summarization.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        """
        Initialize summarizer.

        Args:
            model: Gemini model to use for summarization
            api_key: Google API key. Defaults to GOOGLE_API_KEY env var.
        """
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if self.api_key:
            genai.configure(api_key=self.api_key)  # type: ignore[attr-defined]
            self._genai_model = genai.GenerativeModel(model)  # type: ignore[attr-defined]
        else:
            self._genai_model = None
            logger.warning("No Google API key configured - summarization disabled")

    async def summarize(
        self,
        topic: str,
        results: list[dict],
        max_content_per_result: int = 1000,
    ) -> str:
        """
        Use Gemini to create concise summary of research results.

        Args:
            topic: Research topic
            results: List of search result dicts with url, title, content
            max_content_per_result: Max chars of content per result

        Returns:
            Markdown summary string
        """
        if not self._genai_model:
            return self._fallback_summary(topic, results)

        # Format results for prompt
        results_text = self._format_results(results, max_content_per_result)

        prompt = SUMMARIZE_PROMPT.format(topic=topic, results=results_text)

        try:
            response = self._genai_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 0.3,
                },
            )

            summary = response.text.strip()
            logger.info(f"Generated summary for '{topic}': {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Summarization error for '{topic}': {e}")
            return self._fallback_summary(topic, results)

    def _format_results(
        self,
        results: list[dict],
        max_content: int,
    ) -> str:
        """Format results for the summarization prompt."""
        formatted = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")[:max_content]
            description = result.get("description", "")

            text = f"### Result {i}: {title}\n"
            text += f"URL: {url}\n"
            if description:
                text += f"Description: {description}\n"
            if content:
                text += f"Content:\n{content}\n"

            formatted.append(text)

        return "\n---\n".join(formatted)

    def _fallback_summary(self, topic: str, results: list[dict]) -> str:
        """Generate fallback summary without AI."""
        summary_parts = [f"## Research: {topic}\n"]

        for result in results[:5]:  # Limit to top 5
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            description = result.get("description", "")

            summary_parts.append(f"- **{title}**")
            if description:
                summary_parts.append(f"  - {description[:200]}")
            if url:
                summary_parts.append(f"  - [Link]({url})")

        return "\n".join(summary_parts)

    async def store_in_graphiti(
        self,
        topic: str,
        summary: str,
        group_id: str = "auto_research",
    ) -> str:
        """
        Store research summary in Graphiti knowledge graph.

        Args:
            topic: Research topic
            summary: Summary to store
            group_id: Graphiti group for organizing research

        Returns:
            Episode UUID from Graphiti
        """
        try:
            # Import graphiti client
            from ..evaluation.immune_system.graphiti_client import GraphitiClient

            client = GraphitiClient()

            # Create episode body
            episode_body = f"""# Research: {topic}

Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}

{summary}
"""

            # Add to Graphiti
            episode = await client.add_episode(
                name=f"research_{topic.replace(' ', '_')[:50]}",
                episode_body=episode_body,
                group_id=group_id,
                source="auto_research",
            )

            uuid = episode.get("uuid", "unknown")
            logger.info(f"Stored research in Graphiti: {uuid}")
            return uuid

        except ImportError:
            logger.warning("Graphiti client not available - storing locally only")
            return "local_only"
        except Exception as e:
            logger.error(f"Graphiti storage error: {e}")
            return f"error: {e}"

    async def search_past_research(
        self,
        query: str,
        group_id: str = "auto_research",
        max_results: int = 10,
    ) -> list[dict]:
        """
        Search past research in Graphiti.

        Args:
            query: Search query
            group_id: Graphiti group to search
            max_results: Maximum results

        Returns:
            List of matching research entries
        """
        try:
            from ..evaluation.immune_system.graphiti_client import GraphitiClient

            client = GraphitiClient()
            results = await client.search_nodes(
                query=query,
                group_ids=[group_id],
                max_nodes=max_results,
            )
            return results

        except ImportError:
            logger.warning("Graphiti client not available")
            return []
        except Exception as e:
            logger.error(f"Graphiti search error: {e}")
            return []
