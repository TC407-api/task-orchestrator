"""
Perplexity Runner - Research via Perplexity API.

Uses Perplexity's sonar models for web-grounded research queries.
Returns synthesized answers with citations.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv(".env.local")

logger = logging.getLogger(__name__)


class PerplexityRunner:
    """
    Executes research queries via Perplexity API.

    Perplexity returns AI-synthesized answers with citations,
    making it ideal for research summarization in one call.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Perplexity runner.

        Args:
            api_key: Perplexity API key. Defaults to PERPLEXITY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai"
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def research_topic(
        self,
        topic: str,
        model: str = "sonar",
        search_recency: str = "week",
    ) -> dict:
        """
        Research a topic via Perplexity.

        Args:
            topic: Research query/topic
            model: Perplexity model (sonar, sonar-pro, sonar-reasoning)
            search_recency: How recent (day, week, month, year)

        Returns:
            Dict with answer, citations, and metadata
        """
        if not self.api_key:
            logger.warning("No Perplexity API key configured")
            return {
                "topic": topic,
                "answer": "",
                "citations": [],
                "error": "No API key",
            }

        session = await self._ensure_session()

        logger.info(f"Researching via Perplexity: {topic}")

        # Build research prompt
        system_prompt = """You are a research assistant. Provide a comprehensive but concise summary of the latest developments on the given topic. Include:
- Key recent news/updates
- Important trends or changes
- Relevant technical details
- Notable sources

Be factual and cite your sources."""

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Research the latest on: {topic}"},
            ],
            "search_recency_filter": search_recency,
            "return_citations": True,
            "return_related_questions": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_response(topic, data)
                else:
                    error_text = await response.text()
                    logger.error(f"Perplexity error ({response.status}): {error_text}")
                    return {
                        "topic": topic,
                        "answer": "",
                        "citations": [],
                        "error": f"API error: {response.status}",
                    }

        except asyncio.TimeoutError:
            logger.error(f"Perplexity timeout for '{topic}'")
            return {"topic": topic, "answer": "", "citations": [], "error": "Timeout"}
        except Exception as e:
            logger.error(f"Perplexity error for '{topic}': {e}")
            return {"topic": topic, "answer": "", "citations": [], "error": str(e)}

    def _parse_response(self, topic: str, data: dict) -> dict:
        """Parse Perplexity API response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        return {
            "topic": topic,
            "answer": message.get("content", ""),
            "citations": data.get("citations", []),
            "related_questions": data.get("related_questions", []),
            "model": data.get("model", ""),
            "usage": data.get("usage", {}),
            "timestamp": datetime.now().isoformat(),
        }

    async def research_multiple(
        self,
        topics: list[str],
        model: str = "sonar",
        search_recency: str = "week",
    ) -> list[dict]:
        """
        Research multiple topics in parallel.

        Args:
            topics: List of topics to research
            model: Perplexity model
            search_recency: Recency filter

        Returns:
            List of research results
        """
        tasks = [
            self.research_topic(topic, model, search_recency)
            for topic in topics
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
