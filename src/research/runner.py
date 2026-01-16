"""
Research Runner - Firecrawl search execution.

Executes web searches for research topics using Firecrawl MCP server.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv(".env.local")

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result."""
    url: str
    title: str
    description: str
    content: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ResearchRunner:
    """
    Executes Firecrawl searches for research topics.

    Uses Firecrawl API for web search and content extraction.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize research runner.

        Args:
            api_key: Firecrawl API key. Defaults to FIRECRAWL_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        self.base_url = "https://api.firecrawl.dev/v1"
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def search_topic(
        self,
        topic: str,
        limit: int = 5,
        scrape_content: bool = True,
    ) -> list[dict]:
        """
        Search for a topic via Firecrawl.

        Args:
            topic: Search query / topic string
            limit: Maximum results to return
            scrape_content: Whether to scrape content from results

        Returns:
            List of search result dicts
        """
        if not self.api_key:
            logger.warning("No Firecrawl API key configured")
            return []

        session = await self._ensure_session()

        logger.info(f"Searching: {topic}")

        payload = {
            "query": topic,
            "limit": limit,
        }

        if scrape_content:
            payload["scrapeOptions"] = {
                "formats": ["markdown"],
                "onlyMainContent": True,
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                f"{self.base_url}/search",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = self._parse_search_results(data)
                    logger.info(f"Found {len(results)} results for '{topic}'")
                    return results
                else:
                    error_text = await response.text()
                    logger.error(f"Search failed ({response.status}): {error_text}")
                    return []

        except asyncio.TimeoutError:
            logger.error(f"Search timeout for '{topic}'")
            return []
        except Exception as e:
            logger.error(f"Search error for '{topic}': {e}")
            return []

    def _parse_search_results(self, data: dict) -> list[dict]:
        """Parse Firecrawl search response."""
        results = []

        # Handle different response formats
        items = data.get("data", data.get("results", []))

        for item in items:
            result = {
                "url": item.get("url", ""),
                "title": item.get("title", item.get("metadata", {}).get("title", "")),
                "description": item.get("description", item.get("metadata", {}).get("description", "")),
                "content": item.get("markdown", item.get("content", "")),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)

        return results

    async def fetch_url_content(self, url: str) -> dict:
        """
        Fetch content from a specific URL.

        Args:
            url: URL to scrape

        Returns:
            Dict with url, content, and metadata
        """
        if not self.api_key:
            logger.warning("No Firecrawl API key configured")
            return {"url": url, "content": "", "error": "No API key"}

        session = await self._ensure_session()

        logger.info(f"Fetching: {url}")

        payload = {
            "url": url,
            "formats": ["markdown", "links"],
            "onlyMainContent": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                f"{self.base_url}/scrape",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "url": url,
                        "content": data.get("data", {}).get("markdown", ""),
                        "links": data.get("data", {}).get("links", []),
                        "title": data.get("data", {}).get("metadata", {}).get("title", ""),
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Fetch failed ({response.status}): {error_text}")
                    return {"url": url, "content": "", "error": f"Status {response.status}"}

        except asyncio.TimeoutError:
            logger.error(f"Fetch timeout for {url}")
            return {"url": url, "content": "", "error": "Timeout"}
        except Exception as e:
            logger.error(f"Fetch error for {url}: {e}")
            return {"url": url, "content": "", "error": str(e)}

    async def fetch_multiple_urls(self, urls: list[str]) -> list[dict]:
        """
        Fetch content from multiple URLs in parallel.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of content dicts
        """
        tasks = [self.fetch_url_content(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append({
                    "url": urls[i],
                    "content": "",
                    "error": str(result),
                })
            else:
                processed.append(result)

        return processed

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
