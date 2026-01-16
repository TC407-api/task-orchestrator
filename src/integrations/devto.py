"""
Dev.to API integration.

Provides functionality for publishing articles to Dev.to.
Requires Dev.to API key from https://dev.to/settings/extensions.

Usage:
    client = DevToClient(api_key=os.getenv("DEVTO_API_KEY"))
    article = await client.create_article(
        title="My Article",
        body_markdown="# Hello World",
        tags=["python", "tutorial"],
    )
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

from ..core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Dev.to API endpoint
DEVTO_API_BASE = "https://dev.to/api"


@dataclass
class DevToArticle:
    """Represents a Dev.to article."""
    id: int
    title: str
    description: str = ""
    body_markdown: str = ""
    published: bool = False
    tags: list[str] = field(default_factory=list)
    url: str = ""
    canonical_url: str = ""
    cover_image: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    published_at: Optional[datetime] = None
    reading_time_minutes: int = 0
    positive_reactions_count: int = 0
    comments_count: int = 0

    @classmethod
    def from_api_response(cls, data: dict) -> "DevToArticle":
        """Create DevToArticle from API response."""
        published_at = None
        if data.get("published_at"):
            try:
                published_at = datetime.fromisoformat(
                    data["published_at"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return cls(
            id=data.get("id", 0),
            title=data.get("title", ""),
            description=data.get("description", ""),
            body_markdown=data.get("body_markdown", ""),
            published=data.get("published", False),
            tags=data.get("tag_list", data.get("tags", [])),
            url=data.get("url", ""),
            canonical_url=data.get("canonical_url", ""),
            cover_image=data.get("cover_image", ""),
            published_at=published_at,
            reading_time_minutes=data.get("reading_time_minutes", 0),
            positive_reactions_count=data.get("positive_reactions_count", 0),
            comments_count=data.get("comments_count", 0),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "published": self.published,
            "tags": self.tags,
            "url": self.url,
            "created_at": self.created_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "reading_time_minutes": self.reading_time_minutes,
            "positive_reactions_count": self.positive_reactions_count,
            "comments_count": self.comments_count,
        }


class DevToClient:
    """
    Client for Dev.to API operations.

    Uses API key authentication.
    API key can be obtained from https://dev.to/settings/extensions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 30,  # requests per 30 seconds
    ):
        """
        Initialize the Dev.to client.

        Args:
            api_key: Dev.to API key
            rate_limit: Rate limit (requests per period)
        """
        self.api_key = api_key or os.getenv("DEVTO_API_KEY", "")
        self.rate_limiter = RateLimiter(rate_limit)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if the client is configured with credentials."""
        return bool(self.api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=DEVTO_API_BASE,
                timeout=30.0,
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def create_article(
        self,
        title: str,
        body_markdown: str,
        published: bool = False,
        tags: Optional[list[str]] = None,
        series: Optional[str] = None,
        canonical_url: Optional[str] = None,
        description: Optional[str] = None,
        cover_image: Optional[str] = None,
    ) -> DevToArticle:
        """
        Create a new article on Dev.to.

        Args:
            title: Article title
            body_markdown: Article body in Markdown format
            published: Whether to publish immediately (default: False = draft)
            tags: List of tags (max 4)
            series: Series name if part of a series
            canonical_url: Original source URL if republishing
            description: Article description for SEO
            cover_image: URL to cover image

        Returns:
            Created DevToArticle object

        Raises:
            ValueError: If more than 4 tags provided
        """
        tags = tags or []
        if len(tags) > 4:
            logger.warning(f"Dev.to allows max 4 tags, truncating from {len(tags)}")
            tags = tags[:4]

        if not self.is_configured:
            logger.warning("Dev.to client not configured, using dry-run mode")
            return DevToArticle(
                id=int(datetime.now().timestamp()),
                title=title,
                body_markdown=body_markdown,
                published=published,
                tags=tags,
            )

        self.rate_limiter.wait()

        # Build article payload
        article_data = {
            "title": title,
            "body_markdown": body_markdown,
            "published": published,
            "tags": tags,
        }

        if series:
            article_data["series"] = series
        if canonical_url:
            article_data["canonical_url"] = canonical_url
        if description:
            article_data["description"] = description
        if cover_image:
            article_data["cover_image"] = cover_image

        payload = {"article": article_data}

        client = await self._get_client()
        response = await client.post("/articles", json=payload)
        response.raise_for_status()

        data = response.json()
        article = DevToArticle.from_api_response(data)

        logger.info(f"Created Dev.to article: {article.id} - {article.title}")
        return article

    async def update_article(
        self,
        article_id: int,
        title: Optional[str] = None,
        body_markdown: Optional[str] = None,
        published: Optional[bool] = None,
        tags: Optional[list[str]] = None,
    ) -> DevToArticle:
        """
        Update an existing article.

        Args:
            article_id: ID of article to update
            title: New title (optional)
            body_markdown: New body (optional)
            published: New published status (optional)
            tags: New tags (optional)

        Returns:
            Updated DevToArticle object
        """
        if not self.is_configured:
            logger.warning("Dev.to client not configured, using dry-run mode")
            return DevToArticle(
                id=article_id,
                title=title or "",
                body_markdown=body_markdown or "",
            )

        self.rate_limiter.wait()

        article_data = {}
        if title is not None:
            article_data["title"] = title
        if body_markdown is not None:
            article_data["body_markdown"] = body_markdown
        if published is not None:
            article_data["published"] = published
        if tags is not None:
            article_data["tags"] = tags[:4]

        payload = {"article": article_data}

        client = await self._get_client()
        response = await client.put(f"/articles/{article_id}", json=payload)
        response.raise_for_status()

        data = response.json()
        return DevToArticle.from_api_response(data)

    async def get_article(self, article_id: int) -> DevToArticle:
        """
        Get an article by ID.

        Args:
            article_id: Article ID to retrieve

        Returns:
            DevToArticle object
        """
        if not self.is_configured:
            raise ValueError("Dev.to client not configured")

        self.rate_limiter.wait()

        client = await self._get_client()
        response = await client.get(f"/articles/{article_id}")
        response.raise_for_status()

        data = response.json()
        return DevToArticle.from_api_response(data)

    async def list_articles(
        self,
        page: int = 1,
        per_page: int = 30,
        state: str = "all",
    ) -> list[DevToArticle]:
        """
        List user's articles.

        Args:
            page: Page number
            per_page: Articles per page (max 1000)
            state: Filter by state (all, published, unpublished)

        Returns:
            List of DevToArticle objects
        """
        if not self.is_configured:
            return []

        self.rate_limiter.wait()

        client = await self._get_client()
        response = await client.get(
            "/articles/me",
            params={
                "page": page,
                "per_page": min(per_page, 1000),
                "state": state,
            },
        )
        response.raise_for_status()

        data = response.json()
        return [DevToArticle.from_api_response(article) for article in data]

    async def delete_article(self, article_id: int) -> bool:
        """
        Delete an article (unpublish).

        Note: Dev.to API doesn't support full deletion, only unpublishing.

        Args:
            article_id: ID of article to unpublish

        Returns:
            True if unpublished successfully
        """
        if not self.is_configured:
            logger.warning("Dev.to client not configured, skipping delete")
            return True

        # Dev.to doesn't have a delete endpoint, we unpublish instead
        await self.update_article(article_id, published=False)
        return True

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
