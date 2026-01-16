"""
LinkedIn API integration.

Provides functionality for posting content to LinkedIn.
Requires LinkedIn OAuth 2.0 credentials with posting permissions.

Usage:
    client = LinkedInClient(
        client_id=os.getenv("LINKEDIN_CLIENT_ID"),
        client_secret=os.getenv("LINKEDIN_CLIENT_SECRET"),
        access_token=os.getenv("LINKEDIN_ACCESS_TOKEN"),
    )

    post = await client.create_post("Hello LinkedIn!")
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

from ..core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# LinkedIn API endpoints
LINKEDIN_API_BASE = "https://api.linkedin.com/v2"


@dataclass
class LinkedInPost:
    """Represents a LinkedIn post."""
    id: str
    text: str
    created_at: datetime = field(default_factory=datetime.now)
    author: str = ""
    visibility: str = "PUBLIC"
    lifecycle_state: str = "PUBLISHED"
    share_url: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: dict) -> "LinkedInPost":
        """Create LinkedInPost from API response."""
        return cls(
            id=data.get("id", ""),
            text=data.get("specificContent", {})
            .get("com.linkedin.ugc.ShareContent", {})
            .get("shareCommentary", {})
            .get("text", ""),
            author=data.get("author", ""),
            visibility=data.get("visibility", {}).get("com.linkedin.ugc.MemberNetworkVisibility", "PUBLIC"),
            lifecycle_state=data.get("lifecycleState", "PUBLISHED"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "created_at": self.created_at.isoformat(),
            "author": self.author,
            "visibility": self.visibility,
            "lifecycle_state": self.lifecycle_state,
            "share_url": self.share_url,
        }


@dataclass
class LinkedInProfile:
    """Represents a LinkedIn user profile."""
    id: str
    first_name: str = ""
    last_name: str = ""
    headline: str = ""
    profile_url: str = ""

    @property
    def full_name(self) -> str:
        """Get full name."""
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def urn(self) -> str:
        """Get LinkedIn URN for the profile."""
        return f"urn:li:person:{self.id}"


class LinkedInClient:
    """
    Client for LinkedIn API operations.

    Supports OAuth 2.0 for authentication.
    Requires scopes: w_member_social, r_liteprofile
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        rate_limit: int = 100,  # posts per day
    ):
        """
        Initialize the LinkedIn client.

        Args:
            client_id: LinkedIn OAuth Client ID
            client_secret: LinkedIn OAuth Client Secret
            access_token: OAuth Access Token
            rate_limit: Rate limit (requests per period)
        """
        self.client_id = client_id or os.getenv("LINKEDIN_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("LINKEDIN_CLIENT_SECRET", "")
        self.access_token = access_token or os.getenv("LINKEDIN_ACCESS_TOKEN", "")

        self.rate_limiter = RateLimiter(rate_limit)
        self._client: Optional[httpx.AsyncClient] = None
        self._profile: Optional[LinkedInProfile] = None

    @property
    def is_configured(self) -> bool:
        """Check if the client is configured with credentials."""
        return bool(self.access_token)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=LINKEDIN_API_BASE,
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                    "X-Restli-Protocol-Version": "2.0.0",
                },
            )
        return self._client

    async def get_profile(self) -> LinkedInProfile:
        """
        Get the authenticated user's profile.

        Returns:
            LinkedInProfile object
        """
        if self._profile:
            return self._profile

        if not self.is_configured:
            return LinkedInProfile(id="dry-run-user")

        self.rate_limiter.wait()

        client = await self._get_client()
        response = await client.get("/me")
        response.raise_for_status()

        data = response.json()
        self._profile = LinkedInProfile(
            id=data.get("id", ""),
            first_name=data.get("firstName", {}).get("localized", {}).get("en_US", ""),
            last_name=data.get("lastName", {}).get("localized", {}).get("en_US", ""),
        )
        return self._profile

    async def create_post(
        self,
        text: str,
        visibility: str = "PUBLIC",
        link_url: Optional[str] = None,
        link_title: Optional[str] = None,
        link_description: Optional[str] = None,
    ) -> LinkedInPost:
        """
        Create a new LinkedIn post.

        Args:
            text: Post text (max 3000 chars)
            visibility: Post visibility (PUBLIC, CONNECTIONS)
            link_url: Optional URL to share
            link_title: Title for the shared link
            link_description: Description for the shared link

        Returns:
            Created LinkedInPost object

        Raises:
            ValueError: If text exceeds 3000 characters
        """
        if len(text) > 3000:
            raise ValueError(f"Post exceeds 3000 chars: {len(text)}")

        if not self.is_configured:
            logger.warning("LinkedIn client not configured, using dry-run mode")
            return LinkedInPost(
                id=f"dry-run-{datetime.now().timestamp()}",
                text=text,
                visibility=visibility,
            )

        self.rate_limiter.wait()

        # Get user profile for author URN
        profile = await self.get_profile()

        # Build post payload (UGC Posts API)
        payload = {
            "author": profile.urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": visibility,
            },
        }

        # Add link if provided
        if link_url:
            payload["specificContent"]["com.linkedin.ugc.ShareContent"]["shareMediaCategory"] = "ARTICLE"
            payload["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [
                {
                    "status": "READY",
                    "originalUrl": link_url,
                    "title": {"text": link_title or ""},
                    "description": {"text": link_description or ""},
                }
            ]

        client = await self._get_client()
        response = await client.post("/ugcPosts", json=payload)
        response.raise_for_status()

        data = response.json()
        post = LinkedInPost.from_api_response(data)
        post.author = profile.urn

        logger.info(f"Created LinkedIn post: {post.id}")
        return post

    async def delete_post(self, post_id: str) -> bool:
        """
        Delete a LinkedIn post.

        Args:
            post_id: ID of post to delete

        Returns:
            True if deleted successfully
        """
        if not self.is_configured:
            logger.warning("LinkedIn client not configured, skipping delete")
            return True

        self.rate_limiter.wait()

        client = await self._get_client()

        # LinkedIn uses URN format for deletion
        if not post_id.startswith("urn:"):
            post_id = f"urn:li:share:{post_id}"

        response = await client.delete(f"/ugcPosts/{post_id}")
        response.raise_for_status()

        return response.status_code == 204

    async def get_post(self, post_id: str) -> LinkedInPost:
        """
        Get a post by ID.

        Args:
            post_id: Post ID to retrieve

        Returns:
            LinkedInPost object
        """
        if not self.is_configured:
            raise ValueError("LinkedIn client not configured")

        self.rate_limiter.wait()

        client = await self._get_client()

        if not post_id.startswith("urn:"):
            post_id = f"urn:li:share:{post_id}"

        response = await client.get(f"/ugcPosts/{post_id}")
        response.raise_for_status()

        data = response.json()
        return LinkedInPost.from_api_response(data)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
