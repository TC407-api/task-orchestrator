"""
Twitter/X API v2 integration.

Provides functionality for posting tweets and threads.
Requires Twitter API v2 OAuth 2.0 credentials.

Usage:
    client = TwitterClient(
        api_key=os.getenv("TWITTER_API_KEY"),
        api_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_secret=os.getenv("TWITTER_ACCESS_SECRET"),
    )

    tweet = await client.post_tweet("Hello World!")
    thread = await client.post_thread(["Part 1", "Part 2", "Part 3"])
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

from ..core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Twitter API v2 endpoints
TWITTER_API_BASE = "https://api.twitter.com/2"
TWITTER_UPLOAD_BASE = "https://upload.twitter.com/1.1"


@dataclass
class Tweet:
    """Represents a posted tweet."""
    id: str
    text: str
    created_at: datetime = field(default_factory=datetime.now)
    author_id: Optional[str] = None
    conversation_id: Optional[str] = None
    in_reply_to_id: Optional[str] = None
    edit_history_tweet_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, data: dict) -> "Tweet":
        """Create Tweet from API response."""
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            author_id=data.get("author_id"),
            conversation_id=data.get("conversation_id"),
            edit_history_tweet_ids=data.get("edit_history_tweet_ids", []),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "created_at": self.created_at.isoformat(),
            "author_id": self.author_id,
            "conversation_id": self.conversation_id,
            "in_reply_to_id": self.in_reply_to_id,
        }


@dataclass
class TwitterThread:
    """Represents a Twitter thread (multiple tweets)."""
    tweets: list[Tweet] = field(default_factory=list)
    conversation_id: Optional[str] = None

    @property
    def first_tweet(self) -> Optional[Tweet]:
        """Get the first tweet in the thread."""
        return self.tweets[0] if self.tweets else None

    @property
    def last_tweet(self) -> Optional[Tweet]:
        """Get the last tweet in the thread."""
        return self.tweets[-1] if self.tweets else None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tweets": [t.to_dict() for t in self.tweets],
            "conversation_id": self.conversation_id,
            "tweet_count": len(self.tweets),
        }


class TwitterClient:
    """
    Client for Twitter API v2 operations.

    Supports OAuth 1.0a User Context for posting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
        rate_limit: int = 300,  # tweets per 3-hour window
    ):
        """
        Initialize the Twitter client.

        Args:
            api_key: Twitter API Key (Consumer Key)
            api_secret: Twitter API Secret (Consumer Secret)
            access_token: OAuth Access Token
            access_secret: OAuth Access Token Secret
            rate_limit: Rate limit (requests per period)
        """
        self.api_key = api_key or os.getenv("TWITTER_API_KEY", "")
        self.api_secret = api_secret or os.getenv("TWITTER_API_SECRET", "")
        self.access_token = access_token or os.getenv("TWITTER_ACCESS_TOKEN", "")
        self.access_secret = access_secret or os.getenv("TWITTER_ACCESS_SECRET", "")

        self.rate_limiter = RateLimiter(rate_limit)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if the client is configured with credentials."""
        return all([
            self.api_key,
            self.api_secret,
            self.access_token,
            self.access_secret,
        ])

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=TWITTER_API_BASE,
                timeout=30.0,
            )
        return self._client

    def _get_oauth1_header(self, method: str, url: str) -> dict[str, str]:
        """
        Generate OAuth 1.0a authorization header.

        Note: In production, use a proper OAuth library like authlib or tweepy.
        This is a simplified placeholder for the structure.
        """
        # In production, implement proper OAuth 1.0a signature
        # For now, we use Bearer token auth which requires App-only context
        # or implement OAuth 1.0a properly with libraries like authlib
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def post_tweet(
        self,
        text: str,
        reply_to: Optional[str] = None,
        quote_tweet_id: Optional[str] = None,
    ) -> Tweet:
        """
        Post a single tweet.

        Args:
            text: Tweet text (max 280 chars)
            reply_to: Tweet ID to reply to (optional)
            quote_tweet_id: Tweet ID to quote (optional)

        Returns:
            Posted Tweet object

        Raises:
            ValueError: If text exceeds 280 characters
            httpx.HTTPError: On API errors
        """
        if len(text) > 280:
            raise ValueError(f"Tweet exceeds 280 chars: {len(text)}")

        if not self.is_configured:
            logger.warning("Twitter client not configured, using dry-run mode")
            return Tweet(
                id=f"dry-run-{datetime.now().timestamp()}",
                text=text,
                in_reply_to_id=reply_to,
            )

        self.rate_limiter.wait()

        payload = {"text": text}

        if reply_to:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to}

        if quote_tweet_id:
            payload["quote_tweet_id"] = quote_tweet_id

        client = await self._get_client()
        headers = self._get_oauth1_header("POST", f"{TWITTER_API_BASE}/tweets")

        response = await client.post("/tweets", json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        return Tweet.from_api_response(data.get("data", {}))

    async def post_thread(
        self,
        tweets: list[str],
    ) -> TwitterThread:
        """
        Post a thread of tweets.

        Args:
            tweets: List of tweet texts

        Returns:
            TwitterThread with all posted tweets

        Raises:
            ValueError: If any tweet exceeds 280 characters
        """
        if not tweets:
            raise ValueError("Cannot post empty thread")

        # Validate all tweets first
        for i, text in enumerate(tweets):
            if len(text) > 280:
                raise ValueError(f"Tweet {i + 1} exceeds 280 chars: {len(text)}")

        thread = TwitterThread()

        # Post first tweet
        first_tweet = await self.post_tweet(tweets[0])
        thread.tweets.append(first_tweet)
        thread.conversation_id = first_tweet.id

        # Post remaining tweets as replies
        previous_id = first_tweet.id
        for text in tweets[1:]:
            tweet = await self.post_tweet(text, reply_to=previous_id)
            tweet.conversation_id = thread.conversation_id
            thread.tweets.append(tweet)
            previous_id = tweet.id

        logger.info(f"Posted thread with {len(thread.tweets)} tweets")
        return thread

    async def delete_tweet(self, tweet_id: str) -> bool:
        """
        Delete a tweet.

        Args:
            tweet_id: ID of tweet to delete

        Returns:
            True if deleted successfully
        """
        if not self.is_configured:
            logger.warning("Twitter client not configured, skipping delete")
            return True

        self.rate_limiter.wait()

        client = await self._get_client()
        headers = self._get_oauth1_header("DELETE", f"{TWITTER_API_BASE}/tweets/{tweet_id}")

        response = await client.delete(f"/tweets/{tweet_id}", headers=headers)
        response.raise_for_status()

        data = response.json()
        return data.get("data", {}).get("deleted", False)

    async def get_tweet(self, tweet_id: str) -> Tweet:
        """
        Get a tweet by ID.

        Args:
            tweet_id: Tweet ID to retrieve

        Returns:
            Tweet object
        """
        if not self.is_configured:
            raise ValueError("Twitter client not configured")

        self.rate_limiter.wait()

        client = await self._get_client()
        headers = self._get_oauth1_header("GET", f"{TWITTER_API_BASE}/tweets/{tweet_id}")

        response = await client.get(
            f"/tweets/{tweet_id}",
            headers=headers,
            params={"tweet.fields": "created_at,author_id,conversation_id"},
        )
        response.raise_for_status()

        data = response.json()
        return Tweet.from_api_response(data.get("data", {}))

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
