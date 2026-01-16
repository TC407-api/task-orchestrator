"""
Content Generator for Multi-Platform Marketing.

Generates platform-specific content from source material using AI agents.
Respects platform constraints (character limits, formatting requirements).

Usage:
    generator = ContentGenerator()
    campaign = await generator.generate_campaign(
        topic="AI Agent Observability",
        source_content="Long article...",
        platforms=["twitter", "linkedin", "devto", "reddit", "hackernews"],
    )
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


class Platform(str, Enum):
    """Supported content platforms."""
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    DEVTO = "devto"
    REDDIT = "reddit"
    HACKERNEWS = "hackernews"


@dataclass
class PlatformConstraints:
    """Constraints for a specific platform."""
    max_length: int
    format: str  # "thread", "single", "markdown", "plain"
    supports_hashtags: bool = True
    supports_links: bool = True
    supports_images: bool = True
    best_posting_time: str = "09:00"  # EST


# Platform constraint definitions
PLATFORM_CONSTRAINTS: dict[Platform, PlatformConstraints] = {
    Platform.TWITTER: PlatformConstraints(
        max_length=280,
        format="thread",
        supports_hashtags=True,
        supports_links=True,
        supports_images=True,
        best_posting_time="09:00",
    ),
    Platform.LINKEDIN: PlatformConstraints(
        max_length=3000,
        format="single",
        supports_hashtags=True,
        supports_links=True,
        supports_images=True,
        best_posting_time="09:00",
    ),
    Platform.DEVTO: PlatformConstraints(
        max_length=65535,  # Effectively unlimited
        format="markdown",
        supports_hashtags=True,  # Tags
        supports_links=True,
        supports_images=True,
        best_posting_time="10:00",
    ),
    Platform.REDDIT: PlatformConstraints(
        max_length=40000,
        format="markdown",
        supports_hashtags=False,
        supports_links=True,
        supports_images=True,
        best_posting_time="11:00",
    ),
    Platform.HACKERNEWS: PlatformConstraints(
        max_length=2000,
        format="plain",
        supports_hashtags=False,
        supports_links=True,
        supports_images=False,
        best_posting_time="09:00",
    ),
}


@dataclass
class PlatformContent:
    """Content generated for a specific platform."""
    platform: Platform
    content: str
    title: Optional[str] = None
    hashtags: list[str] = field(default_factory=list)
    thread_parts: list[str] = field(default_factory=list)  # For Twitter threads
    tags: list[str] = field(default_factory=list)  # For Dev.to
    subreddit: Optional[str] = None  # For Reddit
    link_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_within_limits(self) -> bool:
        """Check if content is within platform limits."""
        constraints = PLATFORM_CONSTRAINTS[self.platform]
        if self.platform == Platform.TWITTER:
            # Check each thread part
            return all(len(part) <= constraints.max_length for part in self.thread_parts)
        return len(self.content) <= constraints.max_length

    @property
    def character_count(self) -> int:
        """Get total character count."""
        if self.platform == Platform.TWITTER:
            return sum(len(part) for part in self.thread_parts)
        return len(self.content)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "platform": self.platform.value,
            "content": self.content,
            "title": self.title,
            "hashtags": self.hashtags,
            "thread_parts": self.thread_parts,
            "tags": self.tags,
            "subreddit": self.subreddit,
            "link_url": self.link_url,
            "character_count": self.character_count,
            "is_within_limits": self.is_within_limits,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Campaign:
    """A content marketing campaign across multiple platforms."""
    campaign_id: str = field(default_factory=lambda: str(uuid4())[:8])
    topic: str = ""
    source_content: str = ""
    platform_content: dict[Platform, PlatformContent] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "draft"  # draft, scheduled, published, completed

    @property
    def platforms(self) -> list[Platform]:
        """Get list of platforms in this campaign."""
        return list(self.platform_content.keys())

    def get_content(self, platform: Platform) -> Optional[PlatformContent]:
        """Get content for a specific platform."""
        return self.platform_content.get(platform)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "topic": self.topic,
            "platforms": [p.value for p in self.platforms],
            "platform_content": {
                p.value: c.to_dict() for p, c in self.platform_content.items()
            },
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class ContentGenerator:
    """
    Generates platform-specific content from source material.

    Uses Jinja2 templates for consistent formatting and Gemini agents
    for AI-powered content adaptation.
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        use_ai: bool = True,
    ):
        """
        Initialize the content generator.

        Args:
            template_dir: Path to Jinja2 templates. Defaults to src/content/templates.
            use_ai: Whether to use AI for content generation. Set False for testing.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = template_dir
        self.use_ai = use_ai

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self.env.filters["truncate_smart"] = self._truncate_smart
        self.env.filters["format_hashtags"] = self._format_hashtags

    @staticmethod
    def _truncate_smart(text: str, length: int, suffix: str = "...") -> str:
        """Truncate text at word boundary."""
        if len(text) <= length:
            return text

        # Find last space before length limit
        truncated = text[: length - len(suffix)]
        last_space = truncated.rfind(" ")

        if last_space > length // 2:
            truncated = truncated[:last_space]

        return truncated.rstrip() + suffix

    @staticmethod
    def _format_hashtags(tags: list[str], prefix: str = "#") -> str:
        """Format tags as hashtags."""
        return " ".join(f"{prefix}{tag.replace(' ', '')}" for tag in tags)

    async def generate_campaign(
        self,
        topic: str,
        source_content: str,
        platforms: list[str],
        hashtags: Optional[list[str]] = None,
        link_url: Optional[str] = None,
        subreddit: Optional[str] = None,
    ) -> Campaign:
        """
        Generate a full marketing campaign across multiple platforms.

        Args:
            topic: Main topic/headline for the campaign
            source_content: Source material to adapt
            platforms: List of platform names (e.g., ["twitter", "linkedin"])
            hashtags: Optional list of hashtags to include
            link_url: Optional URL to include in posts
            subreddit: Optional subreddit for Reddit posts

        Returns:
            Campaign object with generated content for each platform
        """
        campaign = Campaign(
            topic=topic,
            source_content=source_content,
        )

        hashtags = hashtags or []

        for platform_name in platforms:
            try:
                platform = Platform(platform_name.lower())
                content = await self.generate_for_platform(
                    topic=topic,
                    source_content=source_content,
                    platform=platform,
                    hashtags=hashtags,
                    link_url=link_url,
                    subreddit=subreddit if platform == Platform.REDDIT else None,
                )
                campaign.platform_content[platform] = content
            except ValueError as e:
                logger.warning(f"Invalid platform '{platform_name}': {e}")
                continue

        return campaign

    async def generate_for_platform(
        self,
        topic: str,
        source_content: str,
        platform: Platform,
        hashtags: Optional[list[str]] = None,
        link_url: Optional[str] = None,
        subreddit: Optional[str] = None,
    ) -> PlatformContent:
        """
        Generate content for a specific platform.

        Args:
            topic: Main topic/headline
            source_content: Source material to adapt
            platform: Target platform
            hashtags: Optional hashtags
            link_url: Optional URL to include
            subreddit: Optional subreddit (for Reddit)

        Returns:
            PlatformContent with adapted content
        """
        constraints = PLATFORM_CONSTRAINTS[platform]
        hashtags = hashtags or []

        # Load and render template
        template_name = f"{platform.value}.jinja2"
        try:
            template = self.env.get_template(template_name)
        except Exception:
            # Fall back to generating without template
            logger.warning(f"Template not found for {platform.value}, using fallback")
            return await self._generate_fallback(
                topic, source_content, platform, hashtags, link_url, subreddit
            )

        # Prepare context for template
        context = {
            "topic": topic,
            "source_content": source_content,
            "hashtags": hashtags,
            "link_url": link_url,
            "subreddit": subreddit,
            "max_length": constraints.max_length,
            "supports_hashtags": constraints.supports_hashtags,
            "supports_links": constraints.supports_links,
        }

        if self.use_ai:
            # Use AI to generate optimized content
            context["ai_content"] = await self._generate_ai_content(
                topic, source_content, platform, constraints
            )

        rendered = template.render(**context)

        # Create platform content based on format
        if platform == Platform.TWITTER:
            return self._create_twitter_content(
                rendered, topic, hashtags, link_url
            )
        elif platform == Platform.DEVTO:
            return PlatformContent(
                platform=platform,
                content=rendered,
                title=topic,
                tags=hashtags[:4],  # Dev.to allows max 4 tags
                link_url=link_url,
            )
        elif platform == Platform.REDDIT:
            return PlatformContent(
                platform=platform,
                content=rendered,
                title=topic,
                subreddit=subreddit,
                link_url=link_url,
            )
        else:
            return PlatformContent(
                platform=platform,
                content=rendered,
                title=topic,
                hashtags=hashtags if constraints.supports_hashtags else [],
                link_url=link_url,
            )

    def _create_twitter_content(
        self,
        content: str,
        topic: str,
        hashtags: list[str],
        link_url: Optional[str],
    ) -> PlatformContent:
        """Create Twitter thread content."""
        # Split content into thread parts
        thread_parts = self._split_into_thread(content, max_length=280)

        return PlatformContent(
            platform=Platform.TWITTER,
            content=content,
            title=topic,
            hashtags=hashtags,
            thread_parts=thread_parts,
            link_url=link_url,
        )

    def _split_into_thread(
        self,
        content: str,
        max_length: int = 280,
        thread_indicator: bool = True,
    ) -> list[str]:
        """
        Split content into Twitter thread parts.

        Args:
            content: Full content to split
            max_length: Maximum length per tweet
            thread_indicator: Add (1/n) indicators

        Returns:
            List of tweet-sized strings
        """
        # Split by natural breaks first
        paragraphs = content.split("\n\n")
        parts = []

        for para in paragraphs:
            if not para.strip():
                continue

            # If paragraph fits, add it
            if len(para) <= max_length:
                parts.append(para.strip())
            else:
                # Split paragraph by sentences
                sentences = para.replace(". ", ".|").split("|")
                current = ""

                for sentence in sentences:
                    if len(current + sentence) <= max_length:
                        current += sentence
                    else:
                        if current:
                            parts.append(current.strip())
                        current = sentence

                if current:
                    parts.append(current.strip())

        # Add thread indicators if requested
        if thread_indicator and len(parts) > 1:
            total = len(parts)
            parts = [f"{part} ({i + 1}/{total})" for i, part in enumerate(parts)]

        return parts

    async def _generate_ai_content(
        self,
        topic: str,
        source_content: str,
        platform: Platform,
        constraints: PlatformConstraints,
    ) -> str:
        """
        Use AI to generate optimized content.

        This integrates with the existing spawn_agent infrastructure.
        """
        # For now, return a summary. In production, this would call Gemini.
        # The actual AI integration will use the existing spawn_agent pattern.
        summary = self._truncate_smart(source_content, constraints.max_length // 2)
        return f"{topic}\n\n{summary}"

    async def _generate_fallback(
        self,
        topic: str,
        source_content: str,
        platform: Platform,
        hashtags: list[str],
        link_url: Optional[str],
        subreddit: Optional[str],
    ) -> PlatformContent:
        """Generate content without template (fallback)."""
        constraints = PLATFORM_CONSTRAINTS[platform]

        # Simple truncation-based generation
        content = self._truncate_smart(source_content, constraints.max_length - 100)

        if constraints.supports_hashtags and hashtags:
            content += "\n\n" + self._format_hashtags(hashtags)

        if constraints.supports_links and link_url:
            content += f"\n\n{link_url}"

        if platform == Platform.TWITTER:
            return self._create_twitter_content(content, topic, hashtags, link_url)

        return PlatformContent(
            platform=platform,
            content=content,
            title=topic,
            hashtags=hashtags if constraints.supports_hashtags else [],
            tags=hashtags[:4] if platform == Platform.DEVTO else [],
            subreddit=subreddit if platform == Platform.REDDIT else None,
            link_url=link_url,
        )

    def validate_content(self, content: PlatformContent) -> tuple[bool, list[str]]:
        """
        Validate content against platform constraints.

        Args:
            content: PlatformContent to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        constraints = PLATFORM_CONSTRAINTS[content.platform]

        # Check length
        if not content.is_within_limits:
            if content.platform == Platform.TWITTER:
                for i, part in enumerate(content.thread_parts):
                    if len(part) > constraints.max_length:
                        issues.append(
                            f"Thread part {i + 1} exceeds {constraints.max_length} chars "
                            f"({len(part)} chars)"
                        )
            else:
                issues.append(
                    f"Content exceeds {constraints.max_length} chars "
                    f"({content.character_count} chars)"
                )

        # Check hashtags
        if content.hashtags and not constraints.supports_hashtags:
            issues.append(f"{content.platform.value} does not support hashtags")

        # Check empty content
        if not content.content.strip():
            issues.append("Content is empty")

        return len(issues) == 0, issues
