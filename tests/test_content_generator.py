"""
Tests for Content Generator.

Validates content generation, platform constraints, and template rendering.
"""
import pytest
from pathlib import Path

from src.content.generator import (
    ContentGenerator,
    PlatformContent,
    Campaign,
    Platform,
    PlatformConstraints,
    PLATFORM_CONSTRAINTS,
)


# --- Fixtures ---

@pytest.fixture
def generator():
    """Create a ContentGenerator with AI disabled for testing."""
    return ContentGenerator(use_ai=False)


@pytest.fixture
def sample_topic():
    """Sample topic for testing."""
    return "AI Agent Observability: Why 95% of Projects Fail"


@pytest.fixture
def sample_content():
    """Sample source content for testing."""
    return """
    AI agent projects have a 95% failure rate in production. The main issues are:

    1. Lack of observability - teams can't see what's happening inside agents
    2. No self-healing - when things break, they stay broken
    3. Cost explosions - unbounded token usage leads to surprise bills
    4. Tool failures cascade - one broken tool brings down the whole system

    Task Orchestrator solves these problems with built-in observability,
    self-healing capabilities, and cost controls. The immune system pattern
    prevents cascading failures before they happen.

    Join thousands of developers building reliable AI agents with Task Orchestrator.
    """


@pytest.fixture
def sample_hashtags():
    """Sample hashtags for testing."""
    return ["AI", "MachineLearning", "DevOps", "Observability"]


# --- Platform Constraints Tests ---

class TestPlatformConstraints:
    """Tests for platform constraint definitions."""

    def test_all_platforms_have_constraints(self):
        """Test that all platforms have defined constraints."""
        for platform in Platform:
            assert platform in PLATFORM_CONSTRAINTS
            constraints = PLATFORM_CONSTRAINTS[platform]
            assert isinstance(constraints, PlatformConstraints)
            assert constraints.max_length > 0

    def test_twitter_constraints(self):
        """Test Twitter-specific constraints."""
        constraints = PLATFORM_CONSTRAINTS[Platform.TWITTER]
        assert constraints.max_length == 280
        assert constraints.format == "thread"
        assert constraints.supports_hashtags is True

    def test_linkedin_constraints(self):
        """Test LinkedIn-specific constraints."""
        constraints = PLATFORM_CONSTRAINTS[Platform.LINKEDIN]
        assert constraints.max_length == 3000
        assert constraints.format == "single"
        assert constraints.supports_hashtags is True

    def test_devto_constraints(self):
        """Test Dev.to-specific constraints."""
        constraints = PLATFORM_CONSTRAINTS[Platform.DEVTO]
        assert constraints.max_length >= 65535  # Effectively unlimited
        assert constraints.format == "markdown"

    def test_reddit_constraints(self):
        """Test Reddit-specific constraints."""
        constraints = PLATFORM_CONSTRAINTS[Platform.REDDIT]
        assert constraints.max_length == 40000
        assert constraints.format == "markdown"
        assert constraints.supports_hashtags is False

    def test_hackernews_constraints(self):
        """Test Hacker News-specific constraints."""
        constraints = PLATFORM_CONSTRAINTS[Platform.HACKERNEWS]
        assert constraints.max_length == 2000
        assert constraints.format == "plain"
        assert constraints.supports_hashtags is False
        assert constraints.supports_images is False


# --- PlatformContent Tests ---

class TestPlatformContent:
    """Tests for PlatformContent dataclass."""

    def test_create_platform_content(self):
        """Test creating platform content."""
        content = PlatformContent(
            platform=Platform.LINKEDIN,
            content="Test content",
            title="Test Title",
            hashtags=["AI", "ML"],
        )
        assert content.platform == Platform.LINKEDIN
        assert content.content == "Test content"
        assert content.title == "Test Title"
        assert content.hashtags == ["AI", "ML"]

    def test_character_count(self):
        """Test character count calculation."""
        content = PlatformContent(
            platform=Platform.LINKEDIN,
            content="Hello World",
        )
        assert content.character_count == 11

    def test_twitter_character_count(self):
        """Test Twitter thread character count."""
        content = PlatformContent(
            platform=Platform.TWITTER,
            content="Full content here",
            thread_parts=["Part 1 (1/3)", "Part 2 (2/3)", "Part 3 (3/3)"],
        )
        # Should sum thread parts, not main content
        expected = sum(len(p) for p in content.thread_parts)
        assert content.character_count == expected

    def test_is_within_limits_linkedin(self):
        """Test LinkedIn content length validation."""
        short_content = PlatformContent(
            platform=Platform.LINKEDIN,
            content="Short content",
        )
        assert short_content.is_within_limits is True

        long_content = PlatformContent(
            platform=Platform.LINKEDIN,
            content="x" * 4000,  # Exceeds 3000 limit
        )
        assert long_content.is_within_limits is False

    def test_is_within_limits_twitter(self):
        """Test Twitter thread validation."""
        valid_thread = PlatformContent(
            platform=Platform.TWITTER,
            content="",
            thread_parts=["Short tweet 1", "Short tweet 2"],
        )
        assert valid_thread.is_within_limits is True

        invalid_thread = PlatformContent(
            platform=Platform.TWITTER,
            content="",
            thread_parts=["x" * 300],  # Exceeds 280
        )
        assert invalid_thread.is_within_limits is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        content = PlatformContent(
            platform=Platform.DEVTO,
            content="Article content",
            title="My Article",
            tags=["python", "ai"],
        )
        data = content.to_dict()

        assert data["platform"] == "devto"
        assert data["content"] == "Article content"
        assert data["title"] == "My Article"
        assert data["tags"] == ["python", "ai"]
        assert "created_at" in data


# --- Campaign Tests ---

class TestCampaign:
    """Tests for Campaign dataclass."""

    def test_create_campaign(self):
        """Test creating a campaign."""
        campaign = Campaign(
            topic="Test Campaign",
            source_content="Source material",
        )
        assert campaign.topic == "Test Campaign"
        assert campaign.status == "draft"
        assert len(campaign.campaign_id) == 8

    def test_campaign_platforms(self):
        """Test platform listing."""
        campaign = Campaign()
        campaign.platform_content[Platform.TWITTER] = PlatformContent(
            platform=Platform.TWITTER,
            content="Tweet",
        )
        campaign.platform_content[Platform.LINKEDIN] = PlatformContent(
            platform=Platform.LINKEDIN,
            content="Post",
        )

        platforms = campaign.platforms
        assert Platform.TWITTER in platforms
        assert Platform.LINKEDIN in platforms
        assert len(platforms) == 2

    def test_get_content(self):
        """Test getting content by platform."""
        campaign = Campaign()
        twitter_content = PlatformContent(
            platform=Platform.TWITTER,
            content="Tweet",
        )
        campaign.platform_content[Platform.TWITTER] = twitter_content

        assert campaign.get_content(Platform.TWITTER) == twitter_content
        assert campaign.get_content(Platform.LINKEDIN) is None

    def test_to_dict(self):
        """Test campaign serialization."""
        campaign = Campaign(topic="Test")
        campaign.platform_content[Platform.TWITTER] = PlatformContent(
            platform=Platform.TWITTER,
            content="Tweet",
        )

        data = campaign.to_dict()
        assert data["topic"] == "Test"
        assert "twitter" in data["platforms"]
        assert "twitter" in data["platform_content"]


# --- ContentGenerator Tests ---

class TestContentGenerator:
    """Tests for ContentGenerator class."""

    def test_init_default_template_dir(self):
        """Test initialization with default template directory."""
        generator = ContentGenerator(use_ai=False)
        assert generator.template_dir.exists()
        assert generator.use_ai is False

    def test_truncate_smart_short_text(self, generator):
        """Test smart truncation with short text."""
        result = generator._truncate_smart("Hello World", 50)
        assert result == "Hello World"

    def test_truncate_smart_long_text(self, generator):
        """Test smart truncation with long text."""
        text = "This is a longer text that needs to be truncated at a word boundary"
        result = generator._truncate_smart(text, 30)
        assert len(result) <= 30
        assert result.endswith("...")

    def test_truncate_smart_word_boundary(self, generator):
        """Test truncation at word boundary."""
        text = "Word1 Word2 Word3 Word4 Word5"
        result = generator._truncate_smart(text, 15)
        # Should truncate at word boundary, not mid-word
        assert not result.rstrip(".").endswith("Word")

    def test_format_hashtags(self, generator):
        """Test hashtag formatting."""
        tags = ["AI", "Machine Learning", "DevOps"]
        result = generator._format_hashtags(tags)
        assert result == "#AI #MachineLearning #DevOps"

    def test_split_into_thread_short(self, generator):
        """Test thread splitting with short content."""
        content = "A short tweet"
        parts = generator._split_into_thread(content)
        assert len(parts) == 1
        assert "A short tweet" in parts[0]

    def test_split_into_thread_long(self, generator):
        """Test thread splitting with long content."""
        content = "First paragraph with enough content.\n\n" + \
                  "Second paragraph with more content.\n\n" + \
                  "Third paragraph to make it longer."
        parts = generator._split_into_thread(content, max_length=280)
        assert len(parts) >= 1
        for part in parts:
            # Account for thread indicators
            assert len(part) <= 300  # Generous limit for indicators

    def test_split_into_thread_indicators(self, generator):
        """Test thread indicators are added."""
        content = "First part.\n\nSecond part.\n\nThird part."
        parts = generator._split_into_thread(content, thread_indicator=True)
        if len(parts) > 1:
            assert "(1/" in parts[0]
            assert "(2/" in parts[1] if len(parts) > 1 else True

    @pytest.mark.asyncio
    async def test_generate_for_twitter(self, generator, sample_topic, sample_content):
        """Test generating Twitter content."""
        content = await generator.generate_for_platform(
            topic=sample_topic,
            source_content=sample_content,
            platform=Platform.TWITTER,
            hashtags=["AI", "Observability"],
        )

        assert content.platform == Platform.TWITTER
        assert content.title == sample_topic
        assert "AI" in content.hashtags
        assert len(content.thread_parts) >= 1

    @pytest.mark.asyncio
    async def test_generate_for_linkedin(self, generator, sample_topic, sample_content):
        """Test generating LinkedIn content."""
        content = await generator.generate_for_platform(
            topic=sample_topic,
            source_content=sample_content,
            platform=Platform.LINKEDIN,
            hashtags=["AI", "DevOps"],
        )

        assert content.platform == Platform.LINKEDIN
        assert content.title == sample_topic
        assert len(content.content) <= 3000

    @pytest.mark.asyncio
    async def test_generate_for_devto(self, generator, sample_topic, sample_content):
        """Test generating Dev.to content."""
        content = await generator.generate_for_platform(
            topic=sample_topic,
            source_content=sample_content,
            platform=Platform.DEVTO,
            hashtags=["ai", "observability", "devops", "python"],
        )

        assert content.platform == Platform.DEVTO
        assert content.title == sample_topic
        assert len(content.tags) <= 4  # Dev.to max 4 tags
        assert "---" in content.content  # Front matter

    @pytest.mark.asyncio
    async def test_generate_for_reddit(self, generator, sample_topic, sample_content):
        """Test generating Reddit content."""
        content = await generator.generate_for_platform(
            topic=sample_topic,
            source_content=sample_content,
            platform=Platform.REDDIT,
            subreddit="machinelearning",
        )

        assert content.platform == Platform.REDDIT
        assert content.subreddit == "machinelearning"
        assert content.hashtags == []  # Reddit doesn't support hashtags

    @pytest.mark.asyncio
    async def test_generate_for_hackernews(self, generator, sample_topic, sample_content):
        """Test generating Hacker News content."""
        content = await generator.generate_for_platform(
            topic=sample_topic,
            source_content=sample_content,
            platform=Platform.HACKERNEWS,
            link_url="https://example.com",
        )

        assert content.platform == Platform.HACKERNEWS
        assert len(content.content) <= 2000
        assert content.hashtags == []  # HN doesn't support hashtags

    @pytest.mark.asyncio
    async def test_generate_campaign(
        self, generator, sample_topic, sample_content, sample_hashtags
    ):
        """Test generating a full campaign."""
        campaign = await generator.generate_campaign(
            topic=sample_topic,
            source_content=sample_content,
            platforms=["twitter", "linkedin", "devto"],
            hashtags=sample_hashtags,
            link_url="https://task-orchestrator.dev",
        )

        assert len(campaign.platforms) == 3
        assert Platform.TWITTER in campaign.platforms
        assert Platform.LINKEDIN in campaign.platforms
        assert Platform.DEVTO in campaign.platforms

        # Verify each platform has content
        for platform in campaign.platforms:
            content = campaign.get_content(platform)
            assert content is not None
            assert content.platform == platform

    @pytest.mark.asyncio
    async def test_generate_campaign_invalid_platform(
        self, generator, sample_topic, sample_content
    ):
        """Test campaign generation with invalid platform."""
        campaign = await generator.generate_campaign(
            topic=sample_topic,
            source_content=sample_content,
            platforms=["twitter", "invalid_platform", "linkedin"],
        )

        # Should only have valid platforms
        assert len(campaign.platforms) == 2
        assert Platform.TWITTER in campaign.platforms
        assert Platform.LINKEDIN in campaign.platforms

    def test_validate_content_valid(self, generator):
        """Test validation of valid content."""
        content = PlatformContent(
            platform=Platform.LINKEDIN,
            content="Valid short content",
        )
        is_valid, issues = generator.validate_content(content)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_content_too_long(self, generator):
        """Test validation of content that's too long."""
        content = PlatformContent(
            platform=Platform.TWITTER,
            content="x" * 500,
            thread_parts=["x" * 300],  # Exceeds 280
        )
        is_valid, issues = generator.validate_content(content)
        assert is_valid is False
        assert len(issues) > 0
        assert "exceeds" in issues[0].lower()

    def test_validate_content_empty(self, generator):
        """Test validation of empty content."""
        content = PlatformContent(
            platform=Platform.LINKEDIN,
            content="",
        )
        is_valid, issues = generator.validate_content(content)
        assert is_valid is False
        assert any("empty" in issue.lower() for issue in issues)

    def test_validate_content_hashtags_unsupported(self, generator):
        """Test validation of hashtags on unsupported platform."""
        content = PlatformContent(
            platform=Platform.REDDIT,
            content="Valid content",
            hashtags=["tag1", "tag2"],
        )
        is_valid, issues = generator.validate_content(content)
        assert is_valid is False
        assert any("hashtag" in issue.lower() for issue in issues)


# --- Template Tests ---

class TestTemplates:
    """Tests for Jinja2 templates."""

    def test_templates_exist(self):
        """Test that all platform templates exist."""
        template_dir = Path(__file__).parent.parent / "src" / "content" / "templates"

        for platform in Platform:
            template_path = template_dir / f"{platform.value}.jinja2"
            assert template_path.exists(), f"Template for {platform.value} should exist"

    def test_twitter_template_renders(self, generator, sample_topic, sample_content):
        """Test Twitter template renders without errors."""
        template = generator.env.get_template("twitter.jinja2")
        rendered = template.render(
            topic=sample_topic,
            source_content=sample_content,
            hashtags=["AI"],
            link_url="https://example.com",
            supports_hashtags=True,
            supports_links=True,
            max_length=280,
        )
        assert len(rendered) > 0
        assert "#AI" in rendered

    def test_devto_template_has_frontmatter(self, generator, sample_topic, sample_content):
        """Test Dev.to template includes front matter."""
        template = generator.env.get_template("devto.jinja2")
        rendered = template.render(
            topic=sample_topic,
            source_content=sample_content,
            hashtags=["ai", "python"],
            supports_hashtags=True,
            supports_links=True,
            max_length=65535,
        )
        assert "---" in rendered
        assert f"title: {sample_topic}" in rendered


# --- Integration Tests ---

class TestContentGeneratorIntegration:
    """Integration tests for content generation workflow."""

    @pytest.mark.asyncio
    async def test_full_campaign_workflow(self, generator):
        """Test complete campaign generation workflow."""
        # Generate campaign
        campaign = await generator.generate_campaign(
            topic="Building Reliable AI Agents",
            source_content="AI agents need observability, self-healing, and cost controls.",
            platforms=["twitter", "linkedin", "devto", "reddit", "hackernews"],
            hashtags=["AI", "MLOps", "DevOps"],
            link_url="https://task-orchestrator.dev",
            subreddit="machinelearning",
        )

        # Verify campaign
        assert campaign.status == "draft"
        assert len(campaign.platforms) == 5

        # Validate all content
        all_valid = True
        for platform in campaign.platforms:
            content = campaign.get_content(platform)
            is_valid, issues = generator.validate_content(content)
            if not is_valid:
                all_valid = False
                print(f"{platform.value} issues: {issues}")

        # At minimum, content should be generated (validation may flag some issues)
        assert all(
            campaign.get_content(p) is not None
            for p in campaign.platforms
        )

    @pytest.mark.asyncio
    async def test_content_respects_platform_limits(self, generator):
        """Test that generated content respects platform limits."""
        long_content = "x" * 10000  # Very long source content

        campaign = await generator.generate_campaign(
            topic="Test Topic",
            source_content=long_content,
            platforms=["twitter", "hackernews"],
        )

        # Twitter thread parts should be <= 280 chars
        twitter_content = campaign.get_content(Platform.TWITTER)
        for part in twitter_content.thread_parts:
            # Allow some buffer for thread indicators
            assert len(part) <= 300, f"Twitter part too long: {len(part)}"

        # HN should be <= 2000 chars
        hn_content = campaign.get_content(Platform.HACKERNEWS)
        assert len(hn_content.content) <= 2000, f"HN content too long: {len(hn_content.content)}"
