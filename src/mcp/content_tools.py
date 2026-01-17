"""
MCP Tools for Content Automation.

Provides tools for generating and publishing content across platforms.
Integrates with the Content Automation System.
"""
import logging
from typing import Optional

from ..content import (
    ContentGenerator,
    ContentPublisher,
    Campaign,
    Platform,
    PublishingCalendar,
)
from ..agents.background_tasks import BackgroundTaskScheduler
from ..agents.inbox import UniversalInbox
from ..observability import trace_operation

logger = logging.getLogger(__name__)


# Tool definitions for get_tools()
CONTENT_TOOLS = [
    {
        "name": "content_generate",
        "description": "Generate platform-specific content for a marketing campaign from a topic and source content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Main topic/headline for the content",
                },
                "source_content": {
                    "type": "string",
                    "description": "Source material to adapt for each platform",
                },
                "platforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Platforms to generate for (twitter, linkedin, devto, reddit, hackernews)",
                },
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional hashtags to include",
                },
                "link_url": {
                    "type": "string",
                    "description": "Optional URL to include in posts",
                },
                "subreddit": {
                    "type": "string",
                    "description": "Optional subreddit for Reddit posts",
                },
            },
            "required": ["topic", "source_content", "platforms"],
        },
    },
    {
        "name": "content_schedule",
        "description": "Schedule a content campaign for publishing at optimal times.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "campaign_id": {
                    "type": "string",
                    "description": "ID of the campaign to schedule",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in ISO format (optional, defaults to now)",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone for scheduling (default: America/New_York)",
                },
            },
            "required": ["campaign_id"],
        },
    },
    {
        "name": "content_publish",
        "description": "Publish content to a specific platform immediately.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "campaign_id": {
                    "type": "string",
                    "description": "ID of the campaign",
                },
                "platform": {
                    "type": "string",
                    "enum": ["twitter", "linkedin", "devto", "reddit", "hackernews"],
                    "description": "Platform to publish to",
                },
            },
            "required": ["campaign_id", "platform"],
        },
    },
    {
        "name": "content_publish_all",
        "description": "Publish a campaign to all platforms immediately.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "campaign_id": {
                    "type": "string",
                    "description": "ID of the campaign to publish",
                },
            },
            "required": ["campaign_id"],
        },
    },
    {
        "name": "content_status",
        "description": "Check status of a content campaign.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "campaign_id": {
                    "type": "string",
                    "description": "ID of the campaign to check",
                },
            },
            "required": ["campaign_id"],
        },
    },
    {
        "name": "content_list_campaigns",
        "description": "List all content campaigns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["all", "draft", "scheduled", "published", "completed"],
                    "description": "Filter by status (default: all)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum campaigns to return (default: 10)",
                },
            },
        },
    },
    {
        "name": "content_cancel",
        "description": "Cancel a scheduled content campaign.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "campaign_id": {
                    "type": "string",
                    "description": "ID of the campaign to cancel",
                },
            },
            "required": ["campaign_id"],
        },
    },
]


class ContentToolsHandler:
    """
    Handler for content automation MCP tools.

    Manages ContentGenerator and ContentPublisher instances.
    """

    def __init__(
        self,
        scheduler: BackgroundTaskScheduler,
        inbox: Optional[UniversalInbox] = None,
    ):
        """
        Initialize the content tools handler.

        Args:
            scheduler: BackgroundTaskScheduler for scheduled publishing
            inbox: UniversalInbox for event publishing (optional)
        """
        self.generator = ContentGenerator(use_ai=True)
        self.publisher = ContentPublisher(scheduler, inbox)

        # Store campaigns by ID for reference
        self._campaigns: dict[str, Campaign] = {}

    @trace_operation("content_generate")
    async def handle_content_generate(self, args: dict) -> dict:
        """Generate content for multiple platforms."""
        topic = args.get("topic", "")
        source_content = args.get("source_content", "")
        platforms = args.get("platforms", [])
        hashtags = args.get("hashtags", [])
        link_url = args.get("link_url")
        subreddit = args.get("subreddit")

        if not topic or not source_content or not platforms:
            return {"error": "topic, source_content, and platforms are required"}

        campaign = await self.generator.generate_campaign(
            topic=topic,
            source_content=source_content,
            platforms=platforms,
            hashtags=hashtags,
            link_url=link_url,
            subreddit=subreddit,
        )

        # Store campaign for later reference
        self._campaigns[campaign.campaign_id] = campaign

        # Validate all content
        validation_issues = {}
        for platform in campaign.platforms:
            content = campaign.get_content(platform)
            is_valid, issues = self.generator.validate_content(content)
            if not is_valid:
                validation_issues[platform.value] = issues

        return {
            "campaign_id": campaign.campaign_id,
            "topic": campaign.topic,
            "platforms": [p.value for p in campaign.platforms],
            "status": campaign.status,
            "content_preview": {
                p.value: {
                    "character_count": c.character_count,
                    "is_within_limits": c.is_within_limits,
                    "preview": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                }
                for p, c in campaign.platform_content.items()
            },
            "validation_issues": validation_issues,
            "created_at": campaign.created_at.isoformat(),
        }

    @trace_operation("content_schedule")
    async def handle_content_schedule(self, args: dict) -> dict:
        """Schedule a campaign for publishing."""
        campaign_id = args.get("campaign_id", "")
        start_date_str = args.get("start_date")
        timezone = args.get("timezone", "America/New_York")

        campaign = self._campaigns.get(campaign_id)
        if not campaign:
            return {"error": f"Campaign not found: {campaign_id}"}

        from datetime import datetime
        from zoneinfo import ZoneInfo

        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str)
            except ValueError:
                return {"error": f"Invalid date format: {start_date_str}"}
        else:
            start_date = datetime.now(ZoneInfo(timezone))

        calendar = PublishingCalendar(
            start_date=start_date,
            timezone=timezone,
        )

        task_ids = await self.publisher.schedule_campaign(campaign, calendar)

        return {
            "campaign_id": campaign_id,
            "status": campaign.status,
            "scheduled_tasks": task_ids,
            "platforms": [p.value for p in campaign.platforms],
            "timezone": timezone,
        }

    @trace_operation("content_publish")
    async def handle_content_publish(self, args: dict) -> dict:
        """Publish content to a specific platform."""
        campaign_id = args.get("campaign_id", "")
        platform_str = args.get("platform", "")

        campaign = self._campaigns.get(campaign_id)
        if not campaign:
            return {"error": f"Campaign not found: {campaign_id}"}

        try:
            platform = Platform(platform_str.lower())
        except ValueError:
            return {"error": f"Invalid platform: {platform_str}"}

        content = campaign.get_content(platform)
        if not content:
            return {"error": f"No content for platform: {platform_str}"}

        result = await self.publisher.publish_content(platform, content)

        return {
            "campaign_id": campaign_id,
            "platform": platform.value,
            "success": result.success,
            "post_id": result.post_id,
            "post_url": result.post_url,
            "error": result.error,
            "published_at": result.published_at.isoformat(),
        }

    @trace_operation("content_publish_all")
    async def handle_content_publish_all(self, args: dict) -> dict:
        """Publish a campaign to all platforms."""
        campaign_id = args.get("campaign_id", "")

        campaign = self._campaigns.get(campaign_id)
        if not campaign:
            return {"error": f"Campaign not found: {campaign_id}"}

        results = await self.publisher.publish_campaign_now(campaign)

        return {
            "campaign_id": campaign_id,
            "results": {
                platform: {
                    "success": result.success,
                    "post_id": result.post_id,
                    "post_url": result.post_url,
                    "error": result.error,
                }
                for platform, result in results.items()
            },
            "all_success": all(r.success for r in results.values()),
        }

    @trace_operation("content_status")
    async def handle_content_status(self, args: dict) -> dict:
        """Check status of a campaign."""
        campaign_id = args.get("campaign_id", "")

        campaign = self._campaigns.get(campaign_id)
        if not campaign:
            return {"error": f"Campaign not found: {campaign_id}"}

        status = await self.publisher.get_campaign_status(campaign_id)

        return {
            "campaign_id": campaign_id,
            "campaign_status": campaign.status,
            "topic": campaign.topic,
            "platforms": [p.value for p in campaign.platforms],
            "publisher_status": status.to_dict() if status else None,
            "created_at": campaign.created_at.isoformat(),
        }

    @trace_operation("content_list_campaigns")
    async def handle_content_list_campaigns(self, args: dict) -> dict:
        """List all campaigns."""
        status_filter = args.get("status", "all")
        limit = args.get("limit", 10)

        campaigns = list(self._campaigns.values())

        if status_filter != "all":
            campaigns = [c for c in campaigns if c.status == status_filter]

        # Sort by created_at descending
        campaigns = sorted(campaigns, key=lambda c: c.created_at, reverse=True)

        # Apply limit
        campaigns = campaigns[:limit]

        return {
            "campaigns": [
                {
                    "campaign_id": c.campaign_id,
                    "topic": c.topic,
                    "status": c.status,
                    "platforms": [p.value for p in c.platforms],
                    "created_at": c.created_at.isoformat(),
                }
                for c in campaigns
            ],
            "total": len(self._campaigns),
            "filtered": len(campaigns),
        }

    @trace_operation("content_cancel")
    async def handle_content_cancel(self, args: dict) -> dict:
        """Cancel a scheduled campaign."""
        campaign_id = args.get("campaign_id", "")

        campaign = self._campaigns.get(campaign_id)
        if not campaign:
            return {"error": f"Campaign not found: {campaign_id}"}

        success = await self.publisher.cancel_campaign(campaign_id)

        if success:
            campaign.status = "cancelled"

        return {
            "campaign_id": campaign_id,
            "cancelled": success,
            "status": campaign.status,
        }
