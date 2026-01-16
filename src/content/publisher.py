"""
Content Publisher for Multi-Platform Scheduling.

Schedules and publishes content across platforms using BackgroundTaskScheduler.
Integrates with Twitter, LinkedIn, and Dev.to APIs.

Usage:
    publisher = ContentPublisher(scheduler, inbox)
    task_ids = await publisher.schedule_campaign(campaign, start_date)
    status = await publisher.get_campaign_status(campaign_id)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from ..agents.background_tasks import (
    BackgroundTaskScheduler,
    ScheduledTask,
    TaskScheduleType,
)
from ..agents.inbox import UniversalInbox, AgentEvent, EventType
from ..integrations.twitter import TwitterClient
from ..integrations.linkedin import LinkedInClient
from ..integrations.devto import DevToClient
from .generator import Campaign, Platform, PlatformContent, PLATFORM_CONSTRAINTS

logger = logging.getLogger(__name__)


@dataclass
class PublishingCalendar:
    """Publishing schedule configuration."""
    start_date: datetime = field(default_factory=datetime.now)
    timezone: str = "America/New_York"  # EST
    time_between_posts_minutes: int = 30

    # Platform-specific best times (in EST)
    platform_times: dict[str, str] = field(default_factory=lambda: {
        "twitter": "09:00",
        "linkedin": "09:00",
        "devto": "10:00",
        "reddit": "11:00",
        "hackernews": "09:00",
    })

    def get_publish_time(self, platform: Platform) -> datetime:
        """Get optimal publish time for a platform."""
        time_str = self.platform_times.get(platform.value, "09:00")
        hour, minute = map(int, time_str.split(":"))

        tz = ZoneInfo(self.timezone)
        publish_time = self.start_date.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
            tzinfo=tz,
        )

        # If time has passed today, schedule for tomorrow
        if publish_time < datetime.now(tz):
            publish_time += timedelta(days=1)

        return publish_time


@dataclass
class PublishResult:
    """Result of publishing content to a platform."""
    platform: Platform
    success: bool
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    error: Optional[str] = None
    published_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "platform": self.platform.value,
            "success": self.success,
            "post_id": self.post_id,
            "post_url": self.post_url,
            "error": self.error,
            "published_at": self.published_at.isoformat(),
        }


@dataclass
class CampaignStatus:
    """Status of a scheduled campaign."""
    campaign_id: str
    status: str  # draft, scheduled, publishing, completed, failed
    scheduled_posts: dict[str, str] = field(default_factory=dict)  # platform -> task_id
    published_posts: dict[str, PublishResult] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "status": self.status,
            "scheduled_posts": self.scheduled_posts,
            "published_posts": {
                p: r.to_dict() for p, r in self.published_posts.items()
            },
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ContentPublisher:
    """
    Manages content publishing across multiple platforms.

    Integrates with BackgroundTaskScheduler for scheduled publishing.
    """

    def __init__(
        self,
        scheduler: BackgroundTaskScheduler,
        inbox: Optional[UniversalInbox] = None,
        twitter_client: Optional[TwitterClient] = None,
        linkedin_client: Optional[LinkedInClient] = None,
        devto_client: Optional[DevToClient] = None,
    ):
        """
        Initialize the publisher.

        Args:
            scheduler: BackgroundTaskScheduler for scheduled tasks
            inbox: UniversalInbox for event publishing (optional)
            twitter_client: Twitter API client (optional, created if not provided)
            linkedin_client: LinkedIn API client (optional, created if not provided)
            devto_client: Dev.to API client (optional, created if not provided)
        """
        self.scheduler = scheduler
        self.inbox = inbox

        # Initialize API clients
        self.twitter = twitter_client or TwitterClient()
        self.linkedin = linkedin_client or LinkedInClient()
        self.devto = devto_client or DevToClient()

        # Track campaign statuses
        self._campaign_statuses: dict[str, CampaignStatus] = {}

    async def schedule_campaign(
        self,
        campaign: Campaign,
        calendar: Optional[PublishingCalendar] = None,
    ) -> list[str]:
        """
        Schedule a campaign for publishing.

        Args:
            campaign: Campaign with content for each platform
            calendar: Publishing calendar with times (optional)

        Returns:
            List of scheduled task IDs
        """
        calendar = calendar or PublishingCalendar()
        task_ids = []

        # Create campaign status
        status = CampaignStatus(
            campaign_id=campaign.campaign_id,
            status="scheduled",
            scheduled_at=datetime.now(),
        )

        for platform in campaign.platforms:
            content = campaign.get_content(platform)
            if not content:
                continue

            # Get optimal publish time for platform
            publish_time = calendar.get_publish_time(platform)

            # Create scheduled task
            task = ScheduledTask(
                name=f"publish_{campaign.campaign_id}_{platform.value}",
                func=self._create_publish_func(platform, content),
                schedule_type=TaskScheduleType.ONE_TIME,
                run_at=publish_time,
            )

            task_id = await self.scheduler.schedule_task(task)
            task_ids.append(task_id)
            status.scheduled_posts[platform.value] = task_id

            logger.info(
                f"Scheduled {platform.value} post for {publish_time.isoformat()}"
            )

        self._campaign_statuses[campaign.campaign_id] = status
        campaign.status = "scheduled"

        # Publish scheduling event
        if self.inbox:
            await self.inbox.publish(
                AgentEvent(
                    event_type=EventType.TEXT_OUTPUT,
                    agent_name="ContentPublisher",
                    data={
                        "campaign_id": campaign.campaign_id,
                        "platforms": [p.value for p in campaign.platforms],
                        "task_ids": task_ids,
                        "message": f"Campaign '{campaign.topic}' scheduled",
                    },
                    source="content_publisher",
                )
            )

        return task_ids

    def _create_publish_func(
        self,
        platform: Platform,
        content: PlatformContent,
    ):
        """Create a publishing function for the scheduler."""

        async def publish_func():
            """Publish content to platform."""
            result = await self.publish_content(platform, content)
            return result.to_dict()

        return publish_func

    async def publish_content(
        self,
        platform: Platform,
        content: PlatformContent,
    ) -> PublishResult:
        """
        Publish content to a specific platform immediately.

        Args:
            platform: Target platform
            content: Content to publish

        Returns:
            PublishResult with status
        """
        try:
            if platform == Platform.TWITTER:
                return await self._publish_twitter(content)
            elif platform == Platform.LINKEDIN:
                return await self._publish_linkedin(content)
            elif platform == Platform.DEVTO:
                return await self._publish_devto(content)
            else:
                # Reddit and HN require manual posting or different APIs
                return PublishResult(
                    platform=platform,
                    success=False,
                    error=f"Automated publishing not supported for {platform.value}",
                )

        except Exception as e:
            logger.error(f"Failed to publish to {platform.value}: {e}")
            return PublishResult(
                platform=platform,
                success=False,
                error=str(e),
            )

    async def _publish_twitter(self, content: PlatformContent) -> PublishResult:
        """Publish to Twitter."""
        if content.thread_parts:
            # Post as thread
            thread = await self.twitter.post_thread(content.thread_parts)
            return PublishResult(
                platform=Platform.TWITTER,
                success=True,
                post_id=thread.conversation_id,
                post_url=f"https://twitter.com/i/web/status/{thread.conversation_id}",
            )
        else:
            # Post single tweet
            tweet = await self.twitter.post_tweet(content.content)
            return PublishResult(
                platform=Platform.TWITTER,
                success=True,
                post_id=tweet.id,
                post_url=f"https://twitter.com/i/web/status/{tweet.id}",
            )

    async def _publish_linkedin(self, content: PlatformContent) -> PublishResult:
        """Publish to LinkedIn."""
        post = await self.linkedin.create_post(
            text=content.content,
            link_url=content.link_url,
            link_title=content.title,
        )
        return PublishResult(
            platform=Platform.LINKEDIN,
            success=True,
            post_id=post.id,
            post_url=post.share_url,
        )

    async def _publish_devto(self, content: PlatformContent) -> PublishResult:
        """Publish to Dev.to."""
        article = await self.devto.create_article(
            title=content.title or "Untitled",
            body_markdown=content.content,
            tags=content.tags,
            published=True,
            canonical_url=content.link_url,
        )
        return PublishResult(
            platform=Platform.DEVTO,
            success=True,
            post_id=str(article.id),
            post_url=article.url,
        )

    async def get_campaign_status(self, campaign_id: str) -> Optional[CampaignStatus]:
        """
        Get status of a campaign.

        Args:
            campaign_id: Campaign ID to check

        Returns:
            CampaignStatus or None if not found
        """
        return self._campaign_statuses.get(campaign_id)

    async def cancel_campaign(self, campaign_id: str) -> bool:
        """
        Cancel a scheduled campaign.

        Args:
            campaign_id: Campaign ID to cancel

        Returns:
            True if cancelled successfully
        """
        status = self._campaign_statuses.get(campaign_id)
        if not status:
            return False

        # Cancel all scheduled tasks
        for platform, task_id in status.scheduled_posts.items():
            await self.scheduler.cancel_task(task_id)
            logger.info(f"Cancelled {platform} post task: {task_id}")

        status.status = "cancelled"
        return True

    async def list_scheduled_campaigns(self) -> list[CampaignStatus]:
        """
        List all scheduled campaigns.

        Returns:
            List of CampaignStatus objects
        """
        return [
            status
            for status in self._campaign_statuses.values()
            if status.status in ("scheduled", "publishing")
        ]

    async def publish_campaign_now(self, campaign: Campaign) -> dict[str, PublishResult]:
        """
        Publish a campaign immediately to all platforms.

        Args:
            campaign: Campaign to publish

        Returns:
            Dict mapping platform to PublishResult
        """
        results = {}

        for platform in campaign.platforms:
            content = campaign.get_content(platform)
            if content:
                result = await self.publish_content(platform, content)
                results[platform.value] = result

                # Update status
                if campaign.campaign_id in self._campaign_statuses:
                    self._campaign_statuses[campaign.campaign_id].published_posts[
                        platform.value
                    ] = result

        # Update campaign status
        if campaign.campaign_id in self._campaign_statuses:
            status = self._campaign_statuses[campaign.campaign_id]
            all_success = all(r.success for r in results.values())
            status.status = "completed" if all_success else "failed"
            status.completed_at = datetime.now()

        campaign.status = "published"

        # Publish completion event
        if self.inbox:
            await self.inbox.publish(
                AgentEvent(
                    event_type=EventType.TEXT_OUTPUT,
                    agent_name="ContentPublisher",
                    data={
                        "campaign_id": campaign.campaign_id,
                        "results": {p: r.to_dict() for p, r in results.items()},
                        "message": f"Campaign '{campaign.topic}' published",
                    },
                    source="content_publisher",
                )
            )

        return results
