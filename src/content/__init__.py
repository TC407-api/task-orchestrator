"""
Content Automation System for Task Orchestrator.

Generates and publishes platform-specific content for marketing campaigns.

Usage:
    from src.content import ContentGenerator, ContentPublisher

    generator = ContentGenerator()
    campaign = await generator.generate_campaign(
        topic="AI Agent Observability",
        source_content="Article content...",
        platforms=["twitter", "linkedin", "devto"],
    )

    publisher = ContentPublisher(scheduler, inbox)
    task_ids = await publisher.schedule_campaign(campaign, start_date)
"""
from .generator import (
    ContentGenerator,
    PlatformContent,
    Campaign,
    Platform,
    PLATFORM_CONSTRAINTS,
)
from .publisher import (
    ContentPublisher,
    PublishingCalendar,
    PublishResult,
    CampaignStatus,
)

__all__ = [
    # Generator
    "ContentGenerator",
    "PlatformContent",
    "Campaign",
    "Platform",
    "PLATFORM_CONSTRAINTS",
    # Publisher
    "ContentPublisher",
    "PublishingCalendar",
    "PublishResult",
    "CampaignStatus",
]
