"""
Research Scheduler - Topic registry and scheduling.

Manages research topics across projects and schedules daily research runs
using the BackgroundTaskScheduler.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for research scheduling."""
    run_time: str = "06:00"  # 24-hour format
    timezone: str = "America/New_York"
    max_searches_per_run: int = 10
    provider: str = "perplexity"  # "perplexity" or "firecrawl"


@dataclass
class TopicRegistry:
    """
    Manages research topics across projects.

    Topics are stored in ~/.claude/research/topics.json
    """
    global_topics: list[str] = field(default_factory=list)
    project_topics: dict[str, list[str]] = field(default_factory=dict)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    _registry_path: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "TopicRegistry":
        """
        Load topic registry from JSON file.

        Args:
            path: Path to topics.json. Defaults to ~/.claude/research/topics.json

        Returns:
            TopicRegistry instance
        """
        file_path: Path
        if path is None:
            file_path = Path.home() / ".claude" / "research" / "topics.json"
        else:
            file_path = Path(path)

        if not file_path.exists():
            logger.info(f"Creating new topic registry at {file_path}")
            registry = cls()
            registry._registry_path = file_path
            registry.save()
            return registry

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            schedule_data = data.get("schedule", {})
            schedule = ScheduleConfig(
                run_time=schedule_data.get("run_time", "06:00"),
                timezone=schedule_data.get("timezone", "America/New_York"),
                max_searches_per_run=schedule_data.get("max_searches_per_run", 10),
            )

            registry = cls(
                global_topics=data.get("global_topics", []),
                project_topics=data.get("project_topics", {}),
                schedule=schedule,
            )
            registry._registry_path = file_path

            logger.info(f"Loaded {len(registry.all_topics)} topics from {file_path}")
            return registry

        except Exception as e:
            logger.error(f"Error loading topic registry: {e}")
            registry = cls()
            registry._registry_path = file_path
            return registry

    def save(self) -> None:
        """Save topic registry to JSON file."""
        if self._registry_path is None:
            self._registry_path = Path.home() / ".claude" / "research" / "topics.json"

        # Ensure directory exists
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "global_topics": self.global_topics,
            "project_topics": self.project_topics,
            "schedule": {
                "run_time": self.schedule.run_time,
                "timezone": self.schedule.timezone,
                "max_searches_per_run": self.schedule.max_searches_per_run,
            },
        }

        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved topic registry to {self._registry_path}")

    @property
    def all_topics(self) -> list[str]:
        """Get all topics (global + all projects)."""
        all_t = list(self.global_topics)
        for topics in self.project_topics.values():
            all_t.extend(topics)
        return list(set(all_t))  # Deduplicate

    def get_topics_for_project(self, project: Optional[str] = None) -> list[str]:
        """
        Get topics for a specific project plus global topics.

        Args:
            project: Project name. If None, returns only global topics.

        Returns:
            List of topic strings
        """
        topics = list(self.global_topics)
        if project and project in self.project_topics:
            topics.extend(self.project_topics[project])
        return list(set(topics))

    def add_topic(self, topic: str, project: Optional[str] = None) -> bool:
        """
        Add a topic to the registry.

        Args:
            topic: Topic string to add
            project: Project name. If None, adds to global topics.

        Returns:
            True if added, False if already exists
        """
        if project is None or project == "global":
            if topic not in self.global_topics:
                self.global_topics.append(topic)
                self.save()
                logger.info(f"Added global topic: {topic}")
                return True
        else:
            if project not in self.project_topics:
                self.project_topics[project] = []
            if topic not in self.project_topics[project]:
                self.project_topics[project].append(topic)
                self.save()
                logger.info(f"Added topic '{topic}' to project '{project}'")
                return True

        return False

    def remove_topic(self, topic: str, project: Optional[str] = None) -> bool:
        """
        Remove a topic from the registry.

        Args:
            topic: Topic string to remove
            project: Project name. If None, removes from global topics.

        Returns:
            True if removed, False if not found
        """
        if project is None or project == "global":
            if topic in self.global_topics:
                self.global_topics.remove(topic)
                self.save()
                logger.info(f"Removed global topic: {topic}")
                return True
        else:
            if project in self.project_topics and topic in self.project_topics[project]:
                self.project_topics[project].remove(topic)
                self.save()
                logger.info(f"Removed topic '{topic}' from project '{project}'")
                return True

        return False

    def list_projects(self) -> list[str]:
        """Get list of all projects with topics."""
        return list(self.project_topics.keys())


class ResearchScheduler:
    """
    Schedules and orchestrates research runs.

    Integrates with BackgroundTaskScheduler for recurring execution.
    """

    def __init__(
        self,
        registry: Optional[TopicRegistry] = None,
        registry_path: Optional[str] = None,
    ):
        """
        Initialize research scheduler.

        Args:
            registry: Pre-loaded TopicRegistry. If None, loads from path.
            registry_path: Path to topics.json. Used if registry is None.
        """
        self.registry = registry or TopicRegistry.load(registry_path)
        self._scheduled_task_id: Optional[str] = None
        self._last_run: Optional[datetime] = None

    async def schedule_daily_run(
        self,
        background_scheduler,
        run_time: Optional[str] = None,
    ) -> str:
        """
        Schedule recurring research via BackgroundTaskScheduler.

        Args:
            background_scheduler: BackgroundTaskScheduler instance
            run_time: Time to run daily (HH:MM). Defaults to registry setting.

        Returns:
            Task ID
        """
        from ..agents.background_tasks import ScheduledTask, TaskScheduleType

        time_str = run_time or self.registry.schedule.run_time

        # Parse time
        hour, minute = map(int, time_str.split(":"))

        # Calculate next run time
        now = datetime.now()
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

        # Calculate interval (24 hours)
        interval_seconds = 24 * 60 * 60

        task = ScheduledTask(
            name="auto_research_daily",
            func=self._run_research,
            schedule_type=TaskScheduleType.RECURRING,
            interval_seconds=interval_seconds,
            next_run=next_run,
            timeout_seconds=600,  # 10 minute timeout
        )

        task_id: str = await background_scheduler.schedule_task(task)
        self._scheduled_task_id = task_id
        logger.info(f"Scheduled daily research at {time_str}, next run: {next_run}")

        return task_id

    async def run_now(
        self,
        topics: Optional[list[str]] = None,
        project: Optional[str] = None,
        max_topics: Optional[int] = None,
    ) -> dict:
        """
        Execute research immediately.

        Args:
            topics: Specific topics to research. If None, uses registry topics.
            project: Filter topics by project.
            max_topics: Maximum topics to research this run.

        Returns:
            Dict with research results
        """
        return await self._run_research(
            topics=topics,
            project=project,
            max_topics=max_topics,
        )

    async def _run_research(
        self,
        topics: Optional[list[str]] = None,
        project: Optional[str] = None,
        max_topics: Optional[int] = None,
    ) -> dict:
        """
        Internal research execution.

        Args:
            topics: Topics to research
            project: Project filter
            max_topics: Max topics per run

        Returns:
            Research results dict
        """
        from .primer import ContextPrimer
        from .summarizer import ResearchSummarizer

        start_time = datetime.now()

        # Determine topics
        if topics is None:
            topics = self.registry.get_topics_for_project(project)

        # Apply limit
        max_t = max_topics or self.registry.schedule.max_searches_per_run
        topics = topics[:max_t]

        if not topics:
            logger.warning("No topics to research")
            return {
                "status": "no_topics",
                "message": "No research topics configured",
                "timestamp": start_time.isoformat(),
            }

        provider = self.registry.schedule.provider
        logger.info(f"Starting research run for {len(topics)} topics via {provider}")

        # Execute research based on provider
        results = []
        summaries = {}

        if provider == "perplexity":
            results, summaries = await self._run_perplexity_research(topics)
        else:
            results, summaries = await self._run_firecrawl_research(topics)

        # Store summaries in Graphiti
        summarizer = ResearchSummarizer()
        for topic, summary in summaries.items():
            await summarizer.store_in_graphiti(topic, summary)

        # Generate context file with summaries
        primer = ContextPrimer()
        date_str = start_time.strftime("%Y-%m-%d")
        context_path = primer.generate_context_file(date_str, results, summaries)

        self._last_run = start_time
        end_time = datetime.now()

        return {
            "status": "completed",
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "topics_researched": len(topics),
            "successful": len([r for r in results if r["status"] == "success"]),
            "context_file": str(context_path),
            "results": results,
        }

    async def _run_perplexity_research(
        self, topics: list[str]
    ) -> tuple[list[dict], dict[str, str]]:
        """
        Run research via Perplexity API.

        Returns:
            Tuple of (results list, summaries dict)
        """
        from .perplexity_runner import PerplexityRunner

        runner = PerplexityRunner()
        results = []
        summaries = {}

        try:
            for topic in topics:
                try:
                    result = await runner.research_topic(topic)

                    if result.get("answer"):
                        summaries[topic] = result["answer"]
                        results.append({
                            "topic": topic,
                            "status": "success",
                            "results_count": len(result.get("citations", [])),
                            "summary_length": len(result["answer"]),
                        })
                    else:
                        results.append({
                            "topic": topic,
                            "status": "error",
                            "error": result.get("error", "No answer"),
                        })

                except Exception as e:
                    logger.error(f"Perplexity error for '{topic}': {e}")
                    results.append({
                        "topic": topic,
                        "status": "error",
                        "error": str(e),
                    })
        finally:
            await runner.close()

        return results, summaries

    async def _run_firecrawl_research(
        self, topics: list[str]
    ) -> tuple[list[dict], dict[str, str]]:
        """
        Run research via Firecrawl API.

        Returns:
            Tuple of (results list, summaries dict)
        """
        from .runner import ResearchRunner
        from .summarizer import ResearchSummarizer

        runner = ResearchRunner()
        summarizer = ResearchSummarizer()
        results = []
        summaries = {}

        try:
            for topic in topics:
                try:
                    search_results = await runner.search_topic(topic)

                    if search_results:
                        summary = await summarizer.summarize(topic, search_results)
                        summaries[topic] = summary
                        results.append({
                            "topic": topic,
                            "status": "success",
                            "results_count": len(search_results),
                            "summary_length": len(summary),
                        })
                    else:
                        results.append({
                            "topic": topic,
                            "status": "no_results",
                        })

                except Exception as e:
                    logger.error(f"Firecrawl error for '{topic}': {e}")
                    results.append({
                        "topic": topic,
                        "status": "error",
                        "error": str(e),
                    })
        finally:
            await runner.close()

        return results, summaries

    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "scheduled_task_id": self._scheduled_task_id,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "total_topics": len(self.registry.all_topics),
            "global_topics": len(self.registry.global_topics),
            "projects": self.registry.list_projects(),
            "schedule": {
                "run_time": self.registry.schedule.run_time,
                "timezone": self.registry.schedule.timezone,
                "max_searches_per_run": self.registry.schedule.max_searches_per_run,
                "provider": self.registry.schedule.provider,
            },
        }
