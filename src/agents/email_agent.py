"""Email agent for task extraction and response drafting."""
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from ..integrations.gmail import GmailClient, EmailMessage

# Grade 5 Langfuse Tracing
try:
    from lib.tracing import observe, flush_traces
except ImportError:
    # Fallback if tracing not set up
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    def flush_traces():
        pass


class EmailCategory(Enum):
    """Email categorization."""
    URGENT = "urgent"
    ACTION_REQUIRED = "action_required"
    FOLLOW_UP = "follow_up"
    INFORMATIONAL = "informational"
    NEWSLETTER = "newsletter"
    SPAM = "spam"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExtractedTask:
    """A task extracted from an email."""
    description: str
    priority: TaskPriority
    due_date: Optional[datetime] = None
    source_email_id: str = ""
    source_subject: str = ""
    source_sender: str = ""
    context: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EmailAnalysis:
    """Analysis of an email."""
    email_id: str
    category: EmailCategory
    summary: str
    extracted_tasks: list[ExtractedTask]
    suggested_response: Optional[str] = None
    requires_response: bool = False
    sentiment: str = "neutral"


class EmailAgent:
    """Agent for email processing and task extraction."""

    def __init__(
        self,
        gmail_client: GmailClient,
        llm_client: Optional[object] = None,
    ):
        self.gmail = gmail_client
        self.llm = llm_client

    @observe(name="email_agent.analyze_email")
    async def analyze_email(self, email: EmailMessage) -> EmailAnalysis:
        """
        Analyze an email and extract actionable information.

        Args:
            email: EmailMessage to analyze

        Returns:
            EmailAnalysis with category, tasks, and suggestions
        """
        if self.llm is None:
            # Fallback to rule-based analysis
            return self._rule_based_analysis(email)

        # Use LLM for sophisticated analysis
        prompt = self._build_analysis_prompt(email)
        response = await self._call_llm(prompt)
        return self._parse_analysis_response(response, email)

    def _rule_based_analysis(self, email: EmailMessage) -> EmailAnalysis:
        """Simple rule-based email analysis."""
        subject_lower = email.subject.lower()
        body_lower = email.body.lower()

        # Determine category
        category = EmailCategory.INFORMATIONAL
        if any(word in subject_lower for word in ["urgent", "asap", "critical"]):
            category = EmailCategory.URGENT
        elif any(word in subject_lower for word in ["action", "required", "please"]):
            category = EmailCategory.ACTION_REQUIRED
        elif any(word in subject_lower for word in ["follow up", "reminder"]):
            category = EmailCategory.FOLLOW_UP
        elif any(word in subject_lower for word in ["newsletter", "digest", "weekly"]):
            category = EmailCategory.NEWSLETTER

        # Extract tasks (basic pattern matching)
        tasks = []
        task_indicators = [
            "could you", "can you", "please", "need you to",
            "would you", "by tomorrow", "by friday", "deadline",
        ]

        if any(indicator in body_lower for indicator in task_indicators):
            tasks.append(
                ExtractedTask(
                    description=f"Review and respond to: {email.subject}",
                    priority=TaskPriority.MEDIUM
                    if category == EmailCategory.URGENT
                    else TaskPriority.LOW,
                    source_email_id=email.id,
                    source_subject=email.subject,
                    source_sender=email.sender,
                    context=email.snippet,
                )
            )

        # Determine if response needed
        requires_response = category in [
            EmailCategory.URGENT,
            EmailCategory.ACTION_REQUIRED,
        ]

        return EmailAnalysis(
            email_id=email.id,
            category=category,
            summary=email.snippet[:200],
            extracted_tasks=tasks,
            requires_response=requires_response,
        )

    def _build_analysis_prompt(self, email: EmailMessage) -> str:
        """Build prompt for LLM analysis."""
        return f"""Analyze this email and extract actionable information.

FROM: {email.sender}
TO: {email.to}
DATE: {email.date}
SUBJECT: {email.subject}

BODY:
{email.body[:2000]}

Respond in JSON format:
{{
    "category": "urgent|action_required|follow_up|informational|newsletter|spam",
    "summary": "1-2 sentence summary",
    "sentiment": "positive|neutral|negative",
    "requires_response": true/false,
    "tasks": [
        {{
            "description": "task description",
            "priority": "critical|high|medium|low",
            "due_date": "YYYY-MM-DD or null",
            "tags": ["tag1", "tag2"]
        }}
    ],
    "suggested_response": "draft response if needed, or null"
}}"""

    @observe(name="email_agent.llm_call")
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        if self.llm is None:
            raise ValueError("LLM client not configured")
        # Placeholder - implement based on actual LLM client
        return "{}"

    def _parse_analysis_response(
        self, response: str, email: EmailMessage
    ) -> EmailAnalysis:
        """Parse LLM response into EmailAnalysis."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return self._rule_based_analysis(email)

        category = EmailCategory(data.get("category", "informational"))

        tasks = []
        for task_data in data.get("tasks", []):
            due_date = None
            if task_data.get("due_date"):
                try:
                    due_date = datetime.strptime(
                        task_data["due_date"], "%Y-%m-%d"
                    )
                except ValueError:
                    pass

            tasks.append(
                ExtractedTask(
                    description=task_data.get("description", ""),
                    priority=TaskPriority(task_data.get("priority", "medium")),
                    due_date=due_date,
                    source_email_id=email.id,
                    source_subject=email.subject,
                    source_sender=email.sender,
                    context=email.snippet,
                    tags=task_data.get("tags", []),
                )
            )

        return EmailAnalysis(
            email_id=email.id,
            category=category,
            summary=data.get("summary", email.snippet[:200]),
            extracted_tasks=tasks,
            suggested_response=data.get("suggested_response"),
            requires_response=data.get("requires_response", False),
            sentiment=data.get("sentiment", "neutral"),
        )

    @observe(name="email_agent.process_unread")
    async def process_unread(
        self, max_emails: int = 50
    ) -> list[EmailAnalysis]:
        """
        Process unread emails and extract tasks.

        Args:
            max_emails: Maximum emails to process

        Returns:
            List of EmailAnalysis for each processed email
        """
        # Get unread messages
        messages = self.gmail.list_messages("is:unread", max_results=max_emails)

        analyses = []
        for msg_meta in messages:
            # Get full message
            email = self.gmail.get_message(msg_meta["id"])

            # Analyze
            analysis = await self.analyze_email(email)
            analyses.append(analysis)

        # Flush traces to Langfuse
        flush_traces()
        return analyses

    @observe(name="email_agent.draft_response")
    async def draft_response(
        self,
        email: EmailMessage,
        context: str = "",
        tone: str = "professional",
    ) -> str:
        """
        Draft a response to an email.

        Args:
            email: Original email to respond to
            context: Additional context for response
            tone: Desired tone (professional, casual, formal)

        Returns:
            Draft response text
        """
        if self.llm is None:
            # Basic template response
            return f"""Thank you for your email regarding "{email.subject}".

I have received your message and will review it shortly.

Best regards"""

        prompt = f"""Draft a {tone} response to this email.

FROM: {email.sender}
SUBJECT: {email.subject}

ORIGINAL MESSAGE:
{email.body[:1500]}

CONTEXT: {context}

Write only the response body (no subject line or greeting needed)."""

        return await self._call_llm(prompt)

    def get_task_summary(
        self, analyses: list[EmailAnalysis]
    ) -> dict:
        """
        Summarize extracted tasks from multiple email analyses.

        Args:
            analyses: List of EmailAnalysis objects

        Returns:
            Summary dict with tasks grouped by priority
        """
        summary = {
            "total_emails": len(analyses),
            "urgent": [],
            "action_required": [],
            "tasks_by_priority": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": [],
            },
            "total_tasks": 0,
        }

        for analysis in analyses:
            if analysis.category == EmailCategory.URGENT:
                summary["urgent"].append(
                    {
                        "email_id": analysis.email_id,
                        "summary": analysis.summary,
                    }
                )
            elif analysis.category == EmailCategory.ACTION_REQUIRED:
                summary["action_required"].append(
                    {
                        "email_id": analysis.email_id,
                        "summary": analysis.summary,
                    }
                )

            for task in analysis.extracted_tasks:
                summary["tasks_by_priority"][task.priority.value].append(
                    {
                        "description": task.description,
                        "source": task.source_subject,
                        "due_date": (
                            task.due_date.isoformat() if task.due_date else None
                        ),
                    }
                )
                summary["total_tasks"] += 1

        return summary
