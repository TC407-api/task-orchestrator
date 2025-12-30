"""Gmail API integration."""
import base64
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from typing import Optional

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from ..core.config import settings
from ..core.rate_limiter import RateLimiter


@dataclass
class EmailMessage:
    """Represents an email message."""
    id: str
    thread_id: str
    subject: str
    sender: str
    to: str
    date: str
    snippet: str
    body: str = ""
    labels: list[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []


class GmailClient:
    """Client for Gmail API operations."""

    def __init__(self, credentials: Credentials):
        self.service = build("gmail", "v1", credentials=credentials)
        self.rate_limiter = RateLimiter(settings.gmail_rate_limit)

    def list_messages(
        self,
        query: str = "",
        max_results: int = 100,
        label_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        List messages matching query.

        Args:
            query: Gmail search query (e.g., "is:unread", "from:someone@example.com")
            max_results: Maximum number of messages to return
            label_ids: Filter by label IDs

        Returns:
            List of message metadata dicts with 'id' and 'threadId'
        """
        self.rate_limiter.wait()

        kwargs = {"userId": "me", "q": query, "maxResults": max_results}
        if label_ids:
            kwargs["labelIds"] = label_ids

        results = self.service.users().messages().list(**kwargs).execute()
        return results.get("messages", [])

    def get_message(self, msg_id: str, format: str = "full") -> EmailMessage:
        """
        Get full message content.

        Args:
            msg_id: Message ID
            format: Response format ('full', 'minimal', 'raw')

        Returns:
            EmailMessage object with parsed content
        """
        self.rate_limiter.wait()

        message = (
            self.service.users()
            .messages()
            .get(userId="me", id=msg_id, format=format)
            .execute()
        )

        # Extract headers
        headers = {}
        if "payload" in message and "headers" in message["payload"]:
            headers = {
                h["name"]: h["value"] for h in message["payload"]["headers"]
            }

        # Extract body
        body = self._extract_body(message)

        return EmailMessage(
            id=message["id"],
            thread_id=message["threadId"],
            subject=headers.get("Subject", ""),
            sender=headers.get("From", ""),
            to=headers.get("To", ""),
            date=headers.get("Date", ""),
            snippet=message.get("snippet", ""),
            body=body,
            labels=message.get("labelIds", []),
        )

    def _extract_body(self, message: dict) -> str:
        """Extract plain text body from message."""
        if "payload" not in message:
            return ""

        payload = message["payload"]

        # Check for multipart
        if "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8")

        # Single part message
        if "body" in payload and "data" in payload["body"]:
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode(
                "utf-8"
            )

        return ""

    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Send an email message.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body (plain text)
            thread_id: Optional thread ID for replies

        Returns:
            Sent message metadata
        """
        self.rate_limiter.wait()

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body_dict = {"raw": raw}

        if thread_id:
            body_dict["threadId"] = thread_id

        return (
            self.service.users()
            .messages()
            .send(userId="me", body=body_dict)
            .execute()
        )

    def add_labels(self, msg_id: str, labels: list[str]) -> dict:
        """Add labels to a message."""
        self.rate_limiter.wait()

        return (
            self.service.users()
            .messages()
            .modify(userId="me", id=msg_id, body={"addLabelIds": labels})
            .execute()
        )

    def remove_labels(self, msg_id: str, labels: list[str]) -> dict:
        """Remove labels from a message."""
        self.rate_limiter.wait()

        return (
            self.service.users()
            .messages()
            .modify(userId="me", id=msg_id, body={"removeLabelIds": labels})
            .execute()
        )

    def get_unread_count(self) -> int:
        """Get count of unread messages in inbox."""
        messages = self.list_messages("is:unread", max_results=500)
        return len(messages)
