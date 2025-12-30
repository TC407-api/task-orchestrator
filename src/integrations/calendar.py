"""Google Calendar API integration."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from ..core.config import settings
from ..core.rate_limiter import RateLimiter


@dataclass
class CalendarEvent:
    """Represents a calendar event."""
    id: str
    summary: str
    start: datetime
    end: datetime
    description: str = ""
    location: str = ""
    attendees: list[str] = field(default_factory=list)
    calendar_id: str = "primary"
    html_link: str = ""
    status: str = "confirmed"

    @classmethod
    def from_api_response(cls, event: dict) -> "CalendarEvent":
        """Create CalendarEvent from API response."""
        start = event.get("start", {})
        end = event.get("end", {})

        # Handle all-day events vs timed events
        start_dt = cls._parse_datetime(start)
        end_dt = cls._parse_datetime(end)

        attendees = [
            a.get("email", "")
            for a in event.get("attendees", [])
        ]

        return cls(
            id=event.get("id", ""),
            summary=event.get("summary", "(No title)"),
            start=start_dt,
            end=end_dt,
            description=event.get("description", ""),
            location=event.get("location", ""),
            attendees=attendees,
            calendar_id=event.get("organizer", {}).get("email", "primary"),
            html_link=event.get("htmlLink", ""),
            status=event.get("status", "confirmed"),
        )

    @staticmethod
    def _parse_datetime(dt_dict: dict) -> datetime:
        """Parse datetime from API response."""
        if "dateTime" in dt_dict:
            dt_str = dt_dict["dateTime"]
            # Handle timezone
            if "+" in dt_str or dt_str.endswith("Z"):
                return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return datetime.fromisoformat(dt_str)
        elif "date" in dt_dict:
            # All-day event
            return datetime.strptime(dt_dict["date"], "%Y-%m-%d")
        return datetime.now()


@dataclass
class FreeBusySlot:
    """Represents a busy time slot."""
    start: datetime
    end: datetime


class CalendarClient:
    """Client for Google Calendar API operations."""

    def __init__(self, credentials: Credentials):
        self.service = build("calendar", "v3", credentials=credentials)
        self.rate_limiter = RateLimiter(settings.calendar_rate_limit)

    def list_events(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 100,
        calendar_id: str = "primary",
        single_events: bool = True,
    ) -> list[CalendarEvent]:
        """
        List calendar events.

        Args:
            time_min: Start of time range (default: now)
            time_max: End of time range (default: 7 days from now)
            max_results: Maximum events to return
            calendar_id: Calendar ID (default: primary)
            single_events: Expand recurring events

        Returns:
            List of CalendarEvent objects
        """
        self.rate_limiter.wait()

        if time_min is None:
            time_min = datetime.now(ZoneInfo("UTC"))
        if time_max is None:
            time_max = time_min + timedelta(days=7)

        # Ensure timezone-aware
        if time_min.tzinfo is None:
            time_min = time_min.replace(tzinfo=ZoneInfo("UTC"))
        if time_max.tzinfo is None:
            time_max = time_max.replace(tzinfo=ZoneInfo("UTC"))

        results = (
            self.service.events()
            .list(
                calendarId=calendar_id,
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat(),
                maxResults=max_results,
                singleEvents=single_events,
                orderBy="startTime",
            )
            .execute()
        )

        return [
            CalendarEvent.from_api_response(event)
            for event in results.get("items", [])
        ]

    def get_event(self, event_id: str, calendar_id: str = "primary") -> CalendarEvent:
        """
        Get a specific event.

        Args:
            event_id: Event ID
            calendar_id: Calendar ID

        Returns:
            CalendarEvent object
        """
        self.rate_limiter.wait()

        event = (
            self.service.events()
            .get(calendarId=calendar_id, eventId=event_id)
            .execute()
        )

        return CalendarEvent.from_api_response(event)

    def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: str = "",
        location: str = "",
        attendees: Optional[list[str]] = None,
        calendar_id: str = "primary",
        send_notifications: bool = True,
    ) -> CalendarEvent:
        """
        Create a new event.

        Args:
            summary: Event title
            start: Start datetime
            end: End datetime
            description: Event description
            location: Event location
            attendees: List of attendee email addresses
            calendar_id: Calendar ID
            send_notifications: Send invite emails

        Returns:
            Created CalendarEvent
        """
        self.rate_limiter.wait()

        # Build event body
        event_body = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
        }

        if attendees:
            event_body["attendees"] = [{"email": email} for email in attendees]

        event = (
            self.service.events()
            .insert(
                calendarId=calendar_id,
                body=event_body,
                sendUpdates="all" if send_notifications else "none",
            )
            .execute()
        )

        return CalendarEvent.from_api_response(event)

    def update_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        summary: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> CalendarEvent:
        """
        Update an existing event.

        Args:
            event_id: Event ID to update
            calendar_id: Calendar ID
            summary: New title (if provided)
            start: New start time (if provided)
            end: New end time (if provided)
            description: New description (if provided)
            location: New location (if provided)

        Returns:
            Updated CalendarEvent
        """
        self.rate_limiter.wait()

        # Get existing event
        event = (
            self.service.events()
            .get(calendarId=calendar_id, eventId=event_id)
            .execute()
        )

        # Update fields if provided
        if summary is not None:
            event["summary"] = summary
        if description is not None:
            event["description"] = description
        if location is not None:
            event["location"] = location
        if start is not None:
            event["start"] = {"dateTime": start.isoformat(), "timeZone": "UTC"}
        if end is not None:
            event["end"] = {"dateTime": end.isoformat(), "timeZone": "UTC"}

        updated = (
            self.service.events()
            .update(calendarId=calendar_id, eventId=event_id, body=event)
            .execute()
        )

        return CalendarEvent.from_api_response(updated)

    def delete_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        send_notifications: bool = True,
    ) -> None:
        """
        Delete an event.

        Args:
            event_id: Event ID to delete
            calendar_id: Calendar ID
            send_notifications: Send cancellation emails
        """
        self.rate_limiter.wait()

        self.service.events().delete(
            calendarId=calendar_id,
            eventId=event_id,
            sendUpdates="all" if send_notifications else "none",
        ).execute()

    def get_free_busy(
        self,
        time_min: datetime,
        time_max: datetime,
        calendars: Optional[list[str]] = None,
    ) -> dict[str, list[FreeBusySlot]]:
        """
        Get free/busy information.

        Args:
            time_min: Start of time range
            time_max: End of time range
            calendars: List of calendar IDs (default: primary)

        Returns:
            Dict mapping calendar ID to list of busy slots
        """
        self.rate_limiter.wait()

        if calendars is None:
            calendars = ["primary"]

        body = {
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "items": [{"id": cal} for cal in calendars],
        }

        results = self.service.freebusy().query(body=body).execute()

        free_busy = {}
        for cal_id, cal_data in results.get("calendars", {}).items():
            slots = []
            for busy in cal_data.get("busy", []):
                slots.append(
                    FreeBusySlot(
                        start=datetime.fromisoformat(
                            busy["start"].replace("Z", "+00:00")
                        ),
                        end=datetime.fromisoformat(
                            busy["end"].replace("Z", "+00:00")
                        ),
                    )
                )
            free_busy[cal_id] = slots

        return free_busy

    def find_free_slots(
        self,
        duration_minutes: int,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        working_hours: tuple[int, int] = (9, 17),
        calendars: Optional[list[str]] = None,
    ) -> list[tuple[datetime, datetime]]:
        """
        Find available time slots.

        Args:
            duration_minutes: Required slot duration
            time_min: Start search from (default: now)
            time_max: End search at (default: 7 days)
            working_hours: Tuple of (start_hour, end_hour)
            calendars: Calendars to check

        Returns:
            List of (start, end) tuples for available slots
        """
        if time_min is None:
            time_min = datetime.now(ZoneInfo("UTC"))
        if time_max is None:
            time_max = time_min + timedelta(days=7)

        # Get busy times
        busy_map = self.get_free_busy(time_min, time_max, calendars)

        # Merge all busy slots
        all_busy: list[FreeBusySlot] = []
        for slots in busy_map.values():
            all_busy.extend(slots)

        # Sort by start time
        all_busy.sort(key=lambda x: x.start)

        # Find free slots within working hours
        free_slots = []
        current = time_min
        duration = timedelta(minutes=duration_minutes)

        while current + duration <= time_max:
            # Check if within working hours
            if working_hours[0] <= current.hour < working_hours[1]:
                slot_end = current + duration

                # Check if slot conflicts with any busy time
                is_free = True
                for busy in all_busy:
                    if current < busy.end and slot_end > busy.start:
                        is_free = False
                        # Jump to end of busy period
                        current = busy.end
                        break

                if is_free:
                    free_slots.append((current, slot_end))
                    current = slot_end
            else:
                # Jump to next working hour
                if current.hour >= working_hours[1]:
                    current = current.replace(
                        hour=working_hours[0], minute=0, second=0
                    ) + timedelta(days=1)
                else:
                    current = current.replace(
                        hour=working_hours[0], minute=0, second=0
                    )

        return free_slots
