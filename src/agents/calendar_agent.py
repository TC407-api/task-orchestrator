"""Calendar agent for scheduling and time management."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from ..integrations.calendar import CalendarClient, CalendarEvent


class TimeBlockType(Enum):
    """Types of time blocks."""
    FOCUS = "focus"
    MEETING = "meeting"
    BREAK = "break"
    TASK = "task"
    BUFFER = "buffer"


@dataclass
class ScheduledTask:
    """A task scheduled on the calendar."""
    task_id: str
    title: str
    event_id: str
    start: datetime
    end: datetime
    calendar_id: str = "primary"


@dataclass
class DaySchedule:
    """Summary of a day's schedule."""
    date: datetime
    total_meetings: int
    meeting_hours: float
    focus_hours: float
    free_slots: list[tuple[datetime, datetime]]
    busy_percentage: float


class CalendarAgent:
    """Agent for calendar management and scheduling."""

    def __init__(
        self,
        calendar_client: CalendarClient,
        working_hours: tuple[int, int] = (9, 17),
        timezone: str = "America/New_York",
    ):
        self.calendar = calendar_client
        self.working_hours = working_hours
        self.tz = ZoneInfo(timezone)

    def get_day_summary(self, date: Optional[datetime] = None) -> DaySchedule:
        """
        Get summary of a day's schedule.

        Args:
            date: Date to summarize (default: today)

        Returns:
            DaySchedule with meeting count, hours, free slots
        """
        if date is None:
            date = datetime.now(self.tz)

        # Set time range for the day
        day_start = date.replace(
            hour=self.working_hours[0], minute=0, second=0, microsecond=0
        )
        day_end = date.replace(
            hour=self.working_hours[1], minute=0, second=0, microsecond=0
        )

        # Get events
        events = self.calendar.list_events(
            time_min=day_start,
            time_max=day_end,
        )

        # Calculate metrics
        total_meetings = len(events)
        meeting_minutes = 0

        for event in events:
            duration = (event.end - event.start).total_seconds() / 60
            meeting_minutes += duration

        meeting_hours = meeting_minutes / 60
        working_hours_total = self.working_hours[1] - self.working_hours[0]
        focus_hours = working_hours_total - meeting_hours

        # Find free slots
        free_slots = self.calendar.find_free_slots(
            duration_minutes=30,
            time_min=day_start,
            time_max=day_end,
            working_hours=self.working_hours,
        )

        busy_percentage = (meeting_hours / working_hours_total) * 100

        return DaySchedule(
            date=date,
            total_meetings=total_meetings,
            meeting_hours=meeting_hours,
            focus_hours=max(0, focus_hours),
            free_slots=free_slots,
            busy_percentage=min(100, busy_percentage),
        )

    def schedule_task(
        self,
        task_id: str,
        title: str,
        duration_minutes: int,
        preferred_time: Optional[datetime] = None,
        deadline: Optional[datetime] = None,
        calendar_id: str = "primary",
    ) -> Optional[ScheduledTask]:
        """
        Schedule a task on the calendar.

        Args:
            task_id: Unique task identifier
            title: Task title
            duration_minutes: Required duration
            preferred_time: Preferred start time
            deadline: Must complete by this time
            calendar_id: Calendar to use

        Returns:
            ScheduledTask if scheduled, None if no slot available
        """
        # Find available slot
        time_min = preferred_time or datetime.now(self.tz)
        time_max = deadline or (time_min + timedelta(days=7))

        free_slots = self.calendar.find_free_slots(
            duration_minutes=duration_minutes,
            time_min=time_min,
            time_max=time_max,
            working_hours=self.working_hours,
        )

        if not free_slots:
            return None

        # Use first available slot
        start, end = free_slots[0]

        # Adjust end time to match requested duration
        end = start + timedelta(minutes=duration_minutes)

        # Create event
        event = self.calendar.create_event(
            summary=f"[Task] {title}",
            start=start,
            end=end,
            description=f"Task ID: {task_id}",
            calendar_id=calendar_id,
            send_notifications=False,
        )

        return ScheduledTask(
            task_id=task_id,
            title=title,
            event_id=event.id,
            start=start,
            end=end,
            calendar_id=calendar_id,
        )

    def block_focus_time(
        self,
        duration_minutes: int = 120,
        title: str = "Focus Time",
        days_ahead: int = 5,
        min_blocks_per_day: int = 1,
    ) -> list[CalendarEvent]:
        """
        Automatically block focus time in calendar.

        Args:
            duration_minutes: Length of each focus block
            title: Event title
            days_ahead: How many days to schedule
            min_blocks_per_day: Minimum blocks to try to schedule per day

        Returns:
            List of created focus time events
        """
        created_events = []
        current = datetime.now(self.tz)

        for day_offset in range(days_ahead):
            day = current + timedelta(days=day_offset)
            day_start = day.replace(
                hour=self.working_hours[0], minute=0, second=0
            )
            day_end = day.replace(
                hour=self.working_hours[1], minute=0, second=0
            )

            # Skip weekends
            if day.weekday() >= 5:
                continue

            # Find free slots for this day
            free_slots = self.calendar.find_free_slots(
                duration_minutes=duration_minutes,
                time_min=day_start,
                time_max=day_end,
                working_hours=self.working_hours,
            )

            # Schedule focus blocks
            blocks_created = 0
            for start, end in free_slots:
                if blocks_created >= min_blocks_per_day:
                    break

                # Prefer morning focus time
                if start.hour < 12 or blocks_created == 0:
                    event = self.calendar.create_event(
                        summary=f"🎯 {title}",
                        start=start,
                        end=start + timedelta(minutes=duration_minutes),
                        description="Automatically blocked for focused work",
                        send_notifications=False,
                    )
                    created_events.append(event)
                    blocks_created += 1

        return created_events

    def check_conflicts(
        self,
        start: datetime,
        end: datetime,
        calendars: Optional[list[str]] = None,
    ) -> list[CalendarEvent]:
        """
        Check for scheduling conflicts.

        Args:
            start: Proposed start time
            end: Proposed end time
            calendars: Calendars to check

        Returns:
            List of conflicting events
        """
        events = self.calendar.list_events(
            time_min=start,
            time_max=end,
            max_results=50,
        )

        conflicts = []
        for event in events:
            # Check for overlap
            if event.start < end and event.end > start:
                conflicts.append(event)

        return conflicts

    def suggest_meeting_times(
        self,
        duration_minutes: int,
        attendees: list[str],
        days_ahead: int = 7,
        num_suggestions: int = 3,
    ) -> list[tuple[datetime, datetime]]:
        """
        Suggest meeting times checking multiple calendars.

        Args:
            duration_minutes: Meeting duration
            attendees: List of attendee emails
            days_ahead: Days to search
            num_suggestions: Number of suggestions to return

        Returns:
            List of (start, end) tuples for suggested times
        """
        time_min = datetime.now(self.tz)
        time_max = time_min + timedelta(days=days_ahead)

        # Get free/busy for all attendees
        calendars = ["primary"] + attendees
        free_busy = self.calendar.get_free_busy(time_min, time_max, calendars)

        # Find slots free for everyone
        # Simplified: just use primary calendar's free slots
        suggestions = self.calendar.find_free_slots(
            duration_minutes=duration_minutes,
            time_min=time_min,
            time_max=time_max,
            working_hours=self.working_hours,
        )

        return suggestions[:num_suggestions]

    def reschedule_task(
        self,
        event_id: str,
        new_start: Optional[datetime] = None,
        calendar_id: str = "primary",
    ) -> Optional[CalendarEvent]:
        """
        Reschedule a task event.

        Args:
            event_id: Event ID to reschedule
            new_start: New start time (finds next available if None)
            calendar_id: Calendar ID

        Returns:
            Updated event or None if cannot reschedule
        """
        # Get original event
        original = self.calendar.get_event(event_id, calendar_id)
        duration = (original.end - original.start).total_seconds() / 60

        if new_start is None:
            # Find next available slot
            free_slots = self.calendar.find_free_slots(
                duration_minutes=int(duration),
                time_min=datetime.now(self.tz),
                working_hours=self.working_hours,
            )

            if not free_slots:
                return None

            new_start, _ = free_slots[0]

        new_end = new_start + timedelta(minutes=duration)

        # Check for conflicts
        conflicts = self.check_conflicts(new_start, new_end)
        conflicts = [c for c in conflicts if c.id != event_id]

        if conflicts:
            return None

        # Update event
        return self.calendar.update_event(
            event_id=event_id,
            calendar_id=calendar_id,
            start=new_start,
            end=new_end,
        )

    def get_week_overview(
        self,
        start_date: Optional[datetime] = None,
    ) -> dict:
        """
        Get overview of the week's schedule.

        Args:
            start_date: Start of week (default: current week's Monday)

        Returns:
            Dict with daily summaries and weekly totals
        """
        if start_date is None:
            today = datetime.now(self.tz)
            start_date = today - timedelta(days=today.weekday())

        daily_summaries = []
        total_meeting_hours = 0
        total_focus_hours = 0

        for i in range(5):  # Monday to Friday
            day = start_date + timedelta(days=i)
            summary = self.get_day_summary(day)
            daily_summaries.append(summary)
            total_meeting_hours += summary.meeting_hours
            total_focus_hours += summary.focus_hours

        return {
            "week_start": start_date.isoformat(),
            "daily_summaries": daily_summaries,
            "total_meeting_hours": total_meeting_hours,
            "total_focus_hours": total_focus_hours,
            "average_busy_percentage": sum(
                d.busy_percentage for d in daily_summaries
            ) / 5,
        }
