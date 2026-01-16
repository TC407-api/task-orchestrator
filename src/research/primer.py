"""
Context Primer - Generate context files for Claude injection.

Creates markdown files with research summaries that can be injected
into Claude sessions via hooks.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ContextPrimer:
    """
    Generates context files for Claude injection.

    Creates dated markdown files in ~/.claude/research/
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize context primer.

        Args:
            base_path: Base path for context files. Defaults to ~/.claude/research/
        """
        if base_path is None:
            self.base_path = Path.home() / ".claude" / "research"
        else:
            self.base_path = Path(base_path)

        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def generate_context_file(
        self,
        date: str,
        results: list[dict],
        summaries: Optional[dict[str, str]] = None,
    ) -> Path:
        """
        Create context file for a specific date.

        Args:
            date: Date string (YYYY-MM-DD)
            results: List of research results with topic, status, etc.
            summaries: Optional dict mapping topic -> summary text

        Returns:
            Path to generated context file
        """
        file_path = self.base_path / f"{date}.md"

        # Group results by status
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") != "success"]

        # Build content
        lines = [
            f"# Research Context - {self._format_date(date)}",
            "",
            f"*Auto-generated at {datetime.now().strftime('%H:%M %Z')}*",
            "",
        ]

        if successful:
            lines.append("## Research Findings")
            lines.append("")

            for result in successful:
                topic = result.get("topic", "Unknown")
                count = result.get("results_count", 0)

                lines.append(f"### {topic}")
                lines.append(f"*{count} sources analyzed*")
                lines.append("")

                # Add summary if available
                if summaries and topic in summaries:
                    lines.append(summaries[topic])
                else:
                    lines.append(f"- Researched: {topic}")
                    lines.append(f"- Sources: {count}")

                lines.append("")

        if failed:
            lines.append("## Research Issues")
            lines.append("")
            for result in failed:
                topic = result.get("topic", "Unknown")
                error = result.get("error", result.get("status", "Unknown error"))
                lines.append(f"- **{topic}**: {error}")
            lines.append("")

        lines.append("---")
        lines.append(f"*{len(successful)} topics researched successfully*")

        content = "\n".join(lines)

        # Write file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Generated context file: {file_path}")
        return file_path

    def _format_date(self, date_str: str) -> str:
        """Format date string for display."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%B %d, %Y")
        except ValueError:
            return date_str

    def get_latest_context(self) -> Optional[str]:
        """
        Get content of most recent research context file.

        Returns:
            Content string or None if no files exist
        """
        # Find most recent file
        files = sorted(self.base_path.glob("*.md"), reverse=True)

        if not files:
            return None

        latest = files[0]
        with open(latest, "r", encoding="utf-8") as f:
            return f.read()

    def get_context_for_date(self, date: str) -> Optional[str]:
        """
        Get context content for specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Content string or None if not found
        """
        file_path = self.base_path / f"{date}.md"

        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def list_available_dates(self, limit: int = 30) -> list[str]:
        """
        List dates with available context files.

        Args:
            limit: Maximum number of dates to return

        Returns:
            List of date strings (newest first)
        """
        files = sorted(self.base_path.glob("*.md"), reverse=True)
        dates = []

        for f in files[:limit]:
            # Extract date from filename (YYYY-MM-DD.md)
            date = f.stem
            if len(date) == 10 and date[4] == "-" and date[7] == "-":
                dates.append(date)

        return dates

    def cleanup_old_files(self, keep_days: int = 30) -> int:
        """
        Remove context files older than specified days.

        Args:
            keep_days: Number of days of files to keep

        Returns:
            Number of files deleted
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=keep_days)
        deleted = 0

        for file_path in self.base_path.glob("*.md"):
            try:
                date_str = file_path.stem
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff:
                    file_path.unlink()
                    deleted += 1
                    logger.info(f"Deleted old context file: {file_path.name}")

            except ValueError:
                # Skip files that don't match date format
                continue

        return deleted

    def get_injection_text(self, max_chars: int = 2000) -> str:
        """
        Get formatted text for injection into Claude session.

        Args:
            max_chars: Maximum characters to return

        Returns:
            Formatted injection text
        """
        content = self.get_latest_context()

        if not content:
            return ""

        # Truncate if needed
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n*[Truncated for context limits]*"

        return f"""
<research-context>
{content}
</research-context>
"""
