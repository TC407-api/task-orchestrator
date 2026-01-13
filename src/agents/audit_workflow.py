"""Audit workflow for maintaining persistent memory of architectural decisions and past errors."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import re


class AuditWorkflow:
    """
    Manages persistent audit history for agent decisions and error patterns.

    Responsibilities:
    - Load and maintain audit.md from project root
    - Inject audit history into agent system prompts
    - Record new errors and fixes after agent execution
    - Query historical decisions and patterns for conflict detection
    """

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the audit workflow.

        Args:
            project_root: Path to project root. If None, uses current directory.
        """
        self.project_root = Path(project_root or os.getcwd())
        self.audit_file = self.project_root / "audit.md"
        self.audit_data = self._load_or_initialize()

    def _load_or_initialize(self) -> dict:
        """
        Load audit.md if it exists, otherwise initialize with empty structure.

        Returns:
            Dict with audit sections: metadata, decisions, errors, patterns
        """
        if self.audit_file.exists():
            return self._parse_audit_md()
        else:
            return {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0",
                },
                "decisions": [],  # Architectural decisions
                "errors": [],     # Recorded errors and fixes
                "patterns": [],   # Recurring patterns/conflicts
            }

    def _parse_audit_md(self) -> dict:
        """
        Parse existing audit.md file into structured format.

        Returns:
            Dict with parsed sections
        """
        with open(self.audit_file, "r") as f:
            content = f.read()

        audit_data = {
            "metadata": {},
            "decisions": [],
            "errors": [],
            "patterns": [],
            "_raw": content,  # Preserve raw content
        }

        # Parse metadata section
        metadata_match = re.search(
            r"# Audit Log\n+(?:## Metadata\n)?(.*?)(?=## \w+|$)",
            content,
            re.DOTALL,
        )
        if metadata_match:
            meta_block = metadata_match.group(1).strip()
            audit_data["metadata"] = self._parse_metadata_block(meta_block)

        # Parse decisions section
        decisions_match = re.search(
            r"## Decisions\n+(.*?)(?=## \w+|$)",
            content,
            re.DOTALL,
        )
        if decisions_match:
            audit_data["decisions"] = self._parse_entries(decisions_match.group(1))

        # Parse errors section
        errors_match = re.search(
            r"## Errors & Fixes\n+(.*?)(?=## \w+|$)",
            content,
            re.DOTALL,
        )
        if errors_match:
            audit_data["errors"] = self._parse_entries(errors_match.group(1))

        # Parse patterns section
        patterns_match = re.search(
            r"## Patterns & Conflicts\n+(.*?)(?=## \w+|$)",
            content,
            re.DOTALL,
        )
        if patterns_match:
            audit_data["patterns"] = self._parse_entries(patterns_match.group(1))

        return audit_data

    def _parse_metadata_block(self, block: str) -> dict:
        """Parse metadata key-value pairs."""
        metadata = {}
        for line in block.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        return metadata

    def _parse_entries(self, block: str) -> list[dict]:
        """Parse entry blocks (decisions/errors/patterns)."""
        entries = []
        # Match entry patterns with title and content
        entry_pattern = r"- \*\*(.+?)\*\*\s*(?:\(([^)]+)\))?\s*:\s*(.*?)(?=- \*\*|\Z)"
        matches = re.finditer(entry_pattern, block, re.DOTALL)

        for match in matches:
            title = match.group(1)
            metadata = match.group(2) or ""
            content = match.group(3).strip()

            entry = {
                "title": title,
                "content": content,
                "timestamp": self._extract_timestamp(metadata),
                "metadata": metadata,
            }
            entries.append(entry)

        return entries

    def _extract_timestamp(self, metadata: str) -> str:
        """Extract ISO timestamp from metadata string."""
        # Try to extract ISO format date/timestamp
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        match = re.search(iso_pattern, metadata)
        if match:
            return match.group(0)
        return datetime.now().isoformat()

    def load_audit(self) -> dict:
        """
        Load the complete audit history.

        Returns:
            Dict with all audit sections
        """
        return self.audit_data.copy()

    def inject_to_prompt(self, prompt: str) -> str:
        """
        Inject audit history context into an agent's system prompt.

        This adds relevant past decisions and error patterns to help the agent
        avoid repeating mistakes and maintain consistency.

        Args:
            prompt: Original system prompt

        Returns:
            Augmented prompt with audit context
        """
        if not self.audit_data.get("decisions") and not self.audit_data.get("errors"):
            return prompt

        audit_context = "\n\n## Audit History Context\n"

        # Add recent decisions if they exist
        decisions = self.audit_data.get("decisions", [])
        if decisions:
            audit_context += "\n### Past Architectural Decisions:\n"
            # Include last 5 decisions
            for decision in decisions[-5:]:
                title = decision.get("title", "")
                content = decision.get("content", "")
                audit_context += f"- **{title}**: {content}\n"

        # Add recent error patterns if they exist
        errors = self.audit_data.get("errors", [])
        if errors:
            audit_context += "\n### Known Error Patterns & Fixes:\n"
            # Include last 5 errors
            for error in errors[-5:]:
                title = error.get("title", "")
                content = error.get("content", "")
                audit_context += f"- **{title}**: {content}\n"

        # Add conflict patterns if they exist
        patterns = self.audit_data.get("patterns", [])
        if patterns:
            audit_context += "\n### Known Conflicting Patterns (AVOID):\n"
            for pattern in patterns[-3:]:
                title = pattern.get("title", "")
                content = pattern.get("content", "")
                audit_context += f"- **{title}**: {content}\n"

        # Append audit context to the prompt
        return f"{prompt}{audit_context}"

    def append_entry(
        self,
        entry_type: str,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Append a new entry to the audit log.

        Args:
            entry_type: Type of entry ('decision', 'error', 'pattern')
            content: Main content/description
            title: Optional title for the entry
            metadata: Optional metadata dict (tags, severity, etc.)
        """
        if entry_type not in ["decision", "error", "pattern"]:
            raise ValueError(
                f"Invalid entry_type: {entry_type}. Must be 'decision', 'error', or 'pattern'"
            )

        # Map entry type to section key
        section_key = {
            "decision": "decisions",
            "error": "errors",
            "pattern": "patterns",
        }[entry_type]

        # Generate title if not provided
        if not title:
            title = f"Entry {len(self.audit_data[section_key]) + 1}"

        # Build entry
        entry = {
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Add to appropriate section
        self.audit_data[section_key].append(entry)

        # Persist to file
        self._write_audit_md()

    def append_error(
        self,
        error_description: str,
        fix_description: str,
        error_type: str = "runtime",
        severity: str = "medium",
    ) -> None:
        """
        Append an error and its fix to the audit log.

        Args:
            error_description: What went wrong
            fix_description: How it was fixed
            error_type: Type of error (runtime, logic, api, integration, etc.)
            severity: Severity level (low, medium, high, critical)
        """
        combined_content = f"**Error**: {error_description}\n\n**Fix**: {fix_description}"
        self.append_entry(
            entry_type="error",
            content=combined_content,
            title=f"{error_type.title()} Error ({severity.upper()})",
            metadata={"type": error_type, "severity": severity},
        )

    def append_decision(
        self,
        decision_title: str,
        rationale: str,
        context: Optional[str] = None,
    ) -> None:
        """
        Append an architectural decision to the audit log.

        Args:
            decision_title: Title of the decision
            rationale: Why this decision was made
            context: Optional context (what problem was being solved)
        """
        content = rationale
        if context:
            content = f"**Context**: {context}\n\n**Rationale**: {rationale}"

        self.append_entry(
            entry_type="decision",
            content=content,
            title=decision_title,
            metadata={"context": context},
        )

    def append_pattern(
        self,
        pattern_name: str,
        pattern_description: str,
        recommendation: str,
    ) -> None:
        """
        Append a conflicting pattern or recurring issue to audit log.

        Args:
            pattern_name: Name of the pattern
            pattern_description: Description of the pattern
            recommendation: How to avoid or handle it
        """
        content = f"**Pattern**: {pattern_description}\n\n**Recommendation**: {recommendation}"
        self.append_entry(
            entry_type="pattern",
            content=content,
            title=pattern_name,
        )

    def query_decisions(self, topic: str) -> list[dict]:
        """
        Query historical decisions by topic.

        Args:
            topic: Topic to search for (case-insensitive)

        Returns:
            List of matching decision entries
        """
        topic_lower = topic.lower()
        matches = []

        for decision in self.audit_data.get("decisions", []):
            title = decision.get("title", "").lower()
            content = decision.get("content", "").lower()

            if topic_lower in title or topic_lower in content:
                matches.append(decision)

        return matches

    def query_errors(self, error_type: Optional[str] = None) -> list[dict]:
        """
        Query historical errors, optionally filtered by type.

        Args:
            error_type: Optional error type filter (runtime, logic, api, etc.)

        Returns:
            List of matching error entries
        """
        errors = self.audit_data.get("errors", [])

        if error_type:
            return [
                e for e in errors
                if e.get("metadata", {}).get("type", "").lower() == error_type.lower()
            ]

        return errors

    def query_patterns(self, pattern_name: Optional[str] = None) -> list[dict]:
        """
        Query known conflict patterns.

        Args:
            pattern_name: Optional pattern name to search for (case-insensitive)

        Returns:
            List of matching pattern entries
        """
        patterns = self.audit_data.get("patterns", [])

        if pattern_name:
            pattern_lower = pattern_name.lower()
            return [
                p for p in patterns
                if pattern_lower in p.get("title", "").lower()
                or pattern_lower in p.get("content", "").lower()
            ]

        return patterns

    def check_for_conflicts(self, proposed_decision: str) -> list[dict]:
        """
        Check if a proposed decision conflicts with known patterns.

        Args:
            proposed_decision: Description of the proposed decision/action

        Returns:
            List of conflicting patterns found
        """
        conflicts = []
        proposed_lower = proposed_decision.lower()

        for pattern in self.audit_data.get("patterns", []):
            pattern_desc = pattern.get("content", "").lower()
            # Simple keyword overlap detection
            if any(
                word in proposed_lower
                for word in pattern_desc.split()
                if len(word) > 3
            ):
                conflicts.append(pattern)

        return conflicts

    def _write_audit_md(self) -> None:
        """Write audit data back to audit.md file."""
        lines = ["# Audit Log\n"]

        # Write metadata
        lines.append("## Metadata\n")
        for key, value in self.audit_data.get("metadata", {}).items():
            if key != "created":  # Skip created, will be in file
                lines.append(f"- **{key}**: {value}")
        lines.append(f"- **last_updated**: {datetime.now().isoformat()}\n")

        # Write decisions
        if self.audit_data.get("decisions"):
            lines.append("## Decisions\n")
            for decision in self.audit_data["decisions"]:
                lines.append(
                    f"- **{decision['title']}** ({decision['timestamp']}): "
                    f"{decision['content']}\n"
                )
            lines.append("\n")

        # Write errors
        if self.audit_data.get("errors"):
            lines.append("## Errors & Fixes\n")
            for error in self.audit_data["errors"]:
                severity = error.get("metadata", {}).get("severity", "unknown")
                lines.append(
                    f"- **{error['title']}** ({error['timestamp']}, "
                    f"severity: {severity}): {error['content']}\n"
                )
            lines.append("\n")

        # Write patterns
        if self.audit_data.get("patterns"):
            lines.append("## Patterns & Conflicts\n")
            for pattern in self.audit_data["patterns"]:
                lines.append(
                    f"- **{pattern['title']}** ({pattern['timestamp']}): "
                    f"{pattern['content']}\n"
                )
            lines.append("\n")

        # Write summary stats
        lines.append("## Summary\n")
        lines.append(
            f"- **Total Decisions**: {len(self.audit_data.get('decisions', []))}\n"
        )
        lines.append(
            f"- **Total Errors Logged**: {len(self.audit_data.get('errors', []))}\n"
        )
        lines.append(
            f"- **Known Patterns**: {len(self.audit_data.get('patterns', []))}\n"
        )

        # Write to file
        with open(self.audit_file, "w") as f:
            f.writelines(lines)

    def export_json(self, filepath: Optional[str] = None) -> str:
        """
        Export audit data to JSON format.

        Args:
            filepath: Optional path to write JSON file

        Returns:
            JSON string of audit data
        """
        json_str = json.dumps(self.audit_data, indent=2, default=str)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    def get_summary(self) -> dict:
        """
        Get a summary of the audit log.

        Returns:
            Dict with counts and recent entries
        """
        decisions = self.audit_data.get("decisions", [])
        errors = self.audit_data.get("errors", [])
        patterns = self.audit_data.get("patterns", [])

        return {
            "total_decisions": len(decisions),
            "total_errors": len(errors),
            "total_patterns": len(patterns),
            "recent_decisions": decisions[-3:] if decisions else [],
            "recent_errors": errors[-3:] if errors else [],
            "recent_patterns": patterns[-3:] if patterns else [],
            "last_updated": datetime.now().isoformat(),
        }
