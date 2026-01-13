"""Terminal-to-Editor Loop system for error detection and correction.

Captures subprocess output, detects errors with stack traces, and correlates
them to source files for automated fixing. Integrates with UniversalInbox
for publishing error events.

Features:
- Async subprocess monitoring without blocking
- Multi-language stack trace parsing (Python, Node.js, Go, Rust)
- Error correlation to source files and line numbers
- Error event publishing to UniversalInbox
- Fix proposal generation

Usage:
    listener = TerminalListener()
    await listener.run_command(["python", "script.py"])

    async for event in listener.error_events():
        print(f"Error: {event.file_path}:{event.line_number}")
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional
from uuid import uuid4

from .inbox import AgentEvent, EventType, UniversalInbox


class ErrorLanguage(str, Enum):
    """Programming languages supported for error detection."""
    PYTHON = "python"
    NODEJS = "nodejs"
    GO = "go"
    RUST = "rust"
    GENERIC = "generic"


class ErrorSeverity(str, Enum):
    """Severity level of detected errors."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StackTraceLocation:
    """Location in source code from a stack trace."""
    file_path: str
    line_number: int
    column_number: Optional[int] = None
    function_name: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "function_name": self.function_name,
            "code_snippet": self.code_snippet,
        }


@dataclass
class DetectedError:
    """An error detected from subprocess output."""
    error_id: str = field(default_factory=lambda: str(uuid4())[:8])
    language: ErrorLanguage = ErrorLanguage.GENERIC
    severity: ErrorSeverity = ErrorSeverity.ERROR
    error_type: str = "Unknown"
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw_output: str = ""
    stack_trace_locations: list[StackTraceLocation] = field(default_factory=list)
    primary_location: Optional[StackTraceLocation] = None
    exit_code: Optional[int] = None
    command: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "language": self.language.value,
            "severity": self.severity.value,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "raw_output": self.raw_output,
            "stack_trace_locations": [
                loc.to_dict() for loc in self.stack_trace_locations
            ],
            "primary_location": (
                self.primary_location.to_dict() if self.primary_location else None
            ),
            "exit_code": self.exit_code,
            "command": self.command,
        }


@dataclass
class FixProposal:
    """A proposed fix for a detected error."""
    proposal_id: str = field(default_factory=lambda: str(uuid4())[:8])
    error_id: str = ""
    file_path: str = ""
    line_number: int = 0
    suggested_fix: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    fix_category: str = ""  # e.g., "syntax", "import", "type"
    explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "error_id": self.error_id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
            "fix_category": self.fix_category,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
        }


class StackTraceParser:
    """Parses stack traces from various programming languages."""

    # Python traceback pattern: File "path", line N
    PYTHON_PATTERN = re.compile(
        r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+(\w+))?'
    )

    # Node.js pattern: at Function (path:line:col)
    NODEJS_PATTERN = re.compile(
        r'at\s+(?:(\w+)\s+)?\(?([^:]+):(\d+):(\d+)\)?'
    )

    # Go pattern: path:line:col (e.g., ./main.go:10:5: error message)
    GO_PATTERN = re.compile(
        r'(\.?/?[^\s:]+\.go):(\d+):(\d+):\s*(.+)', re.MULTILINE
    )

    # Rust pattern: --> path:line:col
    RUST_PATTERN = re.compile(
        r'-->\s+([^\s:]+):(\d+):(\d+)'
    )

    @staticmethod
    def parse_python_traceback(output: str) -> list[StackTraceLocation]:
        """
        Parse Python traceback and extract file locations.

        Args:
            output: Raw error output

        Returns:
            List of stack trace locations
        """
        locations = []
        for match in StackTraceParser.PYTHON_PATTERN.finditer(output):
            file_path = match.group(1)
            line_number = int(match.group(2))
            function_name = match.group(3)

            locations.append(
                StackTraceLocation(
                    file_path=file_path,
                    line_number=line_number,
                    function_name=function_name,
                )
            )

        return locations

    @staticmethod
    def parse_nodejs_traceback(output: str) -> list[StackTraceLocation]:
        """
        Parse Node.js stack trace and extract file locations.

        Args:
            output: Raw error output

        Returns:
            List of stack trace locations
        """
        locations = []
        for match in StackTraceParser.NODEJS_PATTERN.finditer(output):
            function_name = match.group(1)
            file_path = match.group(2)
            line_number = int(match.group(3))
            column_number = int(match.group(4))

            locations.append(
                StackTraceLocation(
                    file_path=file_path,
                    line_number=line_number,
                    column_number=column_number,
                    function_name=function_name,
                )
            )

        return locations

    @staticmethod
    def parse_go_traceback(output: str) -> list[StackTraceLocation]:
        """
        Parse Go error output and extract file locations.

        Args:
            output: Raw error output

        Returns:
            List of stack trace locations
        """
        locations = []
        for match in StackTraceParser.GO_PATTERN.finditer(output):
            file_path = match.group(1)
            line_number = int(match.group(2))
            column_number = int(match.group(3))

            locations.append(
                StackTraceLocation(
                    file_path=file_path,
                    line_number=line_number,
                    column_number=column_number,
                )
            )

        return locations

    @staticmethod
    def parse_rust_traceback(output: str) -> list[StackTraceLocation]:
        """
        Parse Rust error output and extract file locations.

        Args:
            output: Raw error output

        Returns:
            List of stack trace locations
        """
        locations = []
        for match in StackTraceParser.RUST_PATTERN.finditer(output):
            file_path = match.group(1)
            line_number = int(match.group(2))
            column_number = int(match.group(3))

            locations.append(
                StackTraceLocation(
                    file_path=file_path,
                    line_number=line_number,
                    column_number=column_number,
                )
            )

        return locations

    @staticmethod
    def parse(output: str, language: ErrorLanguage) -> list[StackTraceLocation]:
        """
        Parse stack trace for a given language.

        Args:
            output: Raw error output
            language: Programming language

        Returns:
            List of stack trace locations
        """
        if language == ErrorLanguage.PYTHON:
            return StackTraceParser.parse_python_traceback(output)
        elif language == ErrorLanguage.NODEJS:
            return StackTraceParser.parse_nodejs_traceback(output)
        elif language == ErrorLanguage.GO:
            return StackTraceParser.parse_go_traceback(output)
        elif language == ErrorLanguage.RUST:
            return StackTraceParser.parse_rust_traceback(output)
        else:
            return []


class ErrorCapture:
    """Captures and analyzes errors from subprocess execution."""

    def __init__(self):
        """Initialize error capture."""
        self.last_error: Optional[DetectedError] = None
        self.error_history: list[DetectedError] = []

    def capture_from_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        command: str,
    ) -> Optional[DetectedError]:
        """
        Capture error from subprocess output.

        Args:
            stdout: Standard output
            stderr: Standard error
            exit_code: Process exit code
            command: Command that was run

        Returns:
            DetectedError if error was found, None otherwise
        """
        if exit_code == 0:
            return None

        # Combine stdout and stderr for analysis
        combined_output = f"{stdout}\n{stderr}"

        # Detect language and error type
        language = self._detect_language(combined_output)
        error_type, error_message = self._extract_error_info(
            combined_output, language
        )
        severity = self._determine_severity(exit_code, error_type)

        # Parse stack trace
        stack_trace_locations = StackTraceParser.parse(
            combined_output, language
        )

        # Create error object
        error = DetectedError(
            language=language,
            severity=severity,
            error_type=error_type,
            error_message=error_message,
            raw_output=combined_output,
            stack_trace_locations=stack_trace_locations,
            primary_location=stack_trace_locations[0]
            if stack_trace_locations
            else None,
            exit_code=exit_code,
            command=command,
        )

        self.last_error = error
        self.error_history.append(error)
        return error

    @staticmethod
    def _detect_language(output: str) -> ErrorLanguage:
        """
        Detect programming language from error output.

        Args:
            output: Error output

        Returns:
            Detected ErrorLanguage
        """
        if "Traceback (most recent call last):" in output or "File" in output and "line" in output:
            return ErrorLanguage.PYTHON
        elif "at " in output and ".js:" in output:
            return ErrorLanguage.NODEJS
        elif re.search(r'\.go:\d+:\d+:', output):
            return ErrorLanguage.GO
        elif "-->" in output and ".rs:" in output:
            return ErrorLanguage.RUST
        else:
            return ErrorLanguage.GENERIC

    @staticmethod
    def _extract_error_info(
        output: str, language: ErrorLanguage
    ) -> tuple[str, str]:
        """
        Extract error type and message.

        Args:
            output: Error output
            language: Programming language

        Returns:
            Tuple of (error_type, error_message)
        """
        lines = output.split("\n")

        if language == ErrorLanguage.PYTHON:
            for line in lines:
                if "Error" in line and ":" in line:
                    parts = line.split(":", 1)
                    return (parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
            return ("PythonError", "Unknown error")

        elif language == ErrorLanguage.NODEJS:
            for line in lines:
                if "Error:" in line or "TypeError:" in line or "SyntaxError:" in line:
                    return ("JavaScriptError", line.strip())
            return ("JavaScriptError", "Unknown error")

        elif language == ErrorLanguage.GO:
            for line in lines:
                if "fatal error" in line or "panic:" in line:
                    return ("GoError", line.strip())
            return ("GoError", "Unknown error")

        elif language == ErrorLanguage.RUST:
            for line in lines:
                if "error:" in line or "error[" in line:
                    return ("RustError", line.strip())
            return ("RustError", "Unknown error")

        else:
            return ("UnknownError", output.split("\n")[0] if output else "Unknown")

    @staticmethod
    def _determine_severity(exit_code: int, error_type: str) -> ErrorSeverity:
        """
        Determine error severity based on exit code and type.

        Args:
            exit_code: Process exit code
            error_type: Type of error

        Returns:
            ErrorSeverity level
        """
        if exit_code > 128:
            return ErrorSeverity.CRITICAL

        if "fatal" in error_type.lower() or "panic" in error_type.lower():
            return ErrorSeverity.CRITICAL

        if exit_code > 0:
            return ErrorSeverity.ERROR

        return ErrorSeverity.WARNING


class FixProposer:
    """Generates fix proposals for detected errors."""

    def __init__(self, max_proposals: int = 5):
        """
        Initialize fix proposer.

        Args:
            max_proposals: Maximum number of proposals to generate
        """
        self.max_proposals = max_proposals

    def propose_fixes(self, error: DetectedError) -> list[FixProposal]:
        """
        Generate fix proposals for a detected error.

        Args:
            error: DetectedError to fix

        Returns:
            List of FixProposal objects
        """
        proposals = []

        if not error.primary_location:
            return proposals

        location = error.primary_location

        # Common error patterns and fixes
        if error.language == ErrorLanguage.PYTHON:
            proposals.extend(
                self._propose_python_fixes(error, location)
            )
        elif error.language == ErrorLanguage.NODEJS:
            proposals.extend(
                self._propose_nodejs_fixes(error, location)
            )
        elif error.language == ErrorLanguage.GO:
            proposals.extend(
                self._propose_go_fixes(error, location)
            )
        elif error.language == ErrorLanguage.RUST:
            proposals.extend(
                self._propose_rust_fixes(error, location)
            )

        return proposals[:self.max_proposals]

    @staticmethod
    def _propose_python_fixes(
        error: DetectedError, location: StackTraceLocation
    ) -> list[FixProposal]:
        """Generate Python-specific fix proposals."""
        proposals = []

        if "ImportError" in error.error_type or "ModuleNotFoundError" in error.error_type:
            proposals.append(
                FixProposal(
                    error_id=error.error_id,
                    file_path=location.file_path,
                    line_number=location.line_number,
                    suggested_fix="Check import statement spelling and module availability",
                    confidence=0.7,
                    fix_category="import",
                    explanation="ModuleNotFoundError typically means the module is not installed or the import path is incorrect.",
                )
            )

        if "NameError" in error.error_type:
            proposals.append(
                FixProposal(
                    error_id=error.error_id,
                    file_path=location.file_path,
                    line_number=location.line_number,
                    suggested_fix="Check variable name spelling and ensure it's defined before use",
                    confidence=0.8,
                    fix_category="syntax",
                    explanation="NameError means a variable is used before being defined.",
                )
            )

        if "TypeError" in error.error_type:
            proposals.append(
                FixProposal(
                    error_id=error.error_id,
                    file_path=location.file_path,
                    line_number=location.line_number,
                    suggested_fix="Check function argument types and counts",
                    confidence=0.75,
                    fix_category="type",
                    explanation="TypeError usually indicates wrong types passed to a function.",
                )
            )

        return proposals

    @staticmethod
    def _propose_nodejs_fixes(
        error: DetectedError, location: StackTraceLocation
    ) -> list[FixProposal]:
        """Generate Node.js-specific fix proposals."""
        proposals = []

        if "Cannot find module" in error.error_message:
            proposals.append(
                FixProposal(
                    error_id=error.error_id,
                    file_path=location.file_path,
                    line_number=location.line_number,
                    suggested_fix="Run 'npm install' or check require/import path",
                    confidence=0.8,
                    fix_category="import",
                    explanation="Module not found in node_modules or incorrect path.",
                )
            )

        if "SyntaxError" in error.error_type:
            proposals.append(
                FixProposal(
                    error_id=error.error_id,
                    file_path=location.file_path,
                    line_number=location.line_number,
                    suggested_fix="Check syntax near the error line",
                    confidence=0.85,
                    fix_category="syntax",
                    explanation="JavaScript syntax error detected.",
                )
            )

        return proposals

    @staticmethod
    def _propose_go_fixes(
        error: DetectedError, location: StackTraceLocation
    ) -> list[FixProposal]:
        """Generate Go-specific fix proposals."""
        proposals = []

        if "undefined" in error.error_message.lower():
            proposals.append(
                FixProposal(
                    error_id=error.error_id,
                    file_path=location.file_path,
                    line_number=location.line_number,
                    suggested_fix="Check identifier spelling and import statements",
                    confidence=0.7,
                    fix_category="syntax",
                    explanation="Undefined identifier in Go code.",
                )
            )

        return proposals

    @staticmethod
    def _propose_rust_fixes(
        error: DetectedError, location: StackTraceLocation
    ) -> list[FixProposal]:
        """Generate Rust-specific fix proposals."""
        proposals = []

        if "error[" in error.error_message:
            proposals.append(
                FixProposal(
                    error_id=error.error_id,
                    file_path=location.file_path,
                    line_number=location.line_number,
                    suggested_fix="Review Rust compiler error message for details",
                    confidence=0.65,
                    fix_category="syntax",
                    explanation="Rust compiler error - check error code documentation.",
                )
            )

        return proposals


class TerminalListener:
    """Monitors subprocess execution and captures errors."""

    def __init__(
        self,
        inbox: Optional[UniversalInbox] = None,
        error_callback: Optional[Callable[[DetectedError], None]] = None,
    ):
        """
        Initialize terminal listener.

        Args:
            inbox: UniversalInbox for publishing events. If None, creates new instance.
            error_callback: Optional callback when error is detected
        """
        self.inbox = inbox or UniversalInbox()
        self.error_callback = error_callback
        self.error_capture = ErrorCapture()
        self.fix_proposer = FixProposer()
        self.last_process: Optional[asyncio.subprocess.Process] = None
        self.active_commands: dict[str, asyncio.Task] = {}

    async def run_command(
        self,
        command: list[str],
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> tuple[str, str, int]:
        """
        Run a command and capture output.

        Args:
            command: Command to run as list
            cwd: Working directory
            timeout: Timeout in seconds

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            self.last_process = process
            command_str = " ".join(command)
            command_id = str(uuid4())[:8]
            self.active_commands[command_id] = asyncio.current_task()

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stdout_str = "TIMEOUT"
                stderr_str = f"Command timed out after {timeout} seconds"
                exit_code = -1
            else:
                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")
                exit_code = process.returncode

            # Capture errors
            error = self.error_capture.capture_from_output(
                stdout_str, stderr_str, exit_code, command_str
            )

            if error:
                await self._publish_error_event(error)

                if self.error_callback:
                    self.error_callback(error)

            return stdout_str, stderr_str, exit_code

        finally:
            self.active_commands.pop(command_id, None)

    async def _publish_error_event(self, error: DetectedError) -> None:
        """
        Publish error event to inbox.

        Args:
            error: DetectedError to publish
        """
        event = AgentEvent(
            event_type=EventType.ERROR,
            agent_name="terminal_listener",
            data=error.to_dict(),
            source="terminal_loop",
        )

        await self.inbox.publish(event)

    async def error_events(self) -> AsyncGenerator[DetectedError, None]:
        """
        Subscribe to error events.

        Yields:
            DetectedError objects as they're captured
        """
        async for event in self.inbox.subscribe():
            if (
                event.event_type == EventType.ERROR
                and event.source == "terminal_loop"
            ):
                error_data = event.data
                error = DetectedError(
                    error_id=error_data.get("error_id"),
                    language=ErrorLanguage(
                        error_data.get("language", "generic")
                    ),
                    severity=ErrorSeverity(
                        error_data.get("severity", "error")
                    ),
                    error_type=error_data.get("error_type", "Unknown"),
                    error_message=error_data.get("error_message", ""),
                    timestamp=datetime.fromisoformat(
                        error_data.get("timestamp")
                    ),
                    raw_output=error_data.get("raw_output", ""),
                    exit_code=error_data.get("exit_code"),
                    command=error_data.get("command"),
                )

                # Reconstruct stack trace locations
                for loc_data in error_data.get("stack_trace_locations", []):
                    error.stack_trace_locations.append(
                        StackTraceLocation(
                            file_path=loc_data.get("file_path", ""),
                            line_number=loc_data.get("line_number", 0),
                            column_number=loc_data.get("column_number"),
                            function_name=loc_data.get("function_name"),
                            code_snippet=loc_data.get("code_snippet"),
                        )
                    )

                yield error

    def get_fix_proposals(
        self, error: Optional[DetectedError] = None
    ) -> list[FixProposal]:
        """
        Get fix proposals for an error.

        Args:
            error: DetectedError. If None, uses last captured error.

        Returns:
            List of FixProposal objects
        """
        target_error = error or self.last_error

        if not target_error:
            return []

        return self.fix_proposer.propose_fixes(target_error)

    @property
    def last_error(self) -> Optional[DetectedError]:
        """Get the last captured error."""
        return self.error_capture.last_error

    @property
    def error_history(self) -> list[DetectedError]:
        """Get error history."""
        return self.error_capture.error_history

    async def clear_history(self) -> None:
        """Clear error history."""
        self.error_capture.error_history.clear()
