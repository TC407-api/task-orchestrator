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
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
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


# ====================================================================
# TerminalLoop - Infinite Agent Loop Implementation
# ====================================================================


class LoopState(str, Enum):
    """State of the terminal loop."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LoopIteration:
    """Record of a single loop iteration."""
    iteration_number: int
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    status: str = "pending"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "iteration_number": self.iteration_number,
            "timestamp": self.timestamp.isoformat(),
            "input_data": self.input_data,
            "output_data": self.output_data,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "status": self.status,
        }


class TerminalLoop:
    """
    Infinite agent loop with pause/resume, retry, and persistence.

    Executes a task repeatedly, chaining outputs to inputs, with support for:
    - Pause/resume control
    - Exponential backoff retry
    - Iteration history tracking
    - Persistent state saving
    - Configurable loop conditions
    - Timeout protection

    Usage:
        loop = TerminalLoop(
            name="data_processor",
            max_iterations=100,
            timeout_seconds=300.0,
            persistent=True
        )

        async def process_task(data):
            # Do work
            return {"result": processed_data}

        result = await loop.run({"start": "value"}, process_task)
    """

    def __init__(
        self,
        name: str,
        max_iterations: int = 100,
        timeout_seconds: float = 300.0,
        retry_policy: str = 'exponential_backoff',
        max_retries: int = 3,
        loop_condition: Optional[Callable] = None,
        on_iteration_complete: Optional[Callable] = None,
        persistent: bool = False,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize terminal loop.

        Args:
            name: Loop identifier
            max_iterations: Maximum iterations before stopping
            timeout_seconds: Total timeout for entire run
            retry_policy: Retry strategy ('exponential_backoff' or 'fixed')
            max_retries: Maximum retry attempts per iteration
            loop_condition: Optional callable(iteration) -> bool to control loop
            on_iteration_complete: Optional callback after each iteration
            persistent: Whether to save state to disk
            persistence_path: Path for saving state (defaults to ./{name}_state.json)
        """
        self.name = name
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy
        self.max_retries = max_retries
        self.loop_condition = loop_condition
        self.on_iteration_complete = on_iteration_complete
        self.persistent = persistent
        self.persistence_path = persistence_path or f"./{name}_state.json"

        # Internal state
        self._state = LoopState.IDLE
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._cancel_flag = False
        self._iteration_history: list[LoopIteration] = []
        self._current_iteration: Optional[LoopIteration] = None
        self._total_iterations_completed = 0
        self._total_errors = 0
        self._total_duration_ms = 0.0
        self._running_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> LoopState:
        """Get current loop state."""
        return self._state

    async def run(self, input_data: Dict, task: Callable) -> Dict:
        """
        Run the loop with the given task.

        Args:
            input_data: Initial input data
            task: Callable that receives data and returns output
                  Can be sync or async function

        Returns:
            Final output data after loop completion

        Raises:
            asyncio.TimeoutError: If timeout_seconds exceeded
            asyncio.CancelledError: If loop is cancelled
        """
        self._state = LoopState.RUNNING
        self._cancel_flag = False
        current_input = input_data

        try:
            # Create tracked task for cancellation
            loop_coro = self._run_loop(current_input, task)
            self._running_task = asyncio.create_task(loop_coro)

            # Wrap entire execution in timeout
            return await asyncio.wait_for(
                self._running_task,
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            self._state = LoopState.FAILED
            raise
        except asyncio.CancelledError:
            self._state = LoopState.CANCELLED
            raise
        except Exception as e:
            self._state = LoopState.FAILED
            raise
        finally:
            self._running_task = None

    async def _run_loop(self, input_data: Dict, task: Callable) -> Dict:
        """Internal loop execution."""
        current_input = input_data

        for iteration_num in range(1, self.max_iterations + 1):
            # Yield control to allow pause/cancel from other tasks
            await asyncio.sleep(0)

            # Check for pause
            await self._pause_event.wait()

            # Check for cancellation
            if self._cancel_flag:
                self._state = LoopState.CANCELLED
                return current_input

            # Execute iteration with retry
            iteration = LoopIteration(
                iteration_number=iteration_num,
                timestamp=datetime.now(),
                input_data=current_input
            )
            self._current_iteration = iteration

            start_time = datetime.now()
            should_continue = True

            try:
                output_data = await self._execute_with_retry(task, current_input)

                duration = (datetime.now() - start_time).total_seconds() * 1000
                iteration.output_data = output_data
                iteration.duration_ms = duration
                iteration.status = "success"

                self._total_iterations_completed += 1
                self._total_duration_ms += duration

                # Chain output to next input
                current_input = output_data

                # Check loop condition with result AFTER successful execution
                if self.loop_condition and not self.loop_condition(output_data):
                    should_continue = False

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                iteration.error = str(e)
                iteration.duration_ms = duration
                iteration.status = "failed"
                self._total_errors += 1

                # Continue with same input on error
                pass

            finally:
                self._iteration_history.append(iteration)
                self._current_iteration = None

                # Call completion callback
                if self.on_iteration_complete:
                    try:
                        if asyncio.iscoroutinefunction(self.on_iteration_complete):
                            await self.on_iteration_complete(iteration)
                        else:
                            self.on_iteration_complete(iteration)
                    except Exception:
                        pass  # Don't let callback errors break loop

            # Break if loop_condition returned False
            if not should_continue:
                self._state = LoopState.COMPLETED
                break

        # Loop completed
        self._state = LoopState.COMPLETED

        # Save state if persistent
        if self.persistent:
            self._save_state()

        return current_input

    async def _execute_with_retry(self, task: Callable, input_data: Dict) -> Dict:
        """
        Execute task with retry policy.

        Args:
            task: Task to execute
            input_data: Input data

        Returns:
            Task output

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Execute task (handle both sync and async)
                if asyncio.iscoroutinefunction(task):
                    result = await task(input_data)
                else:
                    result = task(input_data)

                return result

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    # Calculate backoff delay
                    if self.retry_policy == 'exponential_backoff':
                        delay = 1 * (2 ** attempt)
                    else:
                        delay = 1

                    await asyncio.sleep(delay)
                else:
                    # Max retries exhausted
                    raise last_exception

    def pause(self) -> None:
        """Pause the loop after current iteration completes."""
        if self._state == LoopState.RUNNING:
            self._state = LoopState.PAUSED
            self._pause_event.clear()

    def resume(self) -> None:
        """Resume a paused loop."""
        if self._state == LoopState.PAUSED:
            self._state = LoopState.RUNNING
            self._pause_event.set()

    def cancel(self) -> None:
        """Cancel the loop immediately."""
        self._cancel_flag = True
        self._state = LoopState.CANCELLED
        # Cancel the running task if any
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()

    def get_iteration_history(self) -> List[LoopIteration]:
        """
        Get history of all iterations.

        Returns:
            List of LoopIteration objects
        """
        return self._iteration_history.copy()

    def get_statistics(self) -> Dict:
        """
        Get loop statistics.

        Returns:
            Dictionary with statistics
        """
        total_iterations = len(self._iteration_history)
        successful_iterations = sum(
            1 for it in self._iteration_history if it.status == "success"
        )
        failed_iterations = sum(
            1 for it in self._iteration_history if it.status == "failed"
        )

        avg_duration = (
            self._total_duration_ms / total_iterations
            if total_iterations > 0
            else 0.0
        )

        return {
            "name": self.name,
            "state": self._state.value,
            "total_iterations": total_iterations,
            "successful_iterations": successful_iterations,
            "failed_iterations": failed_iterations,
            "total_errors": self._total_errors,
            "average_duration_ms": avg_duration,
            "total_duration_ms": self._total_duration_ms,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
        }

    def _save_state(self) -> None:
        """Save loop state to JSON file."""
        import json

        state_data = {
            "name": self.name,
            "state": self._state.value,
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "iteration_history": [
                it.to_dict() for it in self._iteration_history
            ],
        }

        try:
            with open(self.persistence_path, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception:
            pass  # Silent fail on persistence errors
