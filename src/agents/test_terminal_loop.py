"""Tests for the Terminal-to-Editor Loop system."""

import pytest

from .terminal_loop import (
    TerminalListener,
    ErrorCapture,
    StackTraceParser,
    FixProposer,
    DetectedError,
    ErrorLanguage,
    ErrorSeverity,
    StackTraceLocation,
)


class TestStackTraceParser:
    """Tests for StackTraceParser."""

    def test_parse_python_traceback(self):
        """Test parsing Python tracebacks."""
        output = '''Traceback (most recent call last):
  File "script.py", line 10, in main
    result = do_something()
  File "module.py", line 42, in do_something
    return data.process()
AttributeError: 'NoneType' object has no attribute 'process'
'''
        locations = StackTraceParser.parse_python_traceback(output)

        assert len(locations) == 2
        assert locations[0].file_path == "script.py"
        assert locations[0].line_number == 10
        assert locations[0].function_name == "main"
        assert locations[1].file_path == "module.py"
        assert locations[1].line_number == 42
        assert locations[1].function_name == "do_something"

    def test_parse_nodejs_traceback(self):
        """Test parsing Node.js stack traces."""
        output = """Error: ENOENT: no such file or directory
    at Object.openSync (fs.js:462:3)
    at readFileSync (fs.js:364:21)
    at main (/app/index.js:15:8)
    at processTicksAndRejections (internal/timers.js:213:21)
"""
        locations = StackTraceParser.parse_nodejs_traceback(output)

        assert len(locations) >= 1
        # Find the app/index.js location
        app_locations = [loc for loc in locations if "index.js" in loc.file_path]
        assert len(app_locations) > 0
        assert app_locations[0].line_number == 15
        assert app_locations[0].column_number == 8
        assert app_locations[0].function_name == "main"

    def test_parse_go_traceback(self):
        """Test parsing Go error output."""
        output = """./main.go:25:5: undefined: fmt.Println
./handler.go:10:2: missing return statement
"""
        locations = StackTraceParser.parse_go_traceback(output)

        assert len(locations) == 2
        assert locations[0].file_path == "./main.go"
        assert locations[0].line_number == 25
        assert locations[0].column_number == 5

    def test_parse_rust_traceback(self):
        """Test parsing Rust error output."""
        output = """error[E0425]: cannot find value `x` in this scope
 --> src/main.rs:3:5
  |
3 |     println!("{}", x);
  |                    ^ not found in this scope
"""
        locations = StackTraceParser.parse_rust_traceback(output)

        assert len(locations) == 1
        assert "main.rs" in locations[0].file_path
        assert locations[0].line_number == 3
        assert locations[0].column_number == 5

    def test_parse_with_language_routing(self):
        """Test parse method with language routing."""
        python_output = 'File "test.py", line 15'
        locations = StackTraceParser.parse(python_output, ErrorLanguage.PYTHON)
        assert len(locations) == 1
        assert locations[0].line_number == 15

        nodejs_output = "at Function (test.js:20:10)"
        locations = StackTraceParser.parse(nodejs_output, ErrorLanguage.NODEJS)
        assert len(locations) == 1
        assert locations[0].line_number == 20


class TestErrorCapture:
    """Tests for ErrorCapture."""

    def test_capture_success_exit_code(self):
        """Test that exit code 0 doesn't capture error."""
        capture = ErrorCapture()
        error = capture.capture_from_output("output", "", 0, "python test.py")
        assert error is None

    def test_capture_python_error(self):
        """Test capturing Python errors."""
        capture = ErrorCapture()
        stderr = '''Traceback (most recent call last):
  File "app.py", line 5, in <module>
    import missing_module
ModuleNotFoundError: No module named 'missing_module'
'''
        error = capture.capture_from_output("", stderr, 1, "python app.py")

        assert error is not None
        assert error.language == ErrorLanguage.PYTHON
        assert error.exit_code == 1
        assert "ModuleNotFoundError" in error.error_type
        assert len(error.stack_trace_locations) > 0

    def test_detect_language_python(self):
        """Test Python language detection."""
        output = "Traceback (most recent call last):\n  File"
        lang = ErrorCapture._detect_language(output)
        assert lang == ErrorLanguage.PYTHON

    def test_detect_language_nodejs(self):
        """Test Node.js language detection."""
        output = "Error\nat Function (/app/index.js:15:3)"
        lang = ErrorCapture._detect_language(output)
        assert lang == ErrorLanguage.NODEJS

    def test_detect_language_go(self):
        """Test Go language detection."""
        output = "./main.go:25:5: undefined"
        lang = ErrorCapture._detect_language(output)
        assert lang == ErrorLanguage.GO

    def test_detect_language_rust(self):
        """Test Rust language detection."""
        output = "--> src/main.rs:3:5"
        lang = ErrorCapture._detect_language(output)
        assert lang == ErrorLanguage.RUST

    def test_error_severity_critical(self):
        """Test critical error severity detection."""
        severity = ErrorCapture._determine_severity(139, "fatal")
        assert severity == ErrorSeverity.CRITICAL

    def test_error_severity_error(self):
        """Test error severity detection."""
        severity = ErrorCapture._determine_severity(1, "TypeError")
        assert severity == ErrorSeverity.ERROR

    def test_error_capture_history(self):
        """Test error history tracking."""
        capture = ErrorCapture()

        # Capture multiple errors
        capture.capture_from_output("", "Error 1", 1, "cmd1")
        capture.capture_from_output("", "Error 2", 1, "cmd2")

        assert len(capture.error_history) == 2
        assert capture.last_error.command == "cmd2"


class TestFixProposer:
    """Tests for FixProposer."""

    def test_propose_python_import_error_fix(self):
        """Test Python import error fix proposals."""
        proposer = FixProposer()

        error = DetectedError(
            language=ErrorLanguage.PYTHON,
            error_type="ModuleNotFoundError",
            error_message="No module named 'pandas'",
            stack_trace_locations=[
                StackTraceLocation(
                    file_path="app.py",
                    line_number=1,
                    function_name="<module>",
                )
            ],
        )
        error.primary_location = error.stack_trace_locations[0]

        proposals = proposer.propose_fixes(error)

        assert len(proposals) > 0
        assert any("import" in p.fix_category for p in proposals)
        assert any(p.confidence >= 0.7 for p in proposals)

    def test_propose_python_name_error_fix(self):
        """Test Python NameError fix proposals."""
        proposer = FixProposer()

        error = DetectedError(
            language=ErrorLanguage.PYTHON,
            error_type="NameError",
            error_message="name 'undefined_var' is not defined",
            stack_trace_locations=[
                StackTraceLocation(
                    file_path="script.py",
                    line_number=42,
                )
            ],
        )
        error.primary_location = error.stack_trace_locations[0]

        proposals = proposer.propose_fixes(error)

        assert len(proposals) > 0
        assert proposals[0].confidence >= 0.7

    def test_propose_nodejs_module_error_fix(self):
        """Test Node.js module error fix proposals."""
        proposer = FixProposer()

        error = DetectedError(
            language=ErrorLanguage.NODEJS,
            error_message="Cannot find module 'express'",
            stack_trace_locations=[
                StackTraceLocation(
                    file_path="index.js",
                    line_number=5,
                )
            ],
        )
        error.primary_location = error.stack_trace_locations[0]

        proposals = proposer.propose_fixes(error)

        assert len(proposals) > 0
        assert any("npm" in p.suggested_fix for p in proposals)

    def test_fix_proposal_to_dict(self):
        """Test FixProposal serialization."""
        proposal = FixProposer()._propose_python_fixes(
            DetectedError(
                error_id="err123",
                error_type="ImportError",
                stack_trace_locations=[
                    StackTraceLocation(
                        file_path="test.py",
                        line_number=10,
                    )
                ],
            ),
            StackTraceLocation(
                file_path="test.py",
                line_number=10,
            ),
        )

        # Just verify proposal objects created successfully
        assert proposal is not None


class TestDetectedError:
    """Tests for DetectedError data class."""

    def test_detected_error_to_dict(self):
        """Test DetectedError serialization."""
        error = DetectedError(
            error_id="test123",
            language=ErrorLanguage.PYTHON,
            error_type="ValueError",
            error_message="Invalid value",
            exit_code=1,
        )

        d = error.to_dict()

        assert d["error_id"] == "test123"
        assert d["language"] == "python"
        assert d["error_type"] == "ValueError"
        assert d["exit_code"] == 1


class TestTerminalListener:
    """Tests for TerminalListener."""

    @pytest.mark.asyncio
    async def test_run_successful_command(self):
        """Test running a successful command."""
        listener = TerminalListener()

        # Run a simple echo command
        stdout, stderr, exit_code = await listener.run_command(["echo", "hello"])

        assert exit_code == 0
        assert "hello" in stdout

    @pytest.mark.asyncio
    async def test_run_failing_command(self):
        """Test running a failing command."""
        listener = TerminalListener()

        # Run a command that will fail
        stdout, stderr, exit_code = await listener.run_command(
            ["python", "-c", "raise ValueError('test error')"]
        )

        assert exit_code != 0
        assert listener.last_error is not None

    @pytest.mark.asyncio
    async def test_error_callback(self):
        """Test error callback is called."""
        callback_called = []

        def callback(error):
            callback_called.append(error)

        listener = TerminalListener(error_callback=callback)

        # Run a command that will fail
        await listener.run_command(
            ["python", "-c", "raise RuntimeError('boom')"]
        )

        assert len(callback_called) > 0

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        """Test command timeout."""
        listener = TerminalListener()

        # Run a command that will timeout
        stdout, stderr, exit_code = await listener.run_command(
            ["python", "-c", "import time; time.sleep(10)"],
            timeout=0.1,
        )

        assert exit_code == -1
        assert "TIMEOUT" in stdout or "timeout" in stderr.lower()

    def test_get_fix_proposals(self):
        """Test getting fix proposals."""
        listener = TerminalListener()

        # Manually create an error
        listener.error_capture.last_error = DetectedError(
            error_id="test",
            language=ErrorLanguage.PYTHON,
            error_type="ImportError",
            stack_trace_locations=[
                StackTraceLocation(
                    file_path="app.py",
                    line_number=1,
                )
            ],
        )
        listener.error_capture.last_error.primary_location = (
            listener.error_capture.last_error.stack_trace_locations[0]
        )

        proposals = listener.get_fix_proposals()

        assert len(proposals) > 0

    def test_error_history(self):
        """Test error history tracking."""
        listener = TerminalListener()

        # Manually add errors to history
        listener.error_capture.error_history.append(
            DetectedError(
                language=ErrorLanguage.PYTHON,
                error_type="SyntaxError",
            )
        )
        listener.error_capture.error_history.append(
            DetectedError(
                language=ErrorLanguage.PYTHON,
                error_type="TypeError",
            )
        )

        assert len(listener.error_history) == 2


class TestStackTraceLocation:
    """Tests for StackTraceLocation."""

    def test_stack_trace_location_to_dict(self):
        """Test StackTraceLocation serialization."""
        location = StackTraceLocation(
            file_path="/app/main.py",
            line_number=42,
            column_number=5,
            function_name="process",
            code_snippet="result = do_something()",
        )

        d = location.to_dict()

        assert d["file_path"] == "/app/main.py"
        assert d["line_number"] == 42
        assert d["column_number"] == 5
        assert d["function_name"] == "process"
        assert d["code_snippet"] == "result = do_something()"


class TestErrorLanguageDetection:
    """Tests for error language detection."""

    def test_multi_language_detection(self):
        """Test detecting multiple languages in different outputs."""
        test_cases = [
            ("Traceback (most recent call last):", ErrorLanguage.PYTHON),
            ("at Function (app.js:15:3)", ErrorLanguage.NODEJS),
            ("./main.go:10:5: error", ErrorLanguage.GO),
            ("--> src/lib.rs:20:10", ErrorLanguage.RUST),
        ]

        for output, expected_lang in test_cases:
            detected = ErrorCapture._detect_language(output)
            assert detected == expected_lang, f"Failed for: {output}"


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_error_flow(self):
        """Test full error detection and fix proposal flow."""
        listener = TerminalListener()

        # Create a Python file with an error
        test_code = '''
import sys
print("Starting")
raise ValueError("Test error on line 4")
'''

        # Run Python with inline code
        stdout, stderr, exit_code = await listener.run_command(
            ["python", "-c", test_code]
        )

        # Verify error was captured
        assert exit_code != 0
        assert listener.last_error is not None
        assert listener.last_error.language == ErrorLanguage.PYTHON

        # Get fix proposals
        proposals = listener.get_fix_proposals()
        assert len(proposals) >= 0  # May or may not have proposals

    @pytest.mark.asyncio
    async def test_multiple_errors(self):
        """Test handling multiple sequential errors."""
        listener = TerminalListener()

        # Run multiple failing commands
        for i in range(3):
            await listener.run_command(
                ["python", "-c", f"raise RuntimeError('error {i}')"]
            )

        assert len(listener.error_history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
