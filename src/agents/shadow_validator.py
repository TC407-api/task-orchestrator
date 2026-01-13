"""
Shadow AST Validator for Agent-Generated Code.

Validates agent-generated code BEFORE showing to user:
1. ShadowValidator class with validate(code, language) method
2. Support Python (ast.parse), JavaScript (basic pattern), TypeScript, JSON
3. LinterRunner that shells out to available linters (ruff, eslint, tsc)
4. ValidationResult dataclass with: valid, errors, warnings, suggestions

Flow:
1. Agent generates code diff
2. ShadowValidator parses for syntax errors
3. LinterRunner checks style/type issues
4. Only present to user if validation passes (or show warnings)

Integration points:
- Can be used by spawn_agent before returning response
- Publishes validation events to UniversalInbox
"""

import ast
import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JSON = "json"


class SeverityLevel(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue (error, warning, or info)."""
    severity: SeverityLevel
    line: int
    column: int
    message: str
    code: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "code": self.code,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    language: Language
    issues: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    lint_results: Dict[str, Any] = field(default_factory=dict)
    parse_time_ms: float = 0.0
    lint_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "language": self.language.value,
            "issues": [i.to_dict() for i in self.issues],
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "suggestions": self.suggestions,
            "lint_results": self.lint_results,
            "parse_time_ms": self.parse_time_ms,
            "lint_time_ms": self.lint_time_ms,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


class SyntaxValidator:
    """Base class for language-specific syntax validators."""

    def validate(self, code: str) -> List[ValidationIssue]:
        """Validate code syntax. Returns list of issues."""
        raise NotImplementedError


class PythonValidator(SyntaxValidator):
    """Validates Python code using ast.parse."""

    def validate(self, code: str) -> List[ValidationIssue]:
        """
        Validate Python syntax.

        Args:
            code: Python source code to validate

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(
                ValidationIssue(
                    severity=SeverityLevel.ERROR,
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    message=f"Syntax error: {e.msg}",
                    code="SYNTAX_ERROR",
                    suggestion=self._suggest_fix(e),
                )
            )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=SeverityLevel.ERROR,
                    line=0,
                    column=0,
                    message=f"Parse error: {str(e)}",
                    code="PARSE_ERROR",
                )
            )

        # Check for common anti-patterns
        issues.extend(self._check_patterns(code))

        return issues

    def _check_patterns(self, code: str) -> List[ValidationIssue]:
        """Check for common Python anti-patterns."""
        issues = []
        lines = code.split("\n")

        for line_no, line in enumerate(lines, 1):
            # Check for bare except
            if re.search(r"except\s*:", line):
                issues.append(
                    ValidationIssue(
                        severity=SeverityLevel.WARNING,
                        line=line_no,
                        column=line.find("except"),
                        message="Bare except clause catches all exceptions",
                        code="BARE_EXCEPT",
                        suggestion="Specify exception type: except Exception:",
                    )
                )

            # Check for hardcoded passwords/secrets
            if re.search(r"(?i)(password|api_?key|secret|token)\s*=\s*['\"]", line):
                issues.append(
                    ValidationIssue(
                        severity=SeverityLevel.WARNING,
                        line=line_no,
                        column=0,
                        message="Hardcoded secret detected",
                        code="HARDCODED_SECRET",
                        suggestion="Use environment variables for secrets",
                    )
                )

        return issues

    @staticmethod
    def _suggest_fix(error: SyntaxError) -> Optional[str]:
        """Generate a suggestion for fixing the syntax error."""
        msg = error.msg.lower()

        if "expected ':'" in msg:
            return "Check if you're missing a colon at end of line"
        elif "invalid syntax" in msg and "==" in msg:
            return "Check for assignment (=) vs comparison (==)"
        elif "unmatched" in msg:
            return "Check for unclosed parentheses, brackets, or braces"

        return None


class JavaScriptValidator(SyntaxValidator):
    """Validates JavaScript code using regex patterns."""

    def validate(self, code: str) -> List[ValidationIssue]:
        """
        Validate JavaScript syntax using regex patterns.

        Args:
            code: JavaScript source code to validate

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        # Check for unclosed braces/parens
        issues.extend(self._check_brackets(code))

        # Check for common JS issues
        issues.extend(self._check_patterns(code))

        return issues

    def _check_brackets(self, code: str) -> List[ValidationIssue]:
        """Check for unclosed brackets."""
        issues = []

        # Simple bracket matching
        stack = []
        bracket_map = {"(": ")", "{": "}", "[": "]"}
        close_map = {")": "(", "}": "{", "]": "["}

        for line_no, line in enumerate(code.split("\n"), 1):
            # Skip comments
            line_clean = re.sub(r"//.*$", "", line)
            line_clean = re.sub(r"/\*.*?\*/", "", line_clean, flags=re.DOTALL)

            for col, char in enumerate(line_clean):
                if char in bracket_map:
                    stack.append((char, line_no, col))
                elif char in close_map:
                    if stack and stack[-1][0] == close_map[char]:
                        stack.pop()
                    else:
                        issues.append(
                            ValidationIssue(
                                severity=SeverityLevel.ERROR,
                                line=line_no,
                                column=col,
                                message=f"Unmatched closing bracket: {char}",
                                code="UNMATCHED_BRACKET",
                            )
                        )

        # Report unclosed brackets
        for bracket, line_no, col in stack:
            issues.append(
                ValidationIssue(
                    severity=SeverityLevel.ERROR,
                    line=line_no,
                    column=col,
                    message=f"Unclosed bracket: {bracket}",
                    code="UNCLOSED_BRACKET",
                )
            )

        return issues

    def _check_patterns(self, code: str) -> List[ValidationIssue]:
        """Check for common JavaScript anti-patterns."""
        issues = []
        lines = code.split("\n")

        for line_no, line in enumerate(lines, 1):
            # Check for == instead of ===
            if "==" in line and "===" not in line and "http" not in line:
                matches = re.finditer(r"([^=!<>])={2}([^=])", line)
                for match in matches:
                    issues.append(
                        ValidationIssue(
                            severity=SeverityLevel.WARNING,
                            line=line_no,
                            column=match.start(),
                            message="Use === instead of ==",
                            code="LOOSE_EQUALITY",
                            suggestion="Replace == with ===",
                        )
                    )

            # Check for var instead of const/let
            if re.match(r"\s*var\s+", line):
                issues.append(
                    ValidationIssue(
                        severity=SeverityLevel.WARNING,
                        line=line_no,
                        column=line.find("var"),
                        message="Use 'const' or 'let' instead of 'var'",
                        code="VAR_USAGE",
                        suggestion="Replace var with const or let",
                    )
                )

        return issues


class TypeScriptValidator(JavaScriptValidator):
    """Validates TypeScript code (extends JavaScript validator)."""

    def validate(self, code: str) -> List[ValidationIssue]:
        """
        Validate TypeScript syntax.

        Args:
            code: TypeScript source code to validate

        Returns:
            List of ValidationIssue objects
        """
        # First run JavaScript validation
        issues = super().validate(code)

        # Add TypeScript-specific checks
        issues.extend(self._check_typescript_patterns(code))

        return issues

    def _check_typescript_patterns(self, code: str) -> List[ValidationIssue]:
        """Check for TypeScript-specific patterns."""
        issues = []
        lines = code.split("\n")

        for line_no, line in enumerate(lines, 1):
            # Check for missing type annotations
            if re.search(r":\s*any\b", line):
                issues.append(
                    ValidationIssue(
                        severity=SeverityLevel.WARNING,
                        line=line_no,
                        column=line.find("any"),
                        message="Avoid using 'any' type",
                        code="ANY_TYPE",
                        suggestion="Use specific type instead of any",
                    )
                )

        return issues


class JSONValidator(SyntaxValidator):
    """Validates JSON code."""

    def validate(self, code: str) -> List[ValidationIssue]:
        """
        Validate JSON syntax.

        Args:
            code: JSON source code to validate

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            issues.append(
                ValidationIssue(
                    severity=SeverityLevel.ERROR,
                    line=e.lineno,
                    column=e.colno,
                    message=f"JSON error: {e.msg}",
                    code="JSON_ERROR",
                )
            )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=SeverityLevel.ERROR,
                    line=0,
                    column=0,
                    message=f"Parse error: {str(e)}",
                    code="PARSE_ERROR",
                )
            )

        return issues


class ShadowValidator:
    """Main validator that orchestrates syntax and lint validation."""

    def __init__(self):
        """Initialize validator with language-specific validators."""
        self.validators: Dict[Language, SyntaxValidator] = {
            Language.PYTHON: PythonValidator(),
            Language.JAVASCRIPT: JavaScriptValidator(),
            Language.TYPESCRIPT: TypeScriptValidator(),
            Language.JSON: JSONValidator(),
        }
        self.linter_runner = LinterRunner()

    async def validate(
        self,
        code: str,
        language: Union[Language, str],
        run_linter: bool = True,
    ) -> ValidationResult:
        """
        Validate code for syntax and style issues.

        Args:
            code: Source code to validate
            language: Programming language (Language enum or string)
            run_linter: Whether to run linter checks (default True)

        Returns:
            ValidationResult with all validation details
        """
        if isinstance(language, str):
            language = Language(language.lower())

        result = ValidationResult(valid=True, language=language)

        # Step 1: Syntax validation
        import time
        start_time = time.time()

        validator = self.validators.get(language)
        if validator:
            issues = validator.validate(code)
            result.issues.extend(issues)

            # Separate errors and warnings
            for issue in issues:
                if issue.severity == SeverityLevel.ERROR:
                    result.errors.append(issue)
                elif issue.severity == SeverityLevel.WARNING:
                    result.warnings.append(issue)

        result.parse_time_ms = (time.time() - start_time) * 1000

        # Step 2: Linter validation
        if run_linter and language in [Language.PYTHON, Language.JAVASCRIPT, Language.TYPESCRIPT]:
            start_time = time.time()
            lint_results = await self.linter_runner.run(code, language)
            result.lint_results = lint_results
            result.lint_time_ms = (time.time() - start_time) * 1000

            # Extract issues from lint results
            if "errors" in lint_results:
                for error in lint_results["errors"]:
                    result.errors.append(
                        ValidationIssue(
                            severity=SeverityLevel.ERROR,
                            line=error.get("line", 0),
                            column=error.get("column", 0),
                            message=error.get("message", ""),
                            code=error.get("code"),
                        )
                    )

            if "warnings" in lint_results:
                for warning in lint_results["warnings"]:
                    result.warnings.append(
                        ValidationIssue(
                            severity=SeverityLevel.WARNING,
                            line=warning.get("line", 0),
                            column=warning.get("column", 0),
                            message=warning.get("message", ""),
                            code=warning.get("code"),
                        )
                    )

        # Step 3: Determine overall validity
        result.valid = len(result.errors) == 0

        # Step 4: Generate suggestions
        result.suggestions = self._generate_suggestions(result)

        return result

    def _generate_suggestions(self, result: ValidationResult) -> List[str]:
        """Generate actionable suggestions based on validation results."""
        suggestions = []

        if result.errors:
            suggestions.append(f"Fix {len(result.errors)} error(s) before running code")

        if result.warnings:
            suggestions.append(f"Address {len(result.warnings)} warning(s) for better code quality")

        # Check for specific patterns
        for error in result.errors:
            if error.suggestion:
                suggestions.append(error.suggestion)

        for warning in result.warnings:
            if warning.suggestion:
                suggestions.append(warning.suggestion)

        return suggestions[:5]  # Limit to top 5 suggestions


class LinterRunner:
    """Runs external linters (ruff for Python, eslint for JS/TS)."""

    async def run(self, code: str, language: Language) -> Dict[str, Any]:
        """
        Run linter for the specified language.

        Args:
            code: Source code to lint
            language: Programming language

        Returns:
            Dictionary with linter results
        """
        if language == Language.PYTHON:
            return await self._run_ruff(code)
        elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            return await self._run_eslint(code, language)

        return {}

    async def _run_ruff(self, code: str) -> Dict[str, Any]:
        """Run ruff linter for Python."""
        try:
            # Check if ruff is available
            result = await asyncio.to_thread(
                subprocess.run,
                ["ruff", "--version"],
                capture_output=True,
                timeout=5,
            )

            if result.returncode != 0:
                return {"available": False, "errors": [], "warnings": []}

            # Create temporary file with code
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["ruff", "check", temp_file, "--output-format", "json"],
                    capture_output=True,
                    timeout=10,
                    text=True,
                )

                if result.stdout:
                    issues = json.loads(result.stdout)
                    return self._parse_ruff_output(issues)

                return {"available": True, "errors": [], "warnings": []}

            finally:
                Path(temp_file).unlink(missing_ok=True)

        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Ruff linter not available: {e}")
            return {"available": False, "errors": [], "warnings": []}

    async def _run_eslint(self, code: str, language: Language) -> Dict[str, Any]:
        """Run eslint linter for JavaScript/TypeScript."""
        try:
            # Check if eslint is available
            result = await asyncio.to_thread(
                subprocess.run,
                ["eslint", "--version"],
                capture_output=True,
                timeout=5,
            )

            if result.returncode != 0:
                return {"available": False, "errors": [], "warnings": []}

            # Create temporary file with code
            import tempfile
            ext = ".ts" if language == Language.TYPESCRIPT else ".js"
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["eslint", temp_file, "--format", "json"],
                    capture_output=True,
                    timeout=10,
                    text=True,
                )

                if result.stdout:
                    issues = json.loads(result.stdout)
                    return self._parse_eslint_output(issues)

                return {"available": True, "errors": [], "warnings": []}

            finally:
                Path(temp_file).unlink(missing_ok=True)

        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"ESLint linter not available: {e}")
            return {"available": False, "errors": [], "warnings": []}

    @staticmethod
    def _parse_ruff_output(issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Parse ruff JSON output."""
        errors = []
        warnings = []

        for issue in issues:
            item = {
                "line": issue.get("location", {}).get("row", 0),
                "column": issue.get("location", {}).get("column", 0),
                "message": issue.get("message", ""),
                "code": issue.get("code", ""),
            }

            if issue.get("kind") == "Error":
                errors.append(item)
            else:
                warnings.append(item)

        return {"available": True, "errors": errors, "warnings": warnings}

    @staticmethod
    def _parse_eslint_output(issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Parse eslint JSON output."""
        errors = []
        warnings = []

        for file_issues in issues:
            for issue in file_issues.get("messages", []):
                item = {
                    "line": issue.get("line", 0),
                    "column": issue.get("column", 0),
                    "message": issue.get("message", ""),
                    "code": issue.get("ruleId", ""),
                }

                if issue.get("severity") == 2:
                    errors.append(item)
                else:
                    warnings.append(item)

        return {"available": True, "errors": errors, "warnings": warnings}
