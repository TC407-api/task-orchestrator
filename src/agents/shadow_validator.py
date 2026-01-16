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

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

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
        self.shadow_comparator = ShadowComparator()

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

    # Shadow Comparison Methods (delegate to ShadowComparator)

    def set_shadow(self, original_code: str) -> None:
        """
        Store shadow code for comparison.

        Args:
            original_code: Original source code to use as shadow
        """
        self.shadow_comparator.set_shadow(original_code)

    def parse(self, source_code: str) -> ASTNode:
        """
        Parse source code to ASTNode tree.

        Args:
            source_code: Python source code to parse

        Returns:
            ASTNode representing the module root
        """
        return self.shadow_comparator.parse(source_code)

    def validate_shadow(self, modified_code: str) -> ShadowValidationResult:
        """
        Compare modified code against shadow and detect mutations.

        Args:
            modified_code: Modified source code to validate

        Returns:
            ShadowValidationResult with violations, mutations, and warnings
        """
        return self.shadow_comparator.validate(modified_code)

    def extract_signature(self, func_name: str, source: str) -> Optional[Dict[str, Any]]:
        """
        Extract function signature from source code.

        Args:
            func_name: Name of function to extract
            source: Source code containing the function

        Returns:
            Dictionary with function signature details or None if not found
        """
        return self.shadow_comparator.extract_signature(func_name, source)

    def extract_call_chain(self, source: str) -> List[str]:
        """
        Extract method call chains from source code.

        Args:
            source: Source code to analyze

        Returns:
            List of method call chains (e.g., ["obj.method1().method2()"])
        """
        return self.shadow_comparator.extract_call_chain(source)

    def extract_returns(self, func_name: str, source: str) -> List[str]:
        """
        Extract return statements from a function.

        Args:
            func_name: Name of function to analyze
            source: Source code containing the function

        Returns:
            List of return statement strings
        """
        return self.shadow_comparator.extract_returns(func_name, source)

    def detect_security_violations(self, code: str) -> List[str]:
        """
        Detect dangerous patterns in code.

        Args:
            code: Source code to analyze

        Returns:
            List of security violation descriptions
        """
        return self.shadow_comparator.detect_security_violations(code)

    def generate_report(self) -> str:
        """
        Generate markdown report of shadow validation.

        Returns:
            Markdown formatted report string
        """
        return self.shadow_comparator.generate_report()


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


# Shadow Comparison Features


class ASTNodeType(Enum):
    """Types of AST nodes."""
    MODULE = auto()
    FUNCTION = auto()
    CLASS = auto()
    IMPORT = auto()
    CALL = auto()
    RETURN = auto()


@dataclass
class ASTNode:
    """Represents a node in the abstract syntax tree."""
    type: ASTNodeType
    name: str
    line_number: int
    source: str
    children: List["ASTNode"] = field(default_factory=list)


@dataclass
class ShadowValidationResult:
    """Result of shadow validation comparing modified code against original."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ShadowComparator:
    """Compares modified code against shadow (original) code to detect mutations."""

    def __init__(self):
        """Initialize shadow comparator."""
        self.shadow_code: Optional[str] = None
        self.shadow_ast: Optional[ASTNode] = None

    def set_shadow(self, original_code: str) -> None:
        """
        Store shadow code for comparison.

        Args:
            original_code: Original source code to use as shadow
        """
        self.shadow_code = original_code
        self.shadow_ast = self.parse(original_code)

    def parse(self, source_code: str) -> ASTNode:
        """
        Parse source code to ASTNode tree.

        Args:
            source_code: Python source code to parse

        Returns:
            ASTNode representing the module root
        """
        try:
            tree = ast.parse(source_code)
            return self._build_ast_node(tree, source_code)
        except SyntaxError as e:
            logger.error(f"Failed to parse source code: {e}")
            return ASTNode(
                type=ASTNodeType.MODULE,
                name="<error>",
                line_number=0,
                source=source_code,
                children=[]
            )

    def _build_ast_node(self, node: ast.AST, source: str) -> ASTNode:
        """Build ASTNode tree from ast.AST node."""
        children = []

        if isinstance(node, ast.Module):
            ast_node = ASTNode(
                type=ASTNodeType.MODULE,
                name="<module>",
                line_number=0,
                source=source,
                children=[]
            )
            for child in node.body:
                children.append(self._build_ast_node(child, source))

        elif isinstance(node, ast.FunctionDef):
            lines = source.split("\n")
            func_source = "\n".join(lines[node.lineno - 1:node.end_lineno if node.end_lineno else node.lineno])
            ast_node = ASTNode(
                type=ASTNodeType.FUNCTION,
                name=node.name,
                line_number=node.lineno,
                source=func_source,
                children=[]
            )
            for child in node.body:
                children.append(self._build_ast_node(child, source))

        elif isinstance(node, ast.ClassDef):
            lines = source.split("\n")
            class_source = "\n".join(lines[node.lineno - 1:node.end_lineno if node.end_lineno else node.lineno])
            ast_node = ASTNode(
                type=ASTNodeType.CLASS,
                name=node.name,
                line_number=node.lineno,
                source=class_source,
                children=[]
            )
            for child in node.body:
                children.append(self._build_ast_node(child, source))

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            ast_node = ASTNode(
                type=ASTNodeType.IMPORT,
                name=ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                line_number=getattr(node, 'lineno', 0),
                source=ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                children=[]
            )

        elif isinstance(node, ast.Return):
            ast_node = ASTNode(
                type=ASTNodeType.RETURN,
                name="<return>",
                line_number=getattr(node, 'lineno', 0),
                source=ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                children=[]
            )

        elif isinstance(node, ast.Call):
            ast_node = ASTNode(
                type=ASTNodeType.CALL,
                name=ast.unparse(node.func) if hasattr(ast, 'unparse') else str(node.func),
                line_number=getattr(node, 'lineno', 0),
                source=ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                children=[]
            )

        else:
            # Generic node
            ast_node = ASTNode(
                type=ASTNodeType.MODULE,
                name=node.__class__.__name__,
                line_number=getattr(node, 'lineno', 0),
                source=ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                children=[]
            )

        ast_node.children = children
        return ast_node

    def validate(self, modified_code: str) -> ShadowValidationResult:
        """
        Compare modified code against shadow and detect mutations.

        Args:
            modified_code: Modified source code to validate

        Returns:
            ShadowValidationResult with violations, mutations, and warnings
        """
        result = ShadowValidationResult(is_valid=True)

        if not self.shadow_code:
            result.warnings.append("No shadow code set for comparison")
            return result

        # Parse modified code
        modified_ast = self.parse(modified_code)

        # Detect security violations
        security_violations = self.detect_security_violations(modified_code)
        result.violations.extend(security_violations)

        # Compare ASTs
        mutations = self._compare_asts(self.shadow_ast, modified_ast)
        result.mutations.extend(mutations)

        # Check for significant changes - mutations also make code invalid
        if security_violations or mutations:
            result.is_valid = False

        return result

    def _compare_asts(self, shadow: Optional[ASTNode], modified: ASTNode) -> List[str]:
        """Compare two AST trees and return list of mutations."""
        mutations = []

        if not shadow:
            return mutations

        # Compare imports
        shadow_imports = self._get_imports(shadow)
        modified_imports = self._get_imports(modified)

        for import_name in modified_imports:
            if import_name not in shadow_imports:
                mutations.append(f"Import '{import_name}' was added")

        for import_name in shadow_imports:
            if import_name not in modified_imports:
                mutations.append(f"Import '{import_name}' was removed")

        # Compare functions
        shadow_funcs = self._get_functions(shadow)
        modified_funcs = self._get_functions(modified)

        # Check for removed functions
        for func_name in shadow_funcs:
            if func_name not in modified_funcs:
                mutations.append(f"Function '{func_name}' was removed")

        # Check for added functions
        for func_name in modified_funcs:
            if func_name not in shadow_funcs:
                mutations.append(f"Function '{func_name}' was added")

        # Check for modified functions
        for func_name in shadow_funcs:
            if func_name in modified_funcs:
                shadow_func = shadow_funcs[func_name]
                modified_func = modified_funcs[func_name]

                # Compare function bodies
                if shadow_func.source != modified_func.source:
                    mutations.append(f"Function '{func_name}' was modified")

        return mutations

    def _get_functions(self, node: ASTNode) -> Dict[str, ASTNode]:
        """Extract all functions from AST node."""
        functions = {}

        if node.type == ASTNodeType.FUNCTION:
            functions[node.name] = node

        for child in node.children:
            functions.update(self._get_functions(child))

        return functions

    def _get_imports(self, node: ASTNode) -> Set[str]:
        """Extract all imports from AST node."""
        imports = set()

        if node.type == ASTNodeType.IMPORT:
            imports.add(node.name)

        for child in node.children:
            imports.update(self._get_imports(child))

        return imports

    def extract_signature(self, func_name: str, source: str) -> Optional[Dict[str, Any]]:
        """
        Extract function signature from source code.

        Args:
            func_name: Name of function to extract
            source: Source code containing the function

        Returns:
            Dictionary with function signature details or None if not found
        """
        try:
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Extract parameters
                    params = []
                    for arg in node.args.args:
                        param_info = {"name": arg.arg}
                        if arg.annotation:
                            param_info["type"] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                        params.append(param_info)

                    # Extract return type
                    return_type = None
                    if node.returns:
                        return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

                    return {
                        "name": func_name,
                        "params": params,
                        "return_type": return_type,
                        "line": node.lineno,
                    }

        except SyntaxError as e:
            logger.error(f"Failed to parse source for signature extraction: {e}")

        return None

    def extract_call_chain(self, source: str) -> List[str]:
        """
        Extract method call chains from source code.

        Args:
            source: Source code to analyze

        Returns:
            List of method call chains (e.g., ["obj.method1().method2()"])
        """
        call_chains = []

        try:
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    chain = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                    # Check if it's a chained call (contains multiple dots)
                    if chain.count('.') > 1:
                        call_chains.append(chain)

        except SyntaxError as e:
            logger.error(f"Failed to parse source for call chain extraction: {e}")

        return call_chains

    def extract_returns(self, func_name: str, source: str) -> List[str]:
        """
        Extract return statements from a function.

        Args:
            func_name: Name of function to analyze
            source: Source code containing the function

        Returns:
            List of return statement strings
        """
        returns = []

        try:
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Walk through function body
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return):
                            return_str = ast.unparse(child) if hasattr(ast, 'unparse') else str(child)
                            returns.append(return_str)

        except SyntaxError as e:
            logger.error(f"Failed to parse source for return extraction: {e}")

        return returns

    def detect_security_violations(self, code: str) -> List[str]:
        """
        Detect dangerous patterns in code.

        Args:
            code: Source code to analyze

        Returns:
            List of security violation descriptions
        """
        violations = []

        # Pattern matching for dangerous functions
        dangerous_patterns = [
            (r'\bos\.system\s*\(', "os.system() detected - potential command injection risk"),
            (r'\beval\s*\(', "eval() detected - code execution risk"),
            (r'\bexec\s*\(', "exec() detected - code execution risk"),
            (r'\b__import__\s*\(', "__import__() detected - dynamic import risk"),
            (r'\bpickle\.loads\s*\(', "pickle.loads() detected - deserialization risk"),
            (r'\bsubprocess\.call\s*\([^,]*shell\s*=\s*True', "subprocess with shell=True - command injection risk"),
            (r'\bopen\s*\([^,]*["\']w["\']', "file write operation detected - verify file path safety"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                violations.append(message)

        # Check for SQL injection patterns
        if re.search(r'execute\s*\([^)]*\+', code):
            violations.append("String concatenation in SQL execute() - SQL injection risk")

        # Check for hardcoded secrets
        if re.search(r'(?i)(password|api_?key|secret|token)\s*=\s*["\'][^"\']+["\']', code):
            violations.append("Hardcoded secret detected - use environment variables")

        return violations

    def generate_report(self) -> str:
        """
        Generate markdown report of shadow validation.

        Returns:
            Markdown formatted report string
        """
        if not self.shadow_code:
            return "# Shadow Validation Report\n\nNo shadow code set.\n"

        report = ["# Shadow Validation Report\n"]

        # Summary
        report.append("## Summary\n")
        report.append(f"- Shadow code lines: {len(self.shadow_code.splitlines())}\n")

        # Functions
        shadow_funcs = self._get_functions(self.shadow_ast) if self.shadow_ast else {}
        report.append(f"- Functions in shadow: {len(shadow_funcs)}\n")

        if shadow_funcs:
            report.append("\n## Functions\n")
            for func_name, func_node in shadow_funcs.items():
                report.append(f"- `{func_name}` (line {func_node.line_number})\n")

        # Security check
        violations = self.detect_security_violations(self.shadow_code)
        if violations:
            report.append("\n## Security Violations\n")
            for violation in violations:
                report.append(f"- {violation}\n")

        return "".join(report)
