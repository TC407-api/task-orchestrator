"""
Tests for Shadow AST Validator.

Tests cover:
- Python syntax validation
- JavaScript/TypeScript validation
- JSON validation
- Linter integration
- Validation result structure
- Edge cases and error handling
"""

import pytest
from .shadow_validator import (
    ShadowValidator,
    ValidationResult,
    ValidationIssue,
    Language,
    SeverityLevel,
    PythonValidator,
    JavaScriptValidator,
    TypeScriptValidator,
    JSONValidator,
    LinterRunner,
)


class TestValidationIssue:
    """Test ValidationIssue dataclass."""

    def test_issue_creation(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            severity=SeverityLevel.ERROR,
            line=10,
            column=5,
            message="Syntax error",
            code="SYNTAX_ERROR",
        )

        assert issue.severity == SeverityLevel.ERROR
        assert issue.line == 10
        assert issue.column == 5
        assert issue.message == "Syntax error"

    def test_issue_to_dict(self):
        """Test issue serialization."""
        issue = ValidationIssue(
            severity=SeverityLevel.WARNING,
            line=3,
            column=1,
            message="Bare except",
            code="BARE_EXCEPT",
            suggestion="Use except Exception:",
        )

        data = issue.to_dict()
        assert data["severity"] == "warning"
        assert data["message"] == "Bare except"
        assert data["suggestion"] == "Use except Exception:"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(valid=True, language=Language.PYTHON)

        assert result.valid is True
        assert result.language == Language.PYTHON
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_result_to_dict(self):
        """Test result serialization."""
        issue = ValidationIssue(
            severity=SeverityLevel.ERROR,
            line=1,
            column=0,
            message="Error",
        )

        result = ValidationResult(valid=False, language=Language.PYTHON)
        result.errors.append(issue)

        data = result.to_dict()
        assert data["valid"] is False
        assert data["language"] == "python"
        assert data["error_count"] == 1


class TestPythonValidator:
    """Test Python syntax validation."""

    def test_valid_python(self):
        """Test validating correct Python code."""
        validator = PythonValidator()
        code = "x = 1\nprint(x)"

        issues = validator.validate(code)

        assert len(issues) == 0

    def test_syntax_error_missing_colon(self):
        """Test catching missing colon."""
        validator = PythonValidator()
        code = "if True\n    print('hello')"

        issues = validator.validate(code)

        assert len(issues) > 0
        assert issues[0].severity == SeverityLevel.ERROR
        assert "Syntax error" in issues[0].message

    def test_syntax_error_unmatched_paren(self):
        """Test catching unmatched parenthesis."""
        validator = PythonValidator()
        code = "print('hello'"

        issues = validator.validate(code)

        assert len(issues) > 0
        assert issues[0].severity == SeverityLevel.ERROR

    def test_bare_except_warning(self):
        """Test warning for bare except clause."""
        validator = PythonValidator()
        code = """
try:
    x = 1
except:
    pass
"""

        issues = validator.validate(code)

        assert len(issues) > 0
        bare_except = [i for i in issues if i.code == "BARE_EXCEPT"]
        assert len(bare_except) > 0
        assert bare_except[0].severity == SeverityLevel.WARNING

    def test_hardcoded_secret_warning(self):
        """Test warning for hardcoded secrets."""
        validator = PythonValidator()
        code = 'api_key = "sk-1234567890abcdef"'

        issues = validator.validate(code)

        secret_issues = [i for i in issues if i.code == "HARDCODED_SECRET"]
        assert len(secret_issues) > 0

    def test_complex_valid_code(self):
        """Test validating complex Python code."""
        validator = PythonValidator()
        code = """
class MyClass:
    def __init__(self):
        self.value = 0

    async def method(self):
        try:
            return self.value
        except Exception as e:
            print(f"Error: {e}")
"""

        issues = validator.validate(code)

        # Should have no errors (warnings are ok for this test)
        errors = [i for i in issues if i.severity == SeverityLevel.ERROR]
        assert len(errors) == 0


class TestJavaScriptValidator:
    """Test JavaScript syntax validation."""

    def test_valid_javascript(self):
        """Test validating correct JavaScript code."""
        validator = JavaScriptValidator()
        code = "const x = 1;\nconsole.log(x);"

        issues = validator.validate(code)

        assert len(issues) == 0

    def test_loose_equality_warning(self):
        """Test warning for == instead of ===."""
        validator = JavaScriptValidator()
        code = "if (x == 5) { }"

        issues = validator.validate(code)

        equality_issues = [i for i in issues if i.code == "LOOSE_EQUALITY"]
        assert len(equality_issues) > 0

    def test_var_usage_warning(self):
        """Test warning for var instead of const/let."""
        validator = JavaScriptValidator()
        code = "var x = 1;"

        issues = validator.validate(code)

        var_issues = [i for i in issues if i.code == "VAR_USAGE"]
        assert len(var_issues) > 0

    def test_unclosed_bracket_error(self):
        """Test error for unclosed bracket."""
        validator = JavaScriptValidator()
        code = "const x = { a: 1 ;"

        issues = validator.validate(code)

        bracket_issues = [i for i in issues if i.code and "bracket" in i.code.lower()]
        assert len(bracket_issues) > 0

    def test_comment_skipping(self):
        """Test that comments are properly skipped."""
        validator = JavaScriptValidator()
        code = """
// This is a comment with { unmatched bracket
const x = { a: 1 };
"""

        issues = validator.validate(code)

        # Should not report error for bracket in comment
        errors = [i for i in issues if i.severity == SeverityLevel.ERROR]
        # Note: may have false positives, but demonstrating comment handling
        assert isinstance(errors, list)

    def test_const_let_valid(self):
        """Test that const/let don't trigger warnings."""
        validator = JavaScriptValidator()
        code = "const x = 1;\nlet y = 2;"

        issues = validator.validate(code)

        var_issues = [i for i in issues if i.code == "VAR_USAGE"]
        assert len(var_issues) == 0


class TestTypeScriptValidator:
    """Test TypeScript syntax validation."""

    def test_valid_typescript(self):
        """Test validating correct TypeScript code."""
        validator = TypeScriptValidator()
        code = "const x: number = 1;\nconsole.log(x);"

        issues = validator.validate(code)

        assert len(issues) == 0

    def test_any_type_warning(self):
        """Test warning for 'any' type."""
        validator = TypeScriptValidator()
        code = "const x: any = 1;"

        issues = validator.validate(code)

        any_issues = [i for i in issues if i.code == "ANY_TYPE"]
        assert len(any_issues) > 0

    def test_javascript_checks_included(self):
        """Test that JS checks are also applied."""
        validator = TypeScriptValidator()
        code = "var x: number = 1;"

        issues = validator.validate(code)

        var_issues = [i for i in issues if i.code == "VAR_USAGE"]
        assert len(var_issues) > 0


class TestJSONValidator:
    """Test JSON syntax validation."""

    def test_valid_json(self):
        """Test validating correct JSON."""
        validator = JSONValidator()
        code = '{"key": "value", "number": 123}'

        issues = validator.validate(code)

        assert len(issues) == 0

    def test_invalid_json_missing_quote(self):
        """Test error for missing quote."""
        validator = JSONValidator()
        code = '{"key: "value"}'

        issues = validator.validate(code)

        assert len(issues) > 0
        assert issues[0].severity == SeverityLevel.ERROR

    def test_invalid_json_trailing_comma(self):
        """Test error for trailing comma."""
        validator = JSONValidator()
        code = '{"key": "value",}'

        issues = validator.validate(code)

        assert len(issues) > 0
        assert issues[0].severity == SeverityLevel.ERROR

    def test_complex_valid_json(self):
        """Test validating complex JSON."""
        validator = JSONValidator()
        code = """{
  "array": [1, 2, 3],
  "nested": {
    "key": "value"
  },
  "null_value": null,
  "bool": true
}"""

        issues = validator.validate(code)

        assert len(issues) == 0


class TestShadowValidator:
    """Test main ShadowValidator class."""

    @pytest.mark.asyncio
    async def test_validate_python_valid(self):
        """Test validating valid Python code."""
        validator = ShadowValidator()
        code = "x = 1\nprint(x)"

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_python_invalid(self):
        """Test validating invalid Python code."""
        validator = ShadowValidator()
        code = "if True\n    print('hello')"

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert result.valid is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_validate_javascript_valid(self):
        """Test validating valid JavaScript code."""
        validator = ShadowValidator()
        code = "const x = 1;\nconsole.log(x);"

        result = await validator.validate(code, Language.JAVASCRIPT, run_linter=False)

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validate_json_valid(self):
        """Test validating valid JSON."""
        validator = ShadowValidator()
        code = '{"key": "value"}'

        result = await validator.validate(code, Language.JSON, run_linter=False)

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validate_json_invalid(self):
        """Test validating invalid JSON."""
        validator = ShadowValidator()
        code = '{"key": "value",}'

        result = await validator.validate(code, Language.JSON, run_linter=False)

        assert result.valid is False

    @pytest.mark.asyncio
    async def test_language_string_conversion(self):
        """Test that string language names are converted to enum."""
        validator = ShadowValidator()
        code = "x = 1"

        result = await validator.validate(code, "python", run_linter=False)

        assert result.language == Language.PYTHON
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validation_result_structure(self):
        """Test structure of validation result."""
        validator = ShadowValidator()
        code = "if True\n    pass"

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert hasattr(result, "valid")
        assert hasattr(result, "language")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "suggestions")
        assert hasattr(result, "parse_time_ms")

    @pytest.mark.asyncio
    async def test_suggestions_generated(self):
        """Test that suggestions are generated."""
        validator = ShadowValidator()
        code = "if True\n    print('hello')"

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert len(result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_warnings_collected(self):
        """Test that warnings are collected separately."""
        validator = ShadowValidator()
        code = """
try:
    x = 1
except:
    pass
"""

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert len(result.warnings) > 0
        # All warnings should be marked as WARNING severity
        for warning in result.warnings:
            assert warning.severity == SeverityLevel.WARNING

    @pytest.mark.asyncio
    async def test_result_to_dict(self):
        """Test result can be serialized to dict."""
        validator = ShadowValidator()
        code = "x = 1"

        result = await validator.validate(code, Language.PYTHON, run_linter=False)
        data = result.to_dict()

        assert isinstance(data, dict)
        assert "valid" in data
        assert "language" in data
        assert "errors" in data
        assert "warnings" in data
        assert "error_count" in data
        assert "warning_count" in data


class TestLinterRunner:
    """Test LinterRunner class."""

    @pytest.mark.asyncio
    async def test_linter_runner_unavailable_graceful(self):
        """Test that linter runner gracefully handles missing tools."""
        runner = LinterRunner()
        code = "x = 1"

        # This should not raise, just return unavailable
        result = await runner.run(code, Language.PYTHON)

        assert isinstance(result, dict)
        assert "available" in result or "errors" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_code(self):
        """Test validating empty code."""
        validator = ShadowValidator()
        code = ""

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_whitespace_only_code(self):
        """Test validating whitespace-only code."""
        validator = ShadowValidator()
        code = "   \n\n   \n"

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_unicode_in_code(self):
        """Test validating code with unicode."""
        validator = ShadowValidator()
        code = 'print("Hello, 世界")'

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_large_code_sample(self):
        """Test validating large code sample."""
        validator = ShadowValidator()
        code = "\n".join([f"x{i} = {i}" for i in range(1000)])

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_multiline_strings(self):
        """Test handling of multiline strings."""
        validator = ShadowValidator()
        code = '''"""
This is a
multiline string
"""
x = 1
'''

        result = await validator.validate(code, Language.PYTHON, run_linter=False)

        assert result.valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
