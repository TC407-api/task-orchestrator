"""
Concrete implementations of common code and text validators.

Includes graders for JSON validation, regex matching, string containment,
and length checks.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union, Pattern

try:
    import jsonschema
    from jsonschema import ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    jsonschema = None  # type: ignore
    ValidationError = Exception  # type: ignore
    JSONSCHEMA_AVAILABLE = False

from ..trial import GraderResult
from .base import Grader


class NonEmptyGrader(Grader):
    """Checks if the output is not None and not an empty string."""

    def __init__(self, name: str = "NonEmptyCheck", weight: float = 1.0):
        super().__init__(name, weight)

    async def grade(self, output: Any, context: Dict[str, Any]) -> GraderResult:
        if output is None:
            return self._make_result(False, 0.0, "Output is None")

        if isinstance(output, str) and not output.strip():
            return self._make_result(False, 0.0, "Output is an empty string")

        if hasattr(output, "__len__") and len(output) == 0:
            return self._make_result(False, 0.0, "Output has zero length")

        return self._make_result(True, 1.0, "Output is not empty")


class JSONValidGrader(Grader):
    """Checks if the output string is valid JSON."""

    def __init__(self, name: str = "JSONValidCheck", weight: float = 1.0):
        super().__init__(name, weight)

    async def grade(self, output: Any, context: Dict[str, Any]) -> GraderResult:
        # Already a Python object
        if isinstance(output, (dict, list)):
            return self._make_result(True, 1.0, "Output is already a Python object")

        if not isinstance(output, (str, bytes)):
            return self._make_result(
                False, 0.0, f"Expected string or bytes, got {type(output).__name__}"
            )

        # Try to extract JSON from markdown code blocks
        text = output if isinstance(output, str) else output.decode()

        # Handle ```json ... ``` blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            text = json_match.group(1).strip()

        try:
            json.loads(text)
            return self._make_result(True, 1.0, "Valid JSON")
        except json.JSONDecodeError as e:
            return self._make_result(
                False, 0.0, f"Invalid JSON: {e.msg}", position=e.pos, line=e.lineno
            )


class JSONSchemaGrader(Grader):
    """
    Validates the output against a specific JSON schema.

    Requires 'jsonschema' library to be installed.
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        name: str = "JSONSchemaCheck",
        weight: float = 1.0
    ):
        super().__init__(name, weight)
        self.schema = schema
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError("The 'jsonschema' library is required for JSONSchemaGrader.")

    async def grade(self, output: Any, context: Dict[str, Any]) -> GraderResult:
        data_to_validate = output

        # If output is a string, try to parse it first
        if isinstance(output, (str, bytes)):
            text = output if isinstance(output, str) else output.decode()

            # Handle markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if json_match:
                text = json_match.group(1).strip()

            try:
                data_to_validate = json.loads(text)
            except json.JSONDecodeError:
                return self._make_result(
                    False, 0.0, "Output is not valid JSON, cannot validate schema"
                )

        try:
            jsonschema.validate(instance=data_to_validate, schema=self.schema)
            return self._make_result(True, 1.0, "Schema validation passed")
        except ValidationError as e:
            error_msg = getattr(e, 'message', str(e))
            error_path = list(getattr(e, 'path', [])) if hasattr(e, 'path') else []
            return self._make_result(
                False,
                0.0,
                f"Schema validation failed: {error_msg}",
                path=error_path
            )


class RegexGrader(Grader):
    """Checks if the output matches a specific regex pattern."""

    def __init__(
        self,
        pattern: Union[str, Pattern],
        name: str = "RegexCheck",
        weight: float = 1.0
    ):
        super().__init__(name, weight)
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern

    async def grade(self, output: Any, context: Dict[str, Any]) -> GraderResult:
        if not isinstance(output, str):
            output = str(output)

        match = self.pattern.search(output)
        if match:
            return self._make_result(True, 1.0, "Pattern found", match=match.group(0))

        return self._make_result(False, 0.0, "Pattern not found")


class LengthGrader(Grader):
    """Checks if the output length is within specified bounds."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        name: str = "LengthCheck",
        weight: float = 1.0
    ):
        super().__init__(name, weight)
        self.min_length = min_length
        self.max_length = max_length

    async def grade(self, output: Any, context: Dict[str, Any]) -> GraderResult:
        if output is None:
            return self._make_result(False, 0.0, "Output is None", length=0)

        if not hasattr(output, "__len__"):
            output = str(output)

        length = len(output)

        if self.min_length is not None and length < self.min_length:
            return self._make_result(
                False, 0.0, f"Too short: {length} < {self.min_length}", length=length
            )

        if self.max_length is not None and length > self.max_length:
            return self._make_result(
                False, 0.0, f"Too long: {length} > {self.max_length}", length=length
            )

        return self._make_result(True, 1.0, "Length within bounds", length=length)


class ContainsGrader(Grader):
    """
    Checks if output contains specific strings.

    Calculates a partial score based on the percentage of required strings found.
    """

    def __init__(
        self,
        required_strings: List[str],
        case_sensitive: bool = False,
        name: str = "ContainsCheck",
        weight: float = 1.0
    ):
        super().__init__(name, weight)
        self.required_strings = required_strings
        self.case_sensitive = case_sensitive

    async def grade(self, output: Any, context: Dict[str, Any]) -> GraderResult:
        if not isinstance(output, str):
            output = str(output)

        text = output if self.case_sensitive else output.lower()
        targets = self.required_strings if self.case_sensitive else [s.lower() for s in self.required_strings]

        found = []
        missing = []

        for original, target in zip(self.required_strings, targets):
            if target in text:
                found.append(original)
            else:
                missing.append(original)

        total = len(self.required_strings)
        if total == 0:
            return self._make_result(True, 1.0, "No strings required")

        score = len(found) / total
        passed = score == 1.0

        return self._make_result(
            passed,
            score,
            f"Found {len(found)}/{total} strings",
            found=found,
            missing=missing
        )


class NotContainsGrader(Grader):
    """
    Checks if output DOES NOT contain specific forbidden strings.

    This is a strict check: if ANY forbidden string is found, score is 0.0.
    """

    def __init__(
        self,
        forbidden_strings: List[str],
        case_sensitive: bool = False,
        name: str = "NotContainsCheck",
        weight: float = 1.0
    ):
        super().__init__(name, weight)
        self.forbidden_strings = forbidden_strings
        self.case_sensitive = case_sensitive

    async def grade(self, output: Any, context: Dict[str, Any]) -> GraderResult:
        if not isinstance(output, str):
            output = str(output)

        text = output if self.case_sensitive else output.lower()
        targets = self.forbidden_strings if self.case_sensitive else [s.lower() for s in self.forbidden_strings]

        found_forbidden = []

        for original, target in zip(self.forbidden_strings, targets):
            if target in text:
                found_forbidden.append(original)

        if found_forbidden:
            return self._make_result(
                False,
                0.0,
                f"Found forbidden strings: {', '.join(found_forbidden)}",
                found=found_forbidden
            )

        return self._make_result(True, 1.0, "No forbidden strings found")
