"""Global @Workflow trigger system for context injection macros.

Provides context injection capabilities through @Symbol patterns:
- @Refactor: Load related files, inject refactoring guidelines
- @TestGen: Load source + existing tests, inject test patterns
- @Debug: Load error context, inject debugging methodology
- @Review: Load diff, inject review checklist
- @Docs: Load source, inject documentation templates

Each workflow triggers context injection for the appropriate archetype.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import glob as glob_module


class WorkflowType(Enum):
    """Available workflow triggers."""
    REFACTOR = "refactor"
    TEST_GEN = "testgen"
    DEBUG = "debug"
    REVIEW = "review"
    DOCS = "docs"


@dataclass
class ContextManifest:
    """Defines files and prompts to inject for a workflow."""

    trigger: str
    """The @Symbol that triggers this workflow (e.g., '@Refactor')"""

    files_pattern: List[str] = field(default_factory=list)
    """Glob patterns to load context files"""

    system_prompt_addition: str = ""
    """System prompt addition to inject into LLM calls"""

    archetype: str = "builder"
    """Which archetype should handle this workflow"""

    max_context_tokens: int = 4000
    """Maximum tokens to use for injected context"""

    temperature: Optional[float] = None
    """Optional temperature override for this workflow"""

    priority: int = 0
    """Priority level for workflow processing"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to dictionary."""
        return {
            "trigger": self.trigger,
            "files_pattern": self.files_pattern,
            "system_prompt_addition": self.system_prompt_addition,
            "archetype": self.archetype,
            "max_context_tokens": self.max_context_tokens,
            "temperature": self.temperature,
            "priority": self.priority,
        }


class WorkflowRegistry:
    """Stores and manages workflow definitions."""

    def __init__(self):
        """Initialize with built-in workflows."""
        self._workflows: Dict[str, ContextManifest] = {}
        self._register_builtin_workflows()

    def _register_builtin_workflows(self) -> None:
        """Register all built-in workflows."""
        # @Refactor workflow
        self.register(
            ContextManifest(
                trigger="@Refactor",
                files_pattern=[
                    "src/**/*.py",
                    "src/**/*.ts",
                    "src/**/*.tsx",
                    "src/**/*.js",
                    "src/**/*.jsx",
                ],
                system_prompt_addition="""You are in REFACTOR mode. Your task is to improve code structure and quality.

Guidelines:
- Extract duplicated logic into reusable functions
- Improve naming for clarity
- Reduce cyclomatic complexity
- Extract large functions into smaller, focused ones
- Apply SOLID principles
- Document complex logic with clear comments
- Ensure the refactored code maintains original functionality
- Run tests after refactoring to verify behavior

Quality Standards:
- Function max length: 30 lines
- Function max parameters: 4 (use options object for more)
- Nesting max depth: 3 levels
- Test coverage: maintain or improve existing coverage""",
                archetype="builder",
                max_context_tokens=4000,
                temperature=0.3,
                priority=5,
            )
        )

        # @TestGen workflow
        self.register(
            ContextManifest(
                trigger="@TestGen",
                files_pattern=[
                    "src/**/*.py",
                    "src/**/*.ts",
                    "src/**/*.tsx",
                    "tests/**/*.py",
                    "tests/**/*.ts",
                    "tests/**/*.tsx",
                    "**/test_*.py",
                    "**/*.test.ts",
                    "**/*.test.tsx",
                ],
                system_prompt_addition="""You are in TEST_GEN mode. Your task is to generate comprehensive tests.

Guidelines:
- Analyze the source code to understand functionality
- Review existing tests for patterns and coverage gaps
- Generate unit tests for all public functions
- Generate integration tests for critical paths
- Include edge cases and error conditions
- Test validation and error handling
- Aim for 80%+ code coverage for new code
- Follow existing test patterns and naming conventions

Test Requirements:
- Test should be independent and isolated
- Use clear, descriptive test names
- Include setup/teardown as needed
- Mock external dependencies appropriately
- Test both happy path and error cases
- Include tests for boundary conditions""",
                archetype="builder",
                max_context_tokens=6000,
                temperature=0.5,
                priority=5,
            )
        )

        # @Debug workflow
        self.register(
            ContextManifest(
                trigger="@Debug",
                files_pattern=[
                    "src/**/*.py",
                    "src/**/*.ts",
                    "src/**/*.tsx",
                    "logs/**/*.log",
                    "*.log",
                ],
                system_prompt_addition="""You are in DEBUG mode. Your task is to diagnose and fix issues.

Debugging Methodology:
1. REPRODUCE: First, understand how to reproduce the issue
2. ISOLATE: Identify the specific code path causing the problem
3. ROOT CAUSE: Determine the underlying cause (not just symptoms)
4. FIX: Implement a targeted fix
5. VERIFY: Test the fix and ensure no regressions

Guidelines:
- Start with the most recent error messages and stack traces
- Add strategic logging to understand execution flow
- Check for common issues: null references, type mismatches, edge cases
- Review recent changes that might have introduced the bug
- Consider integration points and dependencies
- Verify assumptions about data and state
- Document the root cause and fix rationale
- Add tests to prevent regression""",
                archetype="builder",
                max_context_tokens=5000,
                temperature=0.2,
                priority=10,
            )
        )

        # @Review workflow
        self.register(
            ContextManifest(
                trigger="@Review",
                files_pattern=[
                    "src/**/*.py",
                    "src/**/*.ts",
                    "src/**/*.tsx",
                ],
                system_prompt_addition="""You are in REVIEW mode. Your task is to perform code review.

Code Review Checklist:
- Correctness: Does the code work as intended?
- Clarity: Is the code easy to understand?
- Consistency: Does it follow project conventions?
- Test Coverage: Are critical paths tested?
- Performance: Are there obvious performance issues?
- Security: Any security vulnerabilities?
- Dependencies: Unnecessary or problematic dependencies?
- Documentation: Is the code documented?

Review Guidelines:
- Provide constructive, actionable feedback
- Explain the reasoning behind each comment
- Suggest improvements with examples when helpful
- Consider architectural implications
- Check for compliance with quality standards
- Verify error handling is appropriate
- Ensure logging/observability is adequate
- Identify potential race conditions or threading issues""",
                archetype="architect",
                max_context_tokens=6000,
                temperature=0.1,
                priority=5,
            )
        )

        # @Docs workflow
        self.register(
            ContextManifest(
                trigger="@Docs",
                files_pattern=[
                    "src/**/*.py",
                    "src/**/*.ts",
                    "src/**/*.tsx",
                    "README.md",
                    "docs/**/*.md",
                ],
                system_prompt_addition="""You are in DOCS mode. Your task is to generate or improve documentation.

Documentation Tasks:
- Generate docstrings for functions, classes, and modules
- Create or improve README documentation
- Document public APIs and interfaces
- Add examples where helpful
- Create architecture documentation if needed
- Document configuration options
- Create troubleshooting guides

Documentation Standards:
- Use clear, concise language
- Include examples for complex features
- Document parameters, return values, and exceptions
- Link related documentation
- Keep examples up to date
- Document non-obvious design decisions
- Include usage patterns and best practices
- Add type annotations in docstrings""",
                archetype="researcher",
                max_context_tokens=5000,
                temperature=0.4,
                priority=3,
            )
        )

    def register(self, manifest: ContextManifest) -> None:
        """Register a workflow manifest.

        Args:
            manifest: ContextManifest defining the workflow
        """
        self._workflows[manifest.trigger] = manifest

    def get(self, trigger: str) -> Optional[ContextManifest]:
        """Get workflow by trigger name.

        Args:
            trigger: Trigger string (e.g., '@Refactor')

        Returns:
            ContextManifest if found, None otherwise
        """
        return self._workflows.get(trigger)

    def list_workflows(self) -> List[ContextManifest]:
        """List all registered workflows.

        Returns:
            List of ContextManifest objects
        """
        return list(self._workflows.values())

    def get_all_triggers(self) -> Set[str]:
        """Get all available trigger strings.

        Returns:
            Set of trigger strings
        """
        return set(self._workflows.keys())

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize all workflows to dictionary.

        Returns:
            Dict mapping trigger to workflow definition
        """
        return {
            trigger: manifest.to_dict()
            for trigger, manifest in self._workflows.items()
        }


class WorkflowTrigger:
    """Detects @Symbol patterns in prompts and processes workflows."""

    def __init__(self, registry: Optional[WorkflowRegistry] = None):
        """Initialize with workflow registry.

        Args:
            registry: WorkflowRegistry instance. If None, creates default.
        """
        self.registry = registry or WorkflowRegistry()
        self._trigger_pattern = re.compile(
            r"@([A-Za-z]+)", re.MULTILINE
        )

    def detect_triggers(self, prompt: str) -> Set[str]:
        """Detect all workflow triggers in a prompt.

        Args:
            prompt: User prompt text

        Returns:
            Set of detected trigger strings (e.g., {'@Refactor', '@TestGen'})
        """
        matches = self._trigger_pattern.findall(prompt)
        return {f"@{match}" for match in matches}

    def has_trigger(self, prompt: str) -> bool:
        """Check if prompt contains any workflow trigger.

        Args:
            prompt: User prompt text

        Returns:
            True if at least one trigger is present
        """
        return bool(self._trigger_pattern.search(prompt))

    def extract_trigger_context(
        self,
        trigger: str,
        base_path: str = ".",
        max_files: int = 20,
    ) -> Dict[str, Any]:
        """Extract context files for a specific trigger.

        Args:
            trigger: Trigger string (e.g., '@Refactor')
            base_path: Base directory for glob patterns
            max_files: Maximum files to load

        Returns:
            Dict with 'files' (List[Dict]) and 'manifest' (ContextManifest)
        """
        manifest = self.registry.get(trigger)
        if not manifest:
            return {"files": [], "manifest": None, "error": f"Unknown trigger: {trigger}"}

        files = []
        base_path_obj = Path(base_path)

        for pattern in manifest.files_pattern:
            # Convert glob pattern to absolute path
            full_pattern = str(base_path_obj / pattern)

            # Handle both Unix and Windows paths
            for file_path in glob_module.glob(full_pattern, recursive=True):
                if len(files) >= max_files:
                    break

                try:
                    path_obj = Path(file_path)
                    if path_obj.is_file() and not self._should_skip_file(path_obj):
                        files.append(
                            {
                                "path": str(path_obj),
                                "relative_path": str(
                                    path_obj.relative_to(base_path_obj)
                                ),
                                "size_bytes": path_obj.stat().st_size,
                            }
                        )
                except (OSError, ValueError):
                    # Skip files that can't be accessed
                    continue

            if len(files) >= max_files:
                break

        return {
            "files": files,
            "manifest": manifest,
            "total_files": len(files),
        }

    def _should_skip_file(self, path: Path) -> bool:
        """Check if a file should be skipped.

        Args:
            path: File path to check

        Returns:
            True if file should be skipped
        """
        skip_patterns = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".next",
            "dist",
            "build",
            ".env",
            ".env.local",
        }

        # Check if any skip pattern is in the path
        for pattern in skip_patterns:
            if pattern in path.parts:
                return True

        # Skip large files
        try:
            if path.stat().st_size > 1_000_000:  # 1MB
                return True
        except OSError:
            return True

        return False

    def process_prompt(
        self,
        prompt: str,
        base_path: str = ".",
        include_context: bool = True,
    ) -> Dict[str, Any]:
        """Process prompt and inject workflow context.

        Args:
            prompt: User prompt text
            base_path: Base directory for file search
            include_context: Whether to include file contents

        Returns:
            Dict with processed prompt and metadata
        """
        detected_triggers = self.detect_triggers(prompt)

        if not detected_triggers:
            return {
                "original_prompt": prompt,
                "processed_prompt": prompt,
                "triggers": [],
                "context_injected": False,
            }

        context_sections = []
        metadata = {
            "triggers_detected": list(detected_triggers),
            "manifests": [],
        }

        # Process each trigger
        for trigger in sorted(detected_triggers):
            trigger_data = self.extract_trigger_context(trigger, base_path)
            manifest = trigger_data.get("manifest")

            if manifest:
                metadata["manifests"].append(manifest.to_dict())
                context_sections.append(
                    f"\n[WORKFLOW: {manifest.trigger}]\n"
                    f"{manifest.system_prompt_addition}\n"
                )

        # Build enhanced prompt
        enhanced_prompt = prompt
        if context_sections:
            context_injection = "".join(context_sections)
            enhanced_prompt = f"{prompt}\n\n{context_injection}"

        return {
            "original_prompt": prompt,
            "processed_prompt": enhanced_prompt,
            "triggers": list(detected_triggers),
            "context_injected": bool(context_sections),
            "metadata": metadata,
        }


class WorkflowExecutor:
    """Executes workflows by injecting context into archetype agents."""

    def __init__(
        self,
        registry: Optional[WorkflowRegistry] = None,
        base_path: str = ".",
    ):
        """Initialize workflow executor.

        Args:
            registry: WorkflowRegistry instance
            base_path: Base directory for file searches
        """
        self.registry = registry or WorkflowRegistry()
        self.trigger = WorkflowTrigger(registry)
        self.base_path = base_path

    def prepare_context(
        self,
        prompt: str,
        archetype: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare context for a prompt, optionally filtered by archetype.

        Args:
            prompt: User prompt with potential @Triggers
            archetype: Optional archetype to filter workflows

        Returns:
            Dict with processed prompt, archetype, and system additions
        """
        processed = self.trigger.process_prompt(
            prompt, self.base_path, include_context=True
        )

        if not processed["triggers"]:
            return {
                "prompt": prompt,
                "archetype": archetype or "builder",
                "system_prompt_addition": "",
                "temperature": None,
            }

        # Determine archetype from triggers if not specified
        if not archetype:
            # Get archetype from first trigger's manifest
            for trigger_str in processed["triggers"]:
                manifest = self.registry.get(trigger_str)
                if manifest:
                    archetype = manifest.archetype
                    break

        # Aggregate system prompts from all triggers
        system_additions = []
        temperature_override = None

        for trigger_str in processed["triggers"]:
            manifest = self.registry.get(trigger_str)
            if manifest:
                system_additions.append(manifest.system_prompt_addition)
                # Use temperature from highest priority trigger
                if temperature_override is None and manifest.temperature is not None:
                    temperature_override = manifest.temperature

        return {
            "prompt": processed["processed_prompt"],
            "archetype": archetype or "builder",
            "system_prompt_addition": "\n\n".join(system_additions),
            "temperature": temperature_override,
            "triggers": processed["triggers"],
            "metadata": processed.get("metadata"),
        }


# Singleton instance
_default_registry: Optional[WorkflowRegistry] = None
_default_trigger: Optional[WorkflowTrigger] = None
_default_executor: Optional[WorkflowExecutor] = None


def get_workflow_registry() -> WorkflowRegistry:
    """Get or create the default workflow registry.

    Returns:
        Global WorkflowRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = WorkflowRegistry()
    return _default_registry


def get_workflow_trigger() -> WorkflowTrigger:
    """Get or create the default workflow trigger.

    Returns:
        Global WorkflowTrigger instance
    """
    global _default_trigger
    if _default_trigger is None:
        _default_trigger = WorkflowTrigger(get_workflow_registry())
    return _default_trigger


def get_workflow_executor(base_path: str = ".") -> WorkflowExecutor:
    """Get or create the default workflow executor.

    Args:
        base_path: Base directory for file searches

    Returns:
        Global WorkflowExecutor instance
    """
    global _default_executor
    if _default_executor is None:
        _default_executor = WorkflowExecutor(
            get_workflow_registry(), base_path=base_path
        )
    return _default_executor


def process_prompt_with_workflows(
    prompt: str,
    base_path: str = ".",
    archetype: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a prompt with workflow triggers.

    This is the primary entry point for workflow processing.

    Args:
        prompt: User prompt, may contain @Triggers
        base_path: Base directory for file searches
        archetype: Optional archetype override

    Returns:
        Dict with processed prompt and context metadata
    """
    executor = get_workflow_executor(base_path)
    return executor.prepare_context(prompt, archetype=archetype)
