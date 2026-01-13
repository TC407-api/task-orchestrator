"""Agent Archetype Registry for dynamic tool management and role-based access control.

This module provides a registry system that filters available tools based on agent
archetypes, preventing context pollution and enforcing role-based permissions.

Archetypes:
- Architect: Read-only exploration and analysis tools
- Builder: Implementation and file manipulation tools
- QC: Testing, linting, and validation tools
- Researcher: Web search, documentation, and exploration tools
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Archetype(Enum):
    """Agent role archetypes for role-based access control."""
    ARCHITECT = "architect"
    BUILDER = "builder"
    QC = "qc"
    RESEARCHER = "researcher"


# Tool categories for organization
CATEGORY_READONLY = "readonly"
CATEGORY_WRITE = "write"
CATEGORY_VALIDATION = "validation"
CATEGORY_EXPLORATION = "exploration"


# Define all available tools and their categories
ARCHETYPE_TOOLS = {
    Archetype.ARCHITECT: {
        "tools": [
            # File exploration (read-only)
            "read_file",
            "list_files",
            "search_docs",
            "grep_code",
            "find_files",
            "get_file_structure",
            # Analysis
            "analyze_code",
            "describe_architecture",
            "identify_patterns",
            "trace_dependencies",
            "complexity_analysis",
            # Documentation
            "read_docs",
            "search_documentation",
            "get_examples",
            # Task orchestration
            "tasks_list",
            "tasks_analyze",
            "tasks_briefing",
        ],
        "category": CATEGORY_READONLY,
        "temperature": 0.3,
        "description": "Architect: Reads code, analyzes patterns, designs solutions. No file mutations.",
    },
    Archetype.BUILDER: {
        "tools": [
            # File operations
            "read_file",
            "write_file",
            "edit_file",
            "delete_file",
            "create_file",
            "rename_file",
            # Code generation
            "generate_code",
            "scaffold_project",
            "apply_template",
            # Build operations
            "build_project",
            "compile_code",
            "generate_types",
            # File exploration
            "list_files",
            "find_files",
            "get_file_structure",
            # Task management
            "tasks_add",
            "tasks_complete",
            "tasks_update",
        ],
        "category": CATEGORY_WRITE,
        "temperature": 0.5,
        "description": "Builder: Implements features, writes code, manages files. Full write access.",
    },
    Archetype.QC: {
        "tools": [
            # Testing
            "run_tests",
            "run_test_suite",
            "generate_tests",
            "coverage_report",
            # Validation
            "lint_code",
            "type_check",
            "security_scan",
            "performance_check",
            "format_check",
            "validate_schema",
            # Code quality
            "complexity_analysis",
            "code_review",
            "identify_issues",
            "check_dependencies",
            # File reading (for analysis)
            "read_file",
            "list_files",
            "find_files",
            # Task management
            "tasks_analyze",
            "tasks_list",
        ],
        "category": CATEGORY_VALIDATION,
        "temperature": 0.2,
        "description": "QC: Validates code quality, runs tests, performs security checks. Read-only file access.",
    },
    Archetype.RESEARCHER: {
        "tools": [
            # Web search and exploration
            "web_search",
            "fetch_docs",
            "fetch_url",
            "search_documentation",
            "get_examples",
            # Content analysis
            "summarize_content",
            "extract_information",
            "analyze_data",
            # File reading
            "read_file",
            "list_files",
            "find_files",
            "search_docs",
            "grep_code",
            # Knowledge base
            "search_knowledge_base",
            "get_reference",
            # Task management
            "tasks_add",
            "tasks_list",
            "tasks_briefing",
        ],
        "category": CATEGORY_EXPLORATION,
        "temperature": 0.7,
        "description": "Researcher: Explores documentation, searches web, gathers information. No file mutations.",
    },
}


# System prompts tailored per archetype
ARCHETYPE_SYSTEM_PROMPTS = {
    Archetype.ARCHITECT: """You are an expert Software Architect. Your role is to:
1. Analyze code structure and patterns
2. Design scalable, maintainable solutions
3. Provide architectural guidance and recommendations
4. Identify design patterns and best practices
5. Create high-level documentation and diagrams

CRITICAL CONSTRAINT: You have READ-ONLY access to files. You cannot modify, create, or delete files.
You provide designs and analysis, which others implement.

Focus on:
- System design and architecture
- Pattern identification
- Dependency analysis
- Performance implications
- Scalability considerations
- Design patterns and best practices""",

    Archetype.BUILDER: """You are an expert Software Builder. Your role is to:
1. Implement features from specifications
2. Write clean, maintainable code
3. Create files and modify existing code
4. Follow established patterns and conventions
5. Handle build and compilation tasks

You have FULL FILE ACCESS. You can read, write, create, and delete files.
Use this power responsibly to implement specifications accurately.

Focus on:
- Feature implementation
- Code quality and readability
- Following established patterns
- Error handling and edge cases
- Performance-conscious implementation
- Clear, descriptive variable names and comments""",

    Archetype.QC: """You are an expert Quality Control Engineer. Your role is to:
1. Run comprehensive test suites
2. Validate code quality and standards
3. Perform security scanning
4. Check for performance issues
5. Identify bugs and edge cases

CRITICAL CONSTRAINT: You have READ-ONLY access to files. You cannot modify code.
Your job is to find issues and report them for others to fix.

Focus on:
- Test coverage and edge cases
- Code style consistency
- Security vulnerabilities
- Performance bottlenecks
- Type safety and contracts
- Dependency conflicts
- Clear issue reporting with reproduction steps""",

    Archetype.RESEARCHER: """You are an expert Technical Researcher. Your role is to:
1. Search and explore documentation
2. Gather information from web and knowledge bases
3. Analyze and synthesize findings
4. Provide comprehensive research summaries
5. Identify relevant examples and references

CRITICAL CONSTRAINT: You have READ-ONLY access to files. You cannot modify code.
Your focus is on discovery and knowledge gathering.

Focus on:
- Documentation accuracy
- Comprehensive information gathering
- Citation and attribution
- Synthesizing information from multiple sources
- Clear, organized research summaries
- Relevant examples and use cases""",
}


# Default temperatures per archetype (for LLM calls)
ARCHETYPE_TEMPERATURES = {
    Archetype.ARCHITECT: 0.3,      # Lower: analytical, deterministic
    Archetype.BUILDER: 0.5,         # Medium: balanced creativity and structure
    Archetype.QC: 0.2,              # Very low: strict, rule-based
    Archetype.RESEARCHER: 0.7,      # Higher: exploratory, creative
}


@dataclass
class ArchetypeConfig:
    """Configuration for an agent archetype."""
    archetype: Archetype
    tools: list[str]
    category: str
    description: str
    system_prompt: str
    temperature: float
    max_retries: int = 3
    timeout_seconds: int = 300

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.tools:
            raise ValueError(f"Archetype {self.archetype.value} must have at least one tool")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")


class ArchetypeRegistry:
    """Registry managing tool permissions and configurations per archetype."""

    def __init__(self):
        """Initialize the archetype registry."""
        self._registry = self._build_registry()
        self._cache = {}

    def _build_registry(self) -> dict[Archetype, ArchetypeConfig]:
        """Build the archetype configuration registry."""
        registry = {}
        for archetype in Archetype:
            tool_info = ARCHETYPE_TOOLS[archetype]
            config = ArchetypeConfig(
                archetype=archetype,
                tools=tool_info["tools"],
                category=tool_info["category"],
                description=tool_info["description"],
                system_prompt=ARCHETYPE_SYSTEM_PROMPTS[archetype],
                temperature=ARCHETYPE_TEMPERATURES[archetype],
            )
            registry[archetype] = config
        return registry

    def get_archetype_config(self, archetype: Archetype) -> ArchetypeConfig:
        """Get the complete configuration for an archetype.

        Args:
            archetype: The archetype to get config for

        Returns:
            ArchetypeConfig with all settings for the archetype

        Raises:
            ValueError: If archetype is not registered
        """
        if archetype not in self._registry:
            raise ValueError(f"Unknown archetype: {archetype.value}")
        return self._registry[archetype]

    def get_tools(self, archetype: Archetype) -> list[str]:
        """Get list of allowed tools for an archetype.

        Args:
            archetype: The archetype to get tools for

        Returns:
            List of tool names available to this archetype
        """
        config = self.get_archetype_config(archetype)
        return config.tools.copy()

    def is_allowed(self, archetype: Archetype, tool: str) -> bool:
        """Check if an archetype can use a specific tool.

        Args:
            archetype: The archetype to check
            tool: The tool name to check

        Returns:
            True if the tool is allowed for this archetype, False otherwise
        """
        cache_key = (archetype, tool)
        if cache_key in self._cache:
            return self._cache[cache_key]

        allowed = tool in self.get_tools(archetype)
        self._cache[cache_key] = allowed
        return allowed

    def filter_tools(
        self,
        archetype: Archetype,
        tools: list[str],
    ) -> list[str]:
        """Filter a list of tools to only those allowed for an archetype.

        Args:
            archetype: The archetype to filter for
            tools: List of tool names to filter

        Returns:
            List of tools that are allowed for this archetype

        Example:
            >>> registry = ArchetypeRegistry()
            >>> registry.filter_tools(
            ...     Archetype.BUILDER,
            ...     ["read_file", "delete_file", "web_search"]
            ... )
            ["read_file", "delete_file"]
        """
        allowed_tools = self.get_tools(archetype)
        filtered = [t for t in tools if t in allowed_tools]
        return filtered

    def get_system_prompt(self, archetype: Archetype) -> str:
        """Get the system prompt for an archetype.

        Args:
            archetype: The archetype to get prompt for

        Returns:
            System prompt string for this archetype
        """
        config = self.get_archetype_config(archetype)
        return config.system_prompt

    def get_temperature(self, archetype: Archetype) -> float:
        """Get the LLM temperature setting for an archetype.

        Args:
            archetype: The archetype to get temperature for

        Returns:
            Temperature value (0.0-2.0) for this archetype
        """
        config = self.get_archetype_config(archetype)
        return config.temperature

    def get_category(self, archetype: Archetype) -> str:
        """Get the tool category for an archetype.

        Args:
            archetype: The archetype to get category for

        Returns:
            Category string describing the archetype's focus
        """
        config = self.get_archetype_config(archetype)
        return config.category

    def get_description(self, archetype: Archetype) -> str:
        """Get a human-readable description of an archetype.

        Args:
            archetype: The archetype to describe

        Returns:
            Description string
        """
        config = self.get_archetype_config(archetype)
        return config.description

    def validate_tool_usage(
        self,
        archetype: Archetype,
        tool: str,
    ) -> tuple[bool, str]:
        """Validate that an archetype can use a tool, with detailed feedback.

        Args:
            archetype: The archetype attempting to use the tool
            tool: The tool being attempted

        Returns:
            Tuple of (allowed: bool, message: str)

        Example:
            >>> registry = ArchetypeRegistry()
            >>> allowed, msg = registry.validate_tool_usage(
            ...     Archetype.ARCHITECT, "delete_file"
            ... )
            >>> print(allowed)
            False
            >>> print(msg)
            "Tool 'delete_file' not allowed for Architect. Valid tools: ..."
        """
        if self.is_allowed(archetype, tool):
            return True, f"Tool '{tool}' is allowed for {archetype.value}"

        allowed_tools = self.get_tools(archetype)
        tools_str = ", ".join(allowed_tools[:5])
        if len(allowed_tools) > 5:
            tools_str += f", ... ({len(allowed_tools) - 5} more)"

        message = (
            f"Tool '{tool}' not allowed for {archetype.value}. "
            f"Valid tools for this archetype: {tools_str}"
        )
        return False, message

    def get_all_archetypes(self) -> list[Archetype]:
        """Get all registered archetypes.

        Returns:
            List of all Archetype enum values
        """
        return list(Archetype)

    def get_archetype_by_name(self, name: str) -> Optional[Archetype]:
        """Get an archetype by its string name.

        Args:
            name: The archetype name (e.g., "architect", "builder")

        Returns:
            The Archetype enum value, or None if not found

        Example:
            >>> registry = ArchetypeRegistry()
            >>> archetype = registry.get_archetype_by_name("builder")
            >>> archetype == Archetype.BUILDER
            True
        """
        try:
            return Archetype(name.lower())
        except ValueError:
            return None

    def get_tools_by_category(self, category: str) -> dict[Archetype, list[str]]:
        """Get all tools grouped by archetype for a specific category.

        Args:
            category: The category to filter by (e.g., "readonly", "write")

        Returns:
            Dictionary mapping archetype to list of tools in that category

        Example:
            >>> registry = ArchetypeRegistry()
            >>> tools = registry.get_tools_by_category("write")
            >>> tools[Archetype.BUILDER]
            ["read_file", "write_file", "edit_file", ...]
        """
        result = {}
        for archetype in Archetype:
            config = self.get_archetype_config(archetype)
            if config.category == category:
                result[archetype] = config.tools
        return result

    def is_write_allowed(self, archetype: Archetype) -> bool:
        """Check if an archetype has write permissions.

        Args:
            archetype: The archetype to check

        Returns:
            True if archetype has write tools (file modification), False otherwise
        """
        config = self.get_archetype_config(archetype)
        return config.category in [CATEGORY_WRITE]

    def is_readonly(self, archetype: Archetype) -> bool:
        """Check if an archetype is read-only.

        Args:
            archetype: The archetype to check

        Returns:
            True if archetype has no write permissions, False otherwise
        """
        return not self.is_write_allowed(archetype)

    def get_summary(self) -> dict:
        """Get a complete summary of all archetypes and their tools.

        Returns:
            Dictionary with archetype information and tool listings
        """
        summary = {}
        for archetype in Archetype:
            config = self.get_archetype_config(archetype)
            summary[archetype.value] = {
                "description": config.description,
                "category": config.category,
                "tool_count": len(config.tools),
                "temperature": config.temperature,
                "tools": config.tools,
                "readonly": self.is_readonly(archetype),
            }
        return summary


# Global registry instance
_global_registry: Optional[ArchetypeRegistry] = None


def get_archetype_registry() -> ArchetypeRegistry:
    """Get or create the global archetype registry singleton.

    Returns:
        The global ArchetypeRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ArchetypeRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (primarily for testing)."""
    global _global_registry
    _global_registry = None
