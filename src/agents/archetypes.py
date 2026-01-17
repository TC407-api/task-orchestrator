"""Agent Archetypes System for Task Orchestrator.

Defines 4 distinct agent archetypes with specialized capabilities:
- ARCHITECT: Plans, specs, structure (read-only tools, low temperature)
- BUILDER: Writes code (file write, AST tools, medium temperature)
- QC: Testing, linting, validation (LSP, linters, test runners, zero temperature)
- RESEARCHER: Docs, web search, exploration (web, scrapers, high temperature)

Each archetype has:
1. Defined role and responsibilities
2. Tool access list (whitelist of allowed tools)
3. Temperature setting for LLM behavior
4. Token budget per operation
5. Retry policy and timeout settings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set


class ArchetypeType(Enum):
    """The 4 primary agent archetypes."""
    ARCHITECT = "architect"
    BUILDER = "builder"
    QC = "qc"
    RESEARCHER = "researcher"


class ToolCategory(Enum):
    """Categories of tools that can be filtered per archetype."""
    # File operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"

    # Code analysis
    AST_PARSE = "ast_parse"
    CODE_ANALYSIS = "code_analysis"
    LINT = "lint"

    # Testing
    TEST_RUN = "test_run"
    TEST_DEBUG = "test_debug"
    COVERAGE = "coverage"

    # Web/Network
    WEB_SEARCH = "web_search"
    WEB_SCRAPE = "web_scrape"
    API_CALL = "api_call"

    # System
    SHELL_EXEC = "shell_exec"
    ENV_ACCESS = "env_access"

    # LLM/Reasoning
    LLM_CALL = "llm_call"
    SEARCH_MEMORY = "search_memory"

    # Quality gates
    ASYNC_SPAWN = "async_spawn"
    PARALLEL_SPAWN = "parallel_spawn"


class ToolAccessLevel(Enum):
    """How an archetype can use a tool."""
    NONE = "none"           # Tool not available
    READ_ONLY = "read_only" # Can only read
    WRITE = "write"         # Can read and write
    EXECUTE = "execute"     # Can execute/run
    FULL = "full"           # Full access


@dataclass
class ToolPermission:
    """Permission specification for a single tool."""
    name: str
    category: ToolCategory
    access_level: ToolAccessLevel
    max_calls_per_session: Optional[int] = None
    timeout_seconds: Optional[int] = None
    requires_approval: bool = False
    description: str = ""


@dataclass
class ArchetypeConfig:
    """Configuration for an agent archetype."""

    # Identity
    type: ArchetypeType
    name: str
    description: str

    # LLM parameters
    temperature: float  # 0.0 (deterministic) to 1.0 (creative)
    max_tokens: int
    model_preference: str  # e.g., "gemini-3-flash", "gemini-3-pro"

    # Token budgets
    tokens_per_session: int
    tokens_per_operation: int

    # Operational constraints
    max_concurrent_operations: int
    timeout_seconds: int
    retry_policy: "RetryPolicy"

    # Tool access
    allowed_tools: List[ToolPermission] = field(default_factory=list)
    denied_tool_patterns: List[str] = field(default_factory=list)  # Regex patterns

    # Guardrails
    require_safety_check: bool = True
    require_approval_for_mutations: bool = False
    enable_adaptive_temperature: bool = False

    # Rate limiting
    calls_per_minute: int = 60
    burst_size: int = 10

    def get_tool_permission(self, tool_name: str) -> Optional[ToolPermission]:
        """Look up permission for a specific tool."""
        return next(
            (t for t in self.allowed_tools if t.name == tool_name),
            None
        )

    def can_access_tool(self, tool_name: str, required_level: ToolAccessLevel) -> bool:
        """Check if archetype can access a tool at required level."""
        # Check denied patterns first
        import re
        for pattern in self.denied_tool_patterns:
            if re.match(pattern, tool_name):
                return False

        perm = self.get_tool_permission(tool_name)
        if not perm:
            return False

        # Map access levels to integer hierarchy
        level_hierarchy = {
            ToolAccessLevel.NONE: 0,
            ToolAccessLevel.READ_ONLY: 1,
            ToolAccessLevel.WRITE: 2,
            ToolAccessLevel.EXECUTE: 3,
            ToolAccessLevel.FULL: 4,
        }

        return level_hierarchy[perm.access_level] >= level_hierarchy[required_level]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_preference": self.model_preference,
            "tokens_per_session": self.tokens_per_session,
            "tokens_per_operation": self.tokens_per_operation,
            "max_concurrent_operations": self.max_concurrent_operations,
            "timeout_seconds": self.timeout_seconds,
            "retry_policy": self.retry_policy.to_dict(),
            "calls_per_minute": self.calls_per_minute,
            "burst_size": self.burst_size,
            "guardrails": {
                "require_safety_check": self.require_safety_check,
                "require_approval_for_mutations": self.require_approval_for_mutations,
                "enable_adaptive_temperature": self.enable_adaptive_temperature,
            },
        }


@dataclass
class RetryPolicy:
    """Retry strategy for archetype operations."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_enabled: bool = True
    retryable_errors: List[str] = field(default_factory=lambda: [
        "timeout",
        "rate_limit",
        "temporary_failure",
        "connection_error",
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_retries": self.max_retries,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "exponential_base": self.exponential_base,
            "jitter_enabled": self.jitter_enabled,
            "retryable_errors": self.retryable_errors,
        }


class ArchetypeFactory:
    """Factory for creating standard archetype configurations."""

    @staticmethod
    def create_architect() -> ArchetypeConfig:
        """Create ARCHITECT archetype: Plans, specs, structure."""
        return ArchetypeConfig(
            type=ArchetypeType.ARCHITECT,
            name="Architect",
            description="Plans system structure, writes specifications, analyzes architecture. Read-only tools only.",
            temperature=0.1,  # Low temperature for consistency
            max_tokens=4096,
            model_preference="gemini-3-pro-preview",
            tokens_per_session=16000,
            tokens_per_operation=4000,
            max_concurrent_operations=2,
            timeout_seconds=30,
            retry_policy=RetryPolicy(max_retries=2),
            calls_per_minute=30,
            burst_size=5,
            require_safety_check=True,
            require_approval_for_mutations=False,
            allowed_tools=[
                # File reading
                ToolPermission(
                    name="read_file",
                    category=ToolCategory.FILE_READ,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=200,
                    timeout_seconds=10,
                    description="Read file contents for analysis"
                ),
                ToolPermission(
                    name="glob_search",
                    category=ToolCategory.FILE_READ,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=100,
                    timeout_seconds=10,
                    description="Search files by glob pattern"
                ),
                # Code analysis
                ToolPermission(
                    name="grep_search",
                    category=ToolCategory.CODE_ANALYSIS,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=150,
                    timeout_seconds=10,
                    description="Search code patterns"
                ),
                ToolPermission(
                    name="ast_parse",
                    category=ToolCategory.AST_PARSE,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=50,
                    timeout_seconds=5,
                    description="Parse code into AST for analysis"
                ),
                # Web/docs
                ToolPermission(
                    name="web_search",
                    category=ToolCategory.WEB_SEARCH,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=20,
                    timeout_seconds=15,
                    description="Search documentation"
                ),
                # Memory
                ToolPermission(
                    name="search_memory",
                    category=ToolCategory.SEARCH_MEMORY,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=30,
                    timeout_seconds=5,
                    description="Query knowledge graph for patterns"
                ),
            ],
            denied_tool_patterns=[
                r".*write.*",
                r".*delete.*",
                r".*spawn.*",
                r".*execute.*",
            ],
        )

    @staticmethod
    def create_builder() -> ArchetypeConfig:
        """Create BUILDER archetype: Writes code, implements features."""
        return ArchetypeConfig(
            type=ArchetypeType.BUILDER,
            name="Builder",
            description="Implements features, writes code, modifies files. Can read, write, and test.",
            temperature=0.3,  # Moderate temperature for some creativity
            max_tokens=8192,
            model_preference="gemini-3-flash-preview",
            tokens_per_session=32000,
            tokens_per_operation=8000,
            max_concurrent_operations=1,
            timeout_seconds=60,
            retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=2.0),
            calls_per_minute=60,
            burst_size=10,
            require_safety_check=True,
            require_approval_for_mutations=False,
            enable_adaptive_temperature=True,
            allowed_tools=[
                # File operations (read and write)
                ToolPermission(
                    name="read_file",
                    category=ToolCategory.FILE_READ,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=300,
                    timeout_seconds=10,
                    description="Read file contents"
                ),
                ToolPermission(
                    name="write_file",
                    category=ToolCategory.FILE_WRITE,
                    access_level=ToolAccessLevel.WRITE,
                    max_calls_per_session=100,
                    timeout_seconds=10,
                    requires_approval=True,
                    description="Write file contents (requires approval)"
                ),
                ToolPermission(
                    name="edit_file",
                    category=ToolCategory.FILE_WRITE,
                    access_level=ToolAccessLevel.WRITE,
                    max_calls_per_session=150,
                    timeout_seconds=10,
                    requires_approval=True,
                    description="Edit file with replacements (requires approval)"
                ),
                ToolPermission(
                    name="glob_search",
                    category=ToolCategory.FILE_READ,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=200,
                    timeout_seconds=10,
                    description="Find files by pattern"
                ),
                # Code analysis
                ToolPermission(
                    name="grep_search",
                    category=ToolCategory.CODE_ANALYSIS,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=200,
                    timeout_seconds=10,
                    description="Search code patterns"
                ),
                ToolPermission(
                    name="ast_parse",
                    category=ToolCategory.AST_PARSE,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=100,
                    timeout_seconds=5,
                    description="Parse code into AST"
                ),
                # Testing
                ToolPermission(
                    name="run_tests",
                    category=ToolCategory.TEST_RUN,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=50,
                    timeout_seconds=60,
                    description="Run test suite"
                ),
                # Shell execution (limited)
                ToolPermission(
                    name="bash_exec",
                    category=ToolCategory.SHELL_EXEC,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=30,
                    timeout_seconds=30,
                    requires_approval=True,
                    description="Execute shell commands (requires approval)"
                ),
                # Spawning sub-agents
                ToolPermission(
                    name="spawn_agent",
                    category=ToolCategory.ASYNC_SPAWN,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=5,
                    timeout_seconds=120,
                    requires_approval=True,
                    description="Spawn specialized agent (requires approval)"
                ),
            ],
            denied_tool_patterns=[
                r".*delete.*",
            ],
        )

    @staticmethod
    def create_qc() -> ArchetypeConfig:
        """Create QC archetype: Testing, linting, validation."""
        return ArchetypeConfig(
            type=ArchetypeType.QC,
            name="QC",
            description="Validates code quality, runs tests, checks coverage, performs security scans. Zero tolerance for issues.",
            temperature=0.0,  # Deterministic validation
            max_tokens=4096,
            model_preference="gemini-3-flash-preview",
            tokens_per_session=16000,
            tokens_per_operation=4000,
            max_concurrent_operations=3,
            timeout_seconds=120,
            retry_policy=RetryPolicy(max_retries=2, base_delay_seconds=1.0),
            calls_per_minute=120,
            burst_size=20,
            require_safety_check=True,
            require_approval_for_mutations=False,
            allowed_tools=[
                # File reading for validation
                ToolPermission(
                    name="read_file",
                    category=ToolCategory.FILE_READ,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=500,
                    timeout_seconds=10,
                    description="Read files for validation"
                ),
                ToolPermission(
                    name="glob_search",
                    category=ToolCategory.FILE_READ,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=300,
                    timeout_seconds=10,
                    description="Find files to validate"
                ),
                # Code analysis and linting
                ToolPermission(
                    name="lint",
                    category=ToolCategory.LINT,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=100,
                    timeout_seconds=30,
                    description="Run linters (pylint, eslint, etc.)"
                ),
                ToolPermission(
                    name="grep_search",
                    category=ToolCategory.CODE_ANALYSIS,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=300,
                    timeout_seconds=10,
                    description="Search for code patterns"
                ),
                # Testing
                ToolPermission(
                    name="run_tests",
                    category=ToolCategory.TEST_RUN,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=100,
                    timeout_seconds=120,
                    description="Run test suite"
                ),
                ToolPermission(
                    name="coverage",
                    category=ToolCategory.COVERAGE,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=50,
                    timeout_seconds=60,
                    description="Generate code coverage reports"
                ),
                # Security and validation
                ToolPermission(
                    name="security_scan",
                    category=ToolCategory.CODE_ANALYSIS,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=30,
                    timeout_seconds=60,
                    description="Scan for security issues"
                ),
                ToolPermission(
                    name="type_check",
                    category=ToolCategory.CODE_ANALYSIS,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=50,
                    timeout_seconds=30,
                    description="Run type checker (mypy, pyright)"
                ),
            ],
            denied_tool_patterns=[
                r".*write.*",
                r".*delete.*",
                r".*spawn.*",
                r".*edit.*",
            ],
        )

    @staticmethod
    def create_researcher() -> ArchetypeConfig:
        """Create RESEARCHER archetype: Exploration, web search, documentation."""
        return ArchetypeConfig(
            type=ArchetypeType.RESEARCHER,
            name="Researcher",
            description="Explores documentation, performs research, gathers information. High creativity for discovery.",
            temperature=0.7,  # High temperature for exploration
            max_tokens=6144,
            model_preference="gemini-3-flash-preview",
            tokens_per_session=24000,
            tokens_per_operation=6000,
            max_concurrent_operations=2,
            timeout_seconds=45,
            retry_policy=RetryPolicy(max_retries=2, base_delay_seconds=1.0),
            calls_per_minute=40,
            burst_size=8,
            require_safety_check=True,
            require_approval_for_mutations=False,
            allowed_tools=[
                # Web search and scraping
                ToolPermission(
                    name="web_search",
                    category=ToolCategory.WEB_SEARCH,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=50,
                    timeout_seconds=15,
                    description="Search the web for information"
                ),
                ToolPermission(
                    name="web_fetch",
                    category=ToolCategory.WEB_SCRAPE,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=100,
                    timeout_seconds=20,
                    description="Fetch and parse web content"
                ),
                # File reading for docs
                ToolPermission(
                    name="read_file",
                    category=ToolCategory.FILE_READ,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=400,
                    timeout_seconds=10,
                    description="Read documentation files"
                ),
                ToolPermission(
                    name="glob_search",
                    category=ToolCategory.FILE_READ,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=200,
                    timeout_seconds=10,
                    description="Find documentation files"
                ),
                # Code analysis (read-only)
                ToolPermission(
                    name="grep_search",
                    category=ToolCategory.CODE_ANALYSIS,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=200,
                    timeout_seconds=10,
                    description="Search code for context"
                ),
                # Memory and knowledge
                ToolPermission(
                    name="search_memory",
                    category=ToolCategory.SEARCH_MEMORY,
                    access_level=ToolAccessLevel.READ_ONLY,
                    max_calls_per_session=100,
                    timeout_seconds=5,
                    description="Query knowledge graph"
                ),
                ToolPermission(
                    name="add_memory",
                    category=ToolCategory.SEARCH_MEMORY,
                    access_level=ToolAccessLevel.WRITE,
                    max_calls_per_session=50,
                    timeout_seconds=5,
                    description="Add research findings to knowledge graph"
                ),
                # API calls (controlled)
                ToolPermission(
                    name="api_call",
                    category=ToolCategory.API_CALL,
                    access_level=ToolAccessLevel.EXECUTE,
                    max_calls_per_session=30,
                    timeout_seconds=30,
                    requires_approval=True,
                    description="Make API calls for data gathering (requires approval)"
                ),
            ],
            denied_tool_patterns=[
                r".*write_file.*",
                r".*edit_file.*",
                r".*delete.*",
                r".*spawn.*",
                r".*execute.*",
            ],
        )


class ArchetypeRegistry:
    """Registry of all available archetypes."""

    def __init__(self):
        """Initialize with standard archetypes."""
        self._archetypes: Dict[ArchetypeType, ArchetypeConfig] = {
            ArchetypeType.ARCHITECT: ArchetypeFactory.create_architect(),
            ArchetypeType.BUILDER: ArchetypeFactory.create_builder(),
            ArchetypeType.QC: ArchetypeFactory.create_qc(),
            ArchetypeType.RESEARCHER: ArchetypeFactory.create_researcher(),
        }

    def get(self, archetype_type: ArchetypeType) -> Optional[ArchetypeConfig]:
        """Get archetype by type."""
        return self._archetypes.get(archetype_type)

    def get_by_name(self, name: str) -> Optional[ArchetypeConfig]:
        """Get archetype by name (case-insensitive)."""
        name_lower = name.lower()
        return next(
            (cfg for cfg in self._archetypes.values()
             if cfg.name.lower() == name_lower),
            None
        )

    def list_archetypes(self) -> List[ArchetypeConfig]:
        """List all registered archetypes."""
        return list(self._archetypes.values())

    def register_custom(self, config: ArchetypeConfig) -> None:
        """Register a custom archetype configuration."""
        self._archetypes[config.type] = config

    def get_all_as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get all archetypes as serialized dictionary."""
        return {
            cfg.type.value: cfg.to_dict()
            for cfg in self._archetypes.values()
        }


class ToolFilter:
    """Filters tool access based on archetype configuration."""

    def __init__(self, archetype: ArchetypeConfig):
        """Initialize with an archetype configuration."""
        self.archetype = archetype
        self._allowed_tool_names: Set[str] = {
            t.name for t in archetype.allowed_tools
        }

    def filter_tools(self, available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter list of MCP tools based on archetype permissions."""
        filtered = []

        for tool in available_tools:
            tool_name = tool.get("name", "")

            # Check if tool is in allowed list
            if self._is_allowed(tool_name):
                # Optionally strip sensitive metadata
                filtered_tool = self._apply_restrictions(tool)
                filtered.append(filtered_tool)

        return filtered

    def _is_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed for this archetype."""
        import re

        # Check denied patterns first
        for pattern in self.archetype.denied_tool_patterns:
            if re.match(pattern, tool_name):
                return False

        # Check allowed list
        return tool_name in self._allowed_tool_names

    def _apply_restrictions(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Apply archetype-specific restrictions to tool definition."""
        tool_name = tool.get("name", "")
        perm = self.archetype.get_tool_permission(tool_name)

        if not perm:
            return tool

        # Add archetype-specific metadata
        restricted_tool = tool.copy()
        restricted_tool["archetype_access_level"] = perm.access_level.value
        restricted_tool["archetype_timeout"] = perm.timeout_seconds
        restricted_tool["archetype_max_calls"] = perm.max_calls_per_session
        restricted_tool["requires_approval"] = perm.requires_approval

        return restricted_tool

    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get archetype-specific metadata for a tool."""
        perm = self.archetype.get_tool_permission(tool_name)
        if not perm:
            return None

        return {
            "access_level": perm.access_level.value,
            "timeout_seconds": perm.timeout_seconds,
            "max_calls_per_session": perm.max_calls_per_session,
            "requires_approval": perm.requires_approval,
            "description": perm.description,
        }


class AgentSystemPromptBuilder:
    """Builds system prompts tailored to each archetype."""

    @staticmethod
    def build_prompt(archetype: ArchetypeConfig) -> str:
        """Build a system prompt for the given archetype."""
        base_prompt = f"""You are a {archetype.name} agent in a task orchestration system.

Role: {archetype.description}

Core Responsibilities:
"""

        if archetype.type == ArchetypeType.ARCHITECT:
            prompt_body = """- Analyze system requirements and design solutions
- Create architectural specifications
- Perform code reviews and architectural audits
- Document design decisions
- Suggest improvements and refactoring patterns
- Research design patterns and best practices

Constraints:
- You have READ-ONLY access to files and code
- You cannot modify, create, or delete files
- Focus on planning and analysis, not implementation
- Provide clear, actionable specifications for builders"""

        elif archetype.type == ArchetypeType.BUILDER:
            prompt_body = """- Implement features based on specifications
- Write clean, tested code
- Create and modify files as needed
- Run tests to verify implementation
- Collaborate with QC to ensure quality
- Handle dependencies and integration

Constraints:
- Follow specifications provided by architects
- Ensure code passes all tests before completion
- Request approval for critical file modifications
- Document your implementation decisions
- Maintain code quality standards"""

        elif archetype.type == ArchetypeType.QC:
            prompt_body = """- Run comprehensive test suites
- Perform code quality checks and linting
- Measure and verify code coverage
- Identify security issues
- Validate type safety
- Report quality metrics

Constraints:
- Zero tolerance for failing tests
- All quality checks must pass
- Cannot modify code (read-only testing)
- Provide actionable feedback to builders
- Document all quality issues found"""

        else:  # RESEARCHER
            prompt_body = """- Search and gather information from documentation
- Explore web resources and APIs
- Research technologies and frameworks
- Compile findings and create knowledge
- Explore edge cases and unknown areas
- Suggest resources and learning materials

Constraints:
- Focus on exploration and learning
- Gather diverse perspectives and sources
- Document findings in knowledge graph
- Request approval for external API calls
- Cite sources for all information"""

        temperature_note = f"""

Execution Mode:
- Temperature: {archetype.temperature} (deterministic if 0.0, creative if higher)
- Max tokens per operation: {archetype.max_tokens}
- Timeout: {archetype.timeout_seconds} seconds
- Concurrent operations: {archetype.max_concurrent_operations}

Safety:
- Follow all safety guidelines strictly
- Request approval for mutations: {archetype.require_approval_for_mutations}
- Report failures and blockers clearly
- Prioritize system stability over speed"""

        return base_prompt + prompt_body + temperature_note

    @staticmethod
    def build_user_guidance(archetype: ArchetypeConfig) -> str:
        """Build user-facing guidance for working with this archetype."""
        guidance = f"""# Working with the {archetype.name} Agent

## When to Use
{AgentSystemPromptBuilder._get_use_cases(archetype.type)}

## Capabilities
- Temperature: {archetype.temperature}
- Max tokens: {archetype.max_tokens}
- Max concurrent operations: {archetype.max_concurrent_operations}
- Timeout: {archetype.timeout_seconds}s

## Tool Access
"""

        # Group tools by category
        by_category = {}
        for tool in archetype.allowed_tools:
            cat = tool.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(tool)

        for category, tools in sorted(by_category.items()):
            guidance += f"\n### {category}\n"
            for tool in tools:
                guidance += f"- `{tool.name}`: {tool.description}\n"

        return guidance

    @staticmethod
    def _get_use_cases(archetype_type: ArchetypeType) -> str:
        """Get use cases for an archetype type."""
        use_cases = {
            ArchetypeType.ARCHITECT: """- Planning new features or systems
- Analyzing code structure and design
- Creating specifications and documentation
- Suggesting architectural improvements
- Reviewing design decisions""",
            ArchetypeType.BUILDER: """- Implementing new features
- Writing and modifying code
- Fixing bugs
- Refactoring code
- Creating test cases""",
            ArchetypeType.QC: """- Validating code quality
- Running test suites
- Checking code coverage
- Scanning for security issues
- Verifying type safety""",
            ArchetypeType.RESEARCHER: """- Researching technologies and frameworks
- Gathering documentation
- Exploring code patterns
- Learning from existing implementations
- Creating knowledge base entries""",
        }
        return use_cases.get(archetype_type, "General purpose agent")


# Export singleton registry
_default_registry: Optional[ArchetypeRegistry] = None


def get_archetype_registry() -> ArchetypeRegistry:
    """Get or create the default archetype registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ArchetypeRegistry()
    return _default_registry


def get_archetype(archetype_type: ArchetypeType) -> Optional[ArchetypeConfig]:
    """Get a specific archetype configuration."""
    return get_archetype_registry().get(archetype_type)


def filter_tools_for_archetype(
    tools: List[Dict[str, Any]],
    archetype_type: ArchetypeType
) -> List[Dict[str, Any]]:
    """Filter MCP tools for a specific archetype."""
    archetype = get_archetype(archetype_type)
    if not archetype:
        return []

    filter_obj = ToolFilter(archetype)
    return filter_obj.filter_tools(tools)
