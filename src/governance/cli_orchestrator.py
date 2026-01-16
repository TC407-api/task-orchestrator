"""CLI agent orchestration with @test/@fix/@review chain support."""
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union

# Context imports (stubbed for skeleton validity if files strictly don't exist yet)
try:
    from src.agents.archetypes import ArchetypeType
    from src.mcp.server import spawn_agent, spawn_parallel_agents
    from src.agents.inbox import UniversalInbox
    ARCHITECT = ArchetypeType.ARCHITECT
    BUILDER = ArchetypeType.BUILDER
    QC = ArchetypeType.QC
    RESEARCHER = ArchetypeType.RESEARCHER
except ImportError:
    # Fallback for isolation testing if environment isn't fully set up
    ArchetypeType = str
    ARCHITECT, BUILDER, QC, RESEARCHER = "Architect", "Builder", "QC", "Researcher"
    spawn_agent, spawn_parallel_agents = None, None
    UniversalInbox = None

# Logger setup
audit_logger = logging.getLogger("audit")


@dataclass
class AgentChainStep:
    """Defines a single unit of work within an orchestration chain."""
    name: str
    archetype: Any  # ArchetypeType
    prompt_template: str
    depends_on: List[str] = field(default_factory=list)
    requires_approval: bool = False
    context_keys: List[str] = field(default_factory=list)


@dataclass
class ChainResult:
    """Encapsulates the final status of a chain execution."""
    steps_completed: List[str]
    outputs: Dict[str, Any]
    total_cost: float
    trace_id: str
    status: str = "completed"


class AgentChain:
    """Orchestrates a sequence of agent interactions based on a DAG of steps."""

    def __init__(self, steps: List[AgentChainStep]):
        """
        Initialize the agent chain.

        Args:
            steps: List of AgentChainStep objects defining the workflow
        """
        self.steps = steps
        self.step_map = {s.name: s for s in steps}
        self.inbox = UniversalInbox() if UniversalInbox else None

    def add_step(self, step: AgentChainStep) -> None:
        """
        Dynamically appends a step to the chain.

        Args:
            step: The AgentChainStep to add
        """
        self.steps.append(step)
        self.step_map[step.name] = step

    def execute(self, input_str: str, context: Dict[str, Any] = None) -> ChainResult:
        """
        Executes the chain.

        Args:
            input_str: The primary instruction or error log.
            context: Additional metadata (repo path, build IDs, etc).

        Returns:
            ChainResult object containing execution artifacts.
        """
        if context is None:
            context = {}

        trace_id = str(uuid.uuid4())
        outputs: Dict[str, Any] = {}
        steps_completed: List[str] = []
        total_cost = 0.0

        # Log chain start
        audit_logger.info("Chain execution started", extra={"trace_id": trace_id, "input": input_str})

        # Build execution context
        exec_context = {"input": input_str, **context}

        # Get steps without dependencies (can run in parallel)
        no_deps = [s for s in self.steps if not s.depends_on]
        with_deps = [s for s in self.steps if s.depends_on]

        # Check if we should run parallel
        if len(no_deps) > 1 and spawn_parallel_agents:
            # Request approval for any that need it
            for step in no_deps:
                if step.requires_approval and self.inbox:
                    self.inbox.request_approval(step.name, exec_context)

            # Run parallel
            prompts = [s.prompt_template.format(**exec_context) for s in no_deps]
            results = spawn_parallel_agents(prompts)

            for i, step in enumerate(no_deps):
                result = results[i] if i < len(results) else {"status": "failure"}
                outputs[step.name] = result.get("output")
                exec_context[f"{step.name}_output"] = result.get("output")
                steps_completed.append(step.name)

        else:
            # Run sequentially
            for step in no_deps + with_deps:
                # Check dependencies
                if step.depends_on:
                    if not all(dep in steps_completed for dep in step.depends_on):
                        continue

                # Check approval if required
                if step.requires_approval and self.inbox:
                    self.inbox.request_approval(step.name, exec_context)

                # Execute step
                prompt = step.prompt_template.format(**exec_context)

                if spawn_agent:
                    result = spawn_agent(prompt)
                else:
                    result = {"status": "success", "output": f"Executed {step.name}"}

                # Check for failure
                if result.get("status") == "failure":
                    steps_completed.append(step.name)
                    outputs[step.name] = result.get("output")
                    break

                # Record output
                outputs[step.name] = result.get("output")
                exec_context[f"{step.name}_output"] = result.get("output")
                steps_completed.append(step.name)
                total_cost += result.get("cost", 0.0)

        return ChainResult(
            steps_completed=steps_completed,
            outputs=outputs,
            total_cost=total_cost,
            trace_id=trace_id,
        )

    def _resolve_dependencies(self) -> List[List[AgentChainStep]]:
        """Internal helper to topological sort steps or group parallels."""
        # Group by dependency level
        levels: List[List[AgentChainStep]] = []
        completed: set = set()

        remaining = list(self.steps)
        while remaining:
            # Find all steps whose dependencies are satisfied
            ready = [s for s in remaining if all(d in completed for d in s.depends_on)]
            if not ready:
                break
            levels.append(ready)
            for s in ready:
                completed.add(s.name)
                remaining.remove(s)

        return levels


# --- Predefined Orchestration Chains ---

TEST_FIX_REVIEW = AgentChain(steps=[
    AgentChainStep(
        name="analyze_failure",
        archetype=RESEARCHER,
        prompt_template="Analyze this build error and identify the root cause:\n{input}"
    ),
    AgentChainStep(
        name="generate_fix",
        archetype=BUILDER,
        prompt_template="Create a patch for the following error:\n{analyze_failure_output}",
        depends_on=["analyze_failure"]
    ),
    AgentChainStep(
        name="review_fix",
        archetype=QC,
        prompt_template="Review this patch for safety and correctness:\n{generate_fix_output}",
        depends_on=["generate_fix"],
        requires_approval=True
    )
])

BUILD_VALIDATE_DEPLOY = AgentChain(steps=[
    AgentChainStep(
        name="scaffold",
        archetype=ARCHITECT,
        prompt_template="Scaffold project structure for: {input}"
    ),
    AgentChainStep(
        name="implement",
        archetype=BUILDER,
        prompt_template="Implement core logic",
        depends_on=["scaffold"]
    ),
    AgentChainStep(
        name="validate",
        archetype=QC,
        prompt_template="Run test suite",
        depends_on=["implement"]
    )
])
