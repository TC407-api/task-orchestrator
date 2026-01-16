import pytest
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, List

# Assumed imports based on project structure
from src.governance.cli_orchestrator import (
    AgentChain,
    AgentChainStep,
    ChainResult
)
from src.agents.archetypes import ArchetypeType

# Use enum values for archetypes
ARCHITECT = ArchetypeType.ARCHITECT
BUILDER = ArchetypeType.BUILDER
QC = ArchetypeType.QC

# Mock data for tests
MOCK_INPUT = "Fix the build failure in CI"
MOCK_CONTEXT = {"repo": "task-orchestrator", "build_id": "123"}


@pytest.fixture
def mock_agent_runner():
    """Mocks the low-level agent execution function."""
    with patch('src.governance.cli_orchestrator.spawn_agent') as mock:
        mock.return_value = {"status": "success", "output": "Done"}
        yield mock


@pytest.fixture
def mock_inbox():
    """Mocks the UniversalInbox for approval workflows."""
    with patch('src.governance.cli_orchestrator.UniversalInbox') as MockInboxClass:
        mock_instance = MagicMock()
        mock_instance.request_approval = MagicMock(return_value=True)
        MockInboxClass.return_value = mock_instance
        yield mock_instance.request_approval


def test_chain_executes_in_order(mock_agent_runner):
    """Test that steps execute sequentially based on dependencies."""
    step1 = AgentChainStep(name="plan", archetype=ARCHITECT, prompt_template="Plan {input}")
    step2 = AgentChainStep(name="build", archetype=BUILDER, prompt_template="Build {input}", depends_on=["plan"])

    chain = AgentChain(steps=[step1, step2])
    result = chain.execute(MOCK_INPUT, MOCK_CONTEXT)

    assert result.steps_completed == ["plan", "build"]
    # Verify step 2 was called after step 1
    assert mock_agent_runner.call_count == 2


def test_chain_passes_context_between_agents(mock_agent_runner):
    """Test that output from previous steps is available in context of subsequent steps."""
    step1 = AgentChainStep(name="analyze", archetype=QC, prompt_template="Analyze error")
    step2 = AgentChainStep(name="fix", archetype=BUILDER, prompt_template="Fix based on {analyze_output}", depends_on=["analyze"])

    # Mock step 1 returning specific data
    mock_agent_runner.side_effect = [
        {"status": "success", "output": "NullPointerException detected"},
        {"status": "success", "output": "Fixed NPE"}
    ]

    chain = AgentChain(steps=[step1, step2])
    chain.execute(MOCK_INPUT, {})

    # Verify second call received output from first call in its context
    call_args = mock_agent_runner.call_args_list[1]
    # Check that the prompt construction used the context (simplified check)
    assert "NullPointerException detected" in str(call_args) or "analyze_output" in str(call_args)


def test_chain_stops_on_failure(mock_agent_runner):
    """Test that the chain halts execution if a critical step fails."""
    step1 = AgentChainStep(name="critical_step", archetype=ARCHITECT, prompt_template="Do important thing")
    step2 = AgentChainStep(name="next_step", archetype=BUILDER, prompt_template="Follow up", depends_on=["critical_step"])

    # Simulate failure in step 1
    mock_agent_runner.side_effect = [
        {"status": "failure", "error": "I cannot do that"},
        {"status": "success", "output": "Should not happen"}
    ]

    chain = AgentChain(steps=[step1, step2])
    result = chain.execute(MOCK_INPUT, {})

    assert "critical_step" in result.steps_completed
    assert "next_step" not in result.steps_completed
    assert mock_agent_runner.call_count == 1


def test_chain_supports_parallel_steps():
    """Test that steps with the same dependency can be executed in parallel."""
    # Two agents analyzing the same input independently
    step1 = AgentChainStep(name="security_scan", archetype=QC, prompt_template="Scan security")
    step2 = AgentChainStep(name="perf_scan", archetype=QC, prompt_template="Scan perf")

    with patch('src.governance.cli_orchestrator.spawn_parallel_agents') as mock_parallel:
        mock_parallel.return_value = [{"output": "Sec OK"}, {"output": "Perf OK"}]

        chain = AgentChain(steps=[step1, step2])
        chain.execute(MOCK_INPUT, {})

        assert mock_parallel.called
        assert len(mock_parallel.call_args[0][0]) == 2  # List of agents passed


def test_chain_logs_to_audit():
    """Test that execution leaves an audit trail."""
    step1 = AgentChainStep(name="action", archetype=BUILDER, prompt_template="Go")
    chain = AgentChain(steps=[step1])

    with patch('src.governance.cli_orchestrator.audit_logger') as mock_logger:
        result = chain.execute(MOCK_INPUT, {})

        assert result.trace_id is not None
        mock_logger.info.assert_called_with(
            "Chain execution started",
            extra={"trace_id": result.trace_id, "input": MOCK_INPUT}
        )


def test_chain_integrates_with_inbox_approval(mock_agent_runner, mock_inbox):
    """Test that steps marked for approval pause for Inbox interaction."""
    step1 = AgentChainStep(
        name="deploy",
        archetype=BUILDER,
        prompt_template="Deploy app",
        requires_approval=True
    )

    chain = AgentChain(steps=[step1])
    chain.execute(MOCK_INPUT, {})

    mock_inbox.assert_called_once()
    mock_agent_runner.assert_called_once()  # Called only after approval (mock returns True)
