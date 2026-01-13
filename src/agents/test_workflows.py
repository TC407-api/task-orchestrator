"""Tests for the Global @Workflow trigger system."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src/agents to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .workflows import (
    WorkflowRegistry,
    WorkflowTrigger,
    WorkflowExecutor,
    ContextManifest,
    WorkflowType,
    get_workflow_registry,
    get_workflow_trigger,
    process_prompt_with_workflows,
)


class TestContextManifest:
    """Tests for ContextManifest dataclass."""

    def test_create_manifest(self):
        """Test creating a manifest."""
        manifest = ContextManifest(
            trigger="@TestTrigger",
            files_pattern=["src/**/*.py"],
            system_prompt_addition="Test prompt",
            archetype="builder",
        )
        assert manifest.trigger == "@TestTrigger"
        assert manifest.files_pattern == ["src/**/*.py"]
        assert manifest.system_prompt_addition == "Test prompt"
        assert manifest.archetype == "builder"

    def test_manifest_to_dict(self):
        """Test serializing manifest to dict."""
        manifest = ContextManifest(
            trigger="@Test",
            files_pattern=["src/**/*.py"],
            system_prompt_addition="Prompt",
            archetype="qc",
            temperature=0.5,
            priority=10,
        )
        result = manifest.to_dict()
        assert result["trigger"] == "@Test"
        assert result["archetype"] == "qc"
        assert result["temperature"] == 0.5
        assert result["priority"] == 10

    def test_manifest_defaults(self):
        """Test manifest default values."""
        manifest = ContextManifest(trigger="@Default")
        assert manifest.files_pattern == []
        assert manifest.system_prompt_addition == ""
        assert manifest.archetype == "builder"
        assert manifest.max_context_tokens == 4000
        assert manifest.temperature is None
        assert manifest.priority == 0


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry."""

    def test_registry_initialization(self):
        """Test registry initializes with built-in workflows."""
        registry = WorkflowRegistry()
        assert registry.get("@Refactor") is not None
        assert registry.get("@TestGen") is not None
        assert registry.get("@Debug") is not None
        assert registry.get("@Review") is not None
        assert registry.get("@Docs") is not None

    def test_registry_get_workflow(self):
        """Test retrieving workflow from registry."""
        registry = WorkflowRegistry()
        refactor = registry.get("@Refactor")
        assert refactor is not None
        assert refactor.trigger == "@Refactor"
        assert refactor.archetype == "builder"

    def test_registry_get_unknown_workflow(self):
        """Test retrieving unknown workflow returns None."""
        registry = WorkflowRegistry()
        result = registry.get("@Unknown")
        assert result is None

    def test_registry_register_custom(self):
        """Test registering custom workflow."""
        registry = WorkflowRegistry()
        custom = ContextManifest(
            trigger="@Custom",
            system_prompt_addition="Custom prompt",
            archetype="researcher",
        )
        registry.register(custom)
        assert registry.get("@Custom") is not None
        assert registry.get("@Custom").archetype == "researcher"

    def test_registry_list_workflows(self):
        """Test listing all workflows."""
        registry = WorkflowRegistry()
        workflows = registry.list_workflows()
        assert len(workflows) == 5  # Built-in workflows
        triggers = {w.trigger for w in workflows}
        assert "@Refactor" in triggers

    def test_registry_get_all_triggers(self):
        """Test getting all trigger strings."""
        registry = WorkflowRegistry()
        triggers = registry.get_all_triggers()
        assert isinstance(triggers, set)
        assert "@Refactor" in triggers
        assert "@TestGen" in triggers

    def test_registry_to_dict(self):
        """Test serializing registry to dict."""
        registry = WorkflowRegistry()
        result = registry.to_dict()
        assert isinstance(result, dict)
        assert "@Refactor" in result
        assert isinstance(result["@Refactor"], dict)


class TestWorkflowTrigger:
    """Tests for WorkflowTrigger detection and processing."""

    def test_detect_single_trigger(self):
        """Test detecting single trigger in prompt."""
        trigger = WorkflowTrigger()
        prompt = "Please refactor this code. @Refactor"
        detected = trigger.detect_triggers(prompt)
        assert "@Refactor" in detected

    def test_detect_multiple_triggers(self):
        """Test detecting multiple triggers in prompt."""
        trigger = WorkflowTrigger()
        prompt = "@Refactor this code and @TestGen for it"
        detected = trigger.detect_triggers(prompt)
        assert "@Refactor" in detected
        assert "@TestGen" in detected

    def test_detect_no_triggers(self):
        """Test detecting no triggers returns empty set."""
        trigger = WorkflowTrigger()
        prompt = "Please refactor this code"
        detected = trigger.detect_triggers(prompt)
        assert len(detected) == 0

    def test_detect_triggers_case_sensitive(self):
        """Test trigger detection is case-sensitive."""
        trigger = WorkflowTrigger()
        prompt = "@refactor @REFACTOR @Refactor"
        detected = trigger.detect_triggers(prompt)
        # Should detect different cases as separate triggers
        assert "@Refactor" in detected

    def test_has_trigger(self):
        """Test has_trigger method."""
        trigger = WorkflowTrigger()
        assert trigger.has_trigger("@Refactor code") is True
        assert trigger.has_trigger("refactor code") is False

    def test_extract_trigger_context_unknown(self):
        """Test extracting context for unknown trigger."""
        trigger = WorkflowTrigger()
        result = trigger.extract_trigger_context("@Unknown")
        assert result["files"] == []
        assert result["manifest"] is None
        assert "error" in result

    def test_should_skip_file_pycache(self):
        """Test skipping __pycache__ files."""
        trigger = WorkflowTrigger()
        path = Path("src/__pycache__/test.pyc")
        assert trigger._should_skip_file(path) is True

    def test_should_skip_file_git(self):
        """Test skipping .git directory files."""
        trigger = WorkflowTrigger()
        path = Path("src/.git/config")
        assert trigger._should_skip_file(path) is True

    def test_should_skip_file_node_modules(self):
        """Test skipping node_modules files."""
        trigger = WorkflowTrigger()
        path = Path("node_modules/package/index.js")
        assert trigger._should_skip_file(path) is True

    @patch("workflows.Path.stat")
    def test_should_skip_large_file(self, mock_stat):
        """Test skipping large files."""
        trigger = WorkflowTrigger()
        mock_stat.return_value = MagicMock(st_size=2_000_000)  # 2MB
        path = Path("src/large.bin")
        assert trigger._should_skip_file(path) is True

    @patch("workflows.Path.stat")
    def test_should_not_skip_normal_file(self, mock_stat):
        """Test not skipping normal files."""
        trigger = WorkflowTrigger()
        mock_stat.return_value = MagicMock(st_size=1000)
        path = Path("src/normal.py")
        assert trigger._should_skip_file(path) is False

    @patch("workflows.glob_module.glob")
    def test_extract_trigger_context_with_files(self, mock_glob):
        """Test extracting context with mock files."""
        trigger = WorkflowTrigger()
        mock_glob.return_value = []  # No files found
        result = trigger.extract_trigger_context("@Refactor")
        assert result["files"] == []
        assert result["manifest"] is not None
        assert result["manifest"].trigger == "@Refactor"

    def test_process_prompt_no_triggers(self):
        """Test processing prompt with no triggers."""
        trigger = WorkflowTrigger()
        prompt = "Please refactor this code"
        result = trigger.process_prompt(prompt)
        assert result["original_prompt"] == prompt
        assert result["processed_prompt"] == prompt
        assert result["triggers"] == []
        assert result["context_injected"] is False

    def test_process_prompt_with_trigger(self):
        """Test processing prompt with trigger."""
        trigger = WorkflowTrigger()
        prompt = "Please @Refactor this code"
        result = trigger.process_prompt(prompt)
        assert result["original_prompt"] == prompt
        assert result["context_injected"] is True
        assert "@Refactor" in result["triggers"]
        assert "@Refactor" in result["processed_prompt"]
        assert "WORKFLOW:" in result["processed_prompt"]

    def test_process_prompt_multiple_triggers(self):
        """Test processing prompt with multiple triggers."""
        trigger = WorkflowTrigger()
        prompt = "@Refactor this code and @TestGen for it"
        result = trigger.process_prompt(prompt)
        assert len(result["triggers"]) == 2
        assert "@Refactor" in result["triggers"]
        assert "@TestGen" in result["triggers"]


class TestWorkflowExecutor:
    """Tests for WorkflowExecutor."""

    def test_executor_initialization(self):
        """Test executor initialization."""
        executor = WorkflowExecutor()
        assert executor.registry is not None
        assert executor.trigger is not None
        assert executor.base_path == "."

    def test_executor_prepare_context_no_triggers(self):
        """Test preparing context with no triggers."""
        executor = WorkflowExecutor()
        prompt = "Please refactor this code"
        result = executor.prepare_context(prompt)
        assert result["prompt"] == prompt
        assert result["archetype"] == "builder"
        assert result["system_prompt_addition"] == ""
        assert result["temperature"] is None

    def test_executor_prepare_context_with_trigger(self):
        """Test preparing context with trigger."""
        executor = WorkflowExecutor()
        prompt = "@Refactor this code"
        result = executor.prepare_context(prompt)
        assert result["archetype"] == "builder"
        assert result["system_prompt_addition"] != ""
        assert "refactor" in result["system_prompt_addition"].lower()

    def test_executor_prepare_context_with_archetype_override(self):
        """Test preparing context with archetype override."""
        executor = WorkflowExecutor()
        prompt = "@Review this code"
        result = executor.prepare_context(prompt, archetype="architect")
        assert result["archetype"] == "architect"

    def test_executor_prepare_context_multiple_triggers(self):
        """Test preparing context with multiple triggers."""
        executor = WorkflowExecutor()
        prompt = "@Refactor and @TestGen"
        result = executor.prepare_context(prompt)
        assert len(result["triggers"]) == 2
        # System prompt should contain additions from both
        assert "refactor" in result["system_prompt_addition"].lower()
        assert "test" in result["system_prompt_addition"].lower()

    def test_executor_temperature_override(self):
        """Test temperature override from trigger."""
        executor = WorkflowExecutor()
        prompt = "@Debug this issue"
        result = executor.prepare_context(prompt)
        # Debug workflow has temperature 0.2
        assert result["temperature"] == 0.2


class TestGlobalFunctions:
    """Tests for global helper functions."""

    def test_get_workflow_registry_singleton(self):
        """Test get_workflow_registry returns singleton."""
        reg1 = get_workflow_registry()
        reg2 = get_workflow_registry()
        assert reg1 is reg2

    def test_get_workflow_trigger_singleton(self):
        """Test get_workflow_trigger returns singleton."""
        trig1 = get_workflow_trigger()
        trig2 = get_workflow_trigger()
        assert trig1 is trig2

    def test_process_prompt_with_workflows(self):
        """Test main entry point for workflow processing."""
        prompt = "@Refactor this code"
        result = process_prompt_with_workflows(prompt)
        assert result["prompt"] != ""
        assert "@Refactor" in result["triggers"]
        assert result["archetype"] in ["builder", "architect", "qc", "researcher"]


class TestBuiltInWorkflows:
    """Tests for built-in workflow definitions."""

    def test_refactor_workflow(self):
        """Test @Refactor workflow configuration."""
        registry = WorkflowRegistry()
        manifest = registry.get("@Refactor")
        assert manifest is not None
        assert manifest.archetype == "builder"
        assert manifest.temperature == 0.3
        assert len(manifest.files_pattern) > 0
        assert "refactor" in manifest.system_prompt_addition.lower()

    def test_testgen_workflow(self):
        """Test @TestGen workflow configuration."""
        registry = WorkflowRegistry()
        manifest = registry.get("@TestGen")
        assert manifest is not None
        assert manifest.archetype == "builder"
        assert manifest.temperature == 0.5
        assert "test" in manifest.system_prompt_addition.lower()

    def test_debug_workflow(self):
        """Test @Debug workflow configuration."""
        registry = WorkflowRegistry()
        manifest = registry.get("@Debug")
        assert manifest is not None
        assert manifest.archetype == "builder"
        assert manifest.temperature == 0.2
        assert "debug" in manifest.system_prompt_addition.lower()

    def test_review_workflow(self):
        """Test @Review workflow configuration."""
        registry = WorkflowRegistry()
        manifest = registry.get("@Review")
        assert manifest is not None
        assert manifest.archetype == "architect"
        assert manifest.temperature == 0.1
        assert "review" in manifest.system_prompt_addition.lower()

    def test_docs_workflow(self):
        """Test @Docs workflow configuration."""
        registry = WorkflowRegistry()
        manifest = registry.get("@Docs")
        assert manifest is not None
        assert manifest.archetype == "researcher"
        assert manifest.temperature == 0.4
        assert "doc" in manifest.system_prompt_addition.lower()


class TestIntegration:
    """Integration tests for workflow system."""

    def test_full_workflow_pipeline(self):
        """Test full workflow from trigger detection to context preparation."""
        prompt = """
        I need to refactor the authentication module and generate tests for it.
        @Refactor the code structure
        @TestGen for all functions
        """

        result = process_prompt_with_workflows(prompt)

        assert len(result["triggers"]) == 2
        assert "@Refactor" in result["triggers"]
        assert "@TestGen" in result["triggers"]
        assert result["archetype"] == "builder"
        assert result["system_prompt_addition"] != ""

    def test_workflow_with_archetype_routing(self):
        """Test workflow routes to correct archetype."""
        executor = WorkflowExecutor()

        # @Review should suggest architect
        review_result = executor.prepare_context("@Review this design")
        assert review_result["archetype"] == "architect"

        # @Refactor should suggest builder
        refactor_result = executor.prepare_context("@Refactor this code")
        assert refactor_result["archetype"] == "builder"

    def test_workflow_prompt_enrichment(self):
        """Test that workflow enriches the prompt with context."""
        executor = WorkflowExecutor()
        original = "@Refactor this code"
        result = executor.prepare_context(original)

        # Prompt should be enhanced
        assert result["prompt"] != original
        assert original in result["prompt"]
        assert "WORKFLOW:" in result["prompt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
