"""
Tests for Starter Templates and Template Generator.

TDD RED Phase: These tests define the expected behavior.
The implementation should make them pass.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Try imports - will fail until implementation exists
try:
    from src.cli.template_generator import (
        TemplateGenerator,
        get_template_generator,
    )
except ImportError:
    TemplateGenerator = None
    get_template_generator = None


class TestTemplateGenerator:
    """Tests for TemplateGenerator functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def generator(self):
        """Create a TemplateGenerator instance."""
        if TemplateGenerator is None:
            pytest.skip("TemplateGenerator not implemented yet")
        return TemplateGenerator()

    def test_list_templates_returns_all(self, generator):
        """Test that list_templates returns all available templates."""
        templates = generator.list_templates()

        # Should have at least 5 templates as per the plan
        assert len(templates) >= 5

        # Check for expected templates
        expected = [
            "basic-agent",
            "multi-agent-workflow",
            "cost-controlled",
            "self-healing",
            "federation",
        ]
        for name in expected:
            assert name in templates, f"Template '{name}' should be available"

    def test_generate_basic_agent_creates_files(self, generator, temp_dir):
        """Test that generating basic-agent template creates expected files."""
        output_dir = temp_dir / "test-project"

        generator.generate("basic-agent", output_dir)

        # Check directory was created
        assert output_dir.exists(), "Output directory should exist"

        # Check for expected files
        assert (output_dir / "main.py").exists(), "main.py should exist"
        assert (output_dir / "README.md").exists(), "README.md should exist"
        assert (output_dir / "requirements.txt").exists(), "requirements.txt should exist"

    def test_generate_multi_agent_workflow_creates_files(self, generator, temp_dir):
        """Test generating multi-agent-workflow template."""
        output_dir = temp_dir / "workflow-project"

        generator.generate("multi-agent-workflow", output_dir)

        assert output_dir.exists()
        assert (output_dir / "workflow.py").exists() or (output_dir / "main.py").exists()
        assert (output_dir / "README.md").exists()

    def test_generate_cost_controlled_creates_files(self, generator, temp_dir):
        """Test generating cost-controlled template."""
        output_dir = temp_dir / "budget-project"

        generator.generate("cost-controlled", output_dir)

        assert output_dir.exists()
        # Check for budget/cost related file
        files = list(output_dir.glob("*.py"))
        assert len(files) > 0, "Should have at least one Python file"

    def test_generate_self_healing_creates_files(self, generator, temp_dir):
        """Test generating self-healing template."""
        output_dir = temp_dir / "resilient-project"

        generator.generate("self-healing", output_dir)

        assert output_dir.exists()
        files = list(output_dir.glob("*.py"))
        assert len(files) > 0

    def test_generate_federation_creates_files(self, generator, temp_dir):
        """Test generating federation template."""
        output_dir = temp_dir / "federated-project"

        generator.generate("federation", output_dir)

        assert output_dir.exists()
        files = list(output_dir.glob("*.py"))
        assert len(files) > 0

    def test_validate_template_returns_true_for_valid(self, generator):
        """Test that validate_template returns True for valid templates."""
        valid_templates = generator.list_templates()

        for template_name in valid_templates[:2]:  # Test first 2 for speed
            assert generator.validate_template(template_name), (
                f"Template '{template_name}' should be valid"
            )

    def test_validate_template_returns_false_for_invalid(self, generator):
        """Test that validate_template returns False for invalid templates."""
        result = generator.validate_template("nonexistent-template")
        assert result is False, "Invalid template should return False"

    def test_generate_raises_for_invalid_template(self, generator, temp_dir):
        """Test that generate raises for invalid template name."""
        with pytest.raises((ValueError, KeyError)):
            generator.generate("invalid-template-name", temp_dir / "output")

    def test_generate_does_not_overwrite_existing(self, generator, temp_dir):
        """Test that generate doesn't overwrite existing directory by default."""
        output_dir = temp_dir / "existing-project"
        output_dir.mkdir(parents=True)
        (output_dir / "important.txt").write_text("don't delete me")

        # Should raise or skip when directory exists
        with pytest.raises((FileExistsError, ValueError)):
            generator.generate("basic-agent", output_dir)

        # Original file should still exist
        assert (output_dir / "important.txt").exists()

    def test_generate_with_force_overwrites(self, generator, temp_dir):
        """Test that generate with force=True overwrites existing."""
        output_dir = temp_dir / "existing-project"
        output_dir.mkdir(parents=True)
        (output_dir / "old_file.txt").write_text("old content")

        generator.generate("basic-agent", output_dir, force=True)

        # Template files should exist
        assert (output_dir / "main.py").exists()


class TestTemplateDirectory:
    """Tests for the templates directory structure."""

    def test_templates_directory_exists(self):
        """Test that templates directory exists."""
        templates_dir = PROJECT_ROOT / "templates"
        assert templates_dir.exists(), "templates directory should exist"
        assert templates_dir.is_dir(), "templates should be a directory"

    def test_basic_agent_template_exists(self):
        """Test that basic-agent template exists."""
        template_dir = PROJECT_ROOT / "templates" / "basic-agent"
        assert template_dir.exists(), "basic-agent template should exist"

        # Check for main file
        assert (template_dir / "main.py").exists(), "main.py should exist"
        assert (template_dir / "README.md").exists(), "README.md should exist"

    def test_multi_agent_workflow_template_exists(self):
        """Test that multi-agent-workflow template exists."""
        template_dir = PROJECT_ROOT / "templates" / "multi-agent-workflow"
        assert template_dir.exists(), "multi-agent-workflow template should exist"

    def test_cost_controlled_template_exists(self):
        """Test that cost-controlled template exists."""
        template_dir = PROJECT_ROOT / "templates" / "cost-controlled"
        assert template_dir.exists(), "cost-controlled template should exist"

    def test_self_healing_template_exists(self):
        """Test that self-healing template exists."""
        template_dir = PROJECT_ROOT / "templates" / "self-healing"
        assert template_dir.exists(), "self-healing template should exist"

    def test_federation_template_exists(self):
        """Test that federation template exists."""
        template_dir = PROJECT_ROOT / "templates" / "federation"
        assert template_dir.exists(), "federation template should exist"


class TestTemplateContent:
    """Tests for template content validity."""

    def test_each_template_has_readme(self):
        """Test that each template has a README.md."""
        templates_dir = PROJECT_ROOT / "templates"
        if not templates_dir.exists():
            pytest.skip("templates directory not created yet")

        for template_dir in templates_dir.iterdir():
            if template_dir.is_dir() and not template_dir.name.startswith("."):
                readme = template_dir / "README.md"
                assert readme.exists(), f"Template {template_dir.name} should have README.md"

    def test_each_template_has_requirements(self):
        """Test that each template has requirements.txt."""
        templates_dir = PROJECT_ROOT / "templates"
        if not templates_dir.exists():
            pytest.skip("templates directory not created yet")

        for template_dir in templates_dir.iterdir():
            if template_dir.is_dir() and not template_dir.name.startswith("."):
                req = template_dir / "requirements.txt"
                assert req.exists(), f"Template {template_dir.name} should have requirements.txt"

    def test_basic_agent_template_is_syntactically_valid(self):
        """Test that basic-agent main.py is valid Python."""
        main_py = PROJECT_ROOT / "templates" / "basic-agent" / "main.py"
        if not main_py.exists():
            pytest.skip("basic-agent template not created yet")

        content = main_py.read_text()
        # This will raise SyntaxError if invalid
        compile(content, str(main_py), "exec")


# Standalone tests (run even if imports fail)
def test_template_generator_module_exists():
    """Test that template_generator module can be imported."""
    try:
        from src.cli import template_generator
        assert template_generator is not None
    except ImportError as e:
        pytest.fail(f"template_generator module not found: {e}")


def test_template_generator_class_exists():
    """Test that TemplateGenerator class exists."""
    try:
        from src.cli.template_generator import TemplateGenerator
        assert TemplateGenerator is not None
    except ImportError as e:
        pytest.fail(f"TemplateGenerator class not found: {e}")


def test_get_template_generator_exists():
    """Test that get_template_generator factory function exists."""
    try:
        from src.cli.template_generator import get_template_generator
        assert callable(get_template_generator)
    except ImportError as e:
        pytest.fail(f"get_template_generator function not found: {e}")
