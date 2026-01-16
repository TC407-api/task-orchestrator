"""
Template Generator for Starter Projects.

Provides functionality to list, validate, and generate projects
from starter templates.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Root directory of templates
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

# Available templates
AVAILABLE_TEMPLATES = [
    "basic-agent",
    "multi-agent-workflow",
    "cost-controlled",
    "self-healing",
    "federation",
]


class TemplateGenerator:
    """
    Generate starter projects from templates.

    Provides listing, validation, and generation of projects
    from pre-defined starter templates.
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize the template generator.

        Args:
            templates_dir: Optional custom templates directory.
                          Defaults to the built-in templates.
        """
        self.templates_dir = templates_dir or TEMPLATES_DIR

    def list_templates(self) -> List[str]:
        """
        List all available templates.

        Returns:
            List of template names
        """
        templates = []

        if self.templates_dir.exists():
            for item in self.templates_dir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    templates.append(item.name)

        # Ensure all expected templates are listed
        for name in AVAILABLE_TEMPLATES:
            if name not in templates:
                templates.append(name)

        return sorted(set(templates))

    def validate_template(self, template_name: str) -> bool:
        """
        Check if a template is valid and exists.

        Args:
            template_name: Name of the template to validate

        Returns:
            True if valid, False otherwise
        """
        if template_name not in AVAILABLE_TEMPLATES:
            return False

        template_path = self.templates_dir / template_name
        if not template_path.exists():
            return False

        # Check for essential files
        main_file = template_path / "main.py"
        if not main_file.exists():
            # Check for alternative entry point
            alt_files = list(template_path.glob("*.py"))
            if not alt_files:
                return False

        return True

    def generate(
        self,
        template_name: str,
        output_dir: Path,
        force: bool = False,
    ) -> None:
        """
        Generate a project from a template.

        Args:
            template_name: Name of the template to use
            output_dir: Directory to create the project in
            force: If True, overwrite existing directory

        Raises:
            ValueError: If template doesn't exist
            FileExistsError: If output_dir exists and force=False
        """
        if template_name not in AVAILABLE_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        template_path = self.templates_dir / template_name

        if not template_path.exists():
            raise ValueError(f"Template directory not found: {template_path}")

        # Check for existing output
        if output_dir.exists():
            if not force:
                raise FileExistsError(f"Output directory already exists: {output_dir}")
            shutil.rmtree(output_dir)

        # Copy template to output
        shutil.copytree(template_path, output_dir)

        logger.info(f"Generated project from '{template_name}' at {output_dir}")

    def get_template_info(self, template_name: str) -> dict:
        """
        Get information about a template.

        Args:
            template_name: Name of the template

        Returns:
            Dict with template information
        """
        if template_name not in AVAILABLE_TEMPLATES:
            return {"error": f"Unknown template: {template_name}"}

        template_path = self.templates_dir / template_name

        if not template_path.exists():
            return {"name": template_name, "exists": False}

        readme_path = template_path / "README.md"
        description = ""
        if readme_path.exists():
            # Get first paragraph from README
            content = readme_path.read_text()
            lines = content.split("\n")
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    description = line.strip()
                    break

        files = [f.name for f in template_path.iterdir() if f.is_file()]

        return {
            "name": template_name,
            "exists": True,
            "description": description,
            "files": files,
            "path": str(template_path),
        }


# Global instance
_generator_instance: Optional[TemplateGenerator] = None


def get_template_generator() -> TemplateGenerator:
    """Get or create the global TemplateGenerator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = TemplateGenerator()
    return _generator_instance


__all__ = [
    "TemplateGenerator",
    "get_template_generator",
    "AVAILABLE_TEMPLATES",
    "TEMPLATES_DIR",
]
