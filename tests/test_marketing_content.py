"""
Tests for Marketing Content.

Ensures marketing content has all required sections.
"""

from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
MARKETING_DIR = PROJECT_ROOT / "docs" / "marketing"


class TestMarketingContent:
    """Tests for marketing content validity.

    Note: Marketing docs were moved to a separate location (commit 1dee7eb).
    These tests are skipped when the marketing directory doesn't exist.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_marketing_dir(self):
        """Skip all tests if marketing directory doesn't exist."""
        if not MARKETING_DIR.exists():
            pytest.skip("Marketing docs moved to separate location (commit 1dee7eb)")

    def test_marketing_directory_exists(self):
        """Test that marketing directory exists."""
        assert MARKETING_DIR.exists(), "docs/marketing directory should exist"

    def test_landing_page_has_required_sections(self):
        """Test that landing page has all required sections."""
        landing_page = MARKETING_DIR / "landing-page.md"
        assert landing_page.exists(), "landing-page.md should exist"

        content = landing_page.read_text(encoding="utf-8")

        # Check for required sections
        required = [
            "95%",  # The headline statistic
            "Why AI Agent Projects Fail",  # Problem section
            "Solution",  # Solution section
            "Self-Healing",  # Feature
            "Immune System",  # Feature
            "MCP Tools",  # Feature
            "Cost",  # Feature
            "Federation",  # Feature
            "Compare",  # Comparison
            "Pricing",  # Pricing section
            "Community",  # Free tier
            "Pro",  # Paid tier
            "Enterprise",  # Enterprise tier
        ]

        for section in required:
            assert section.lower() in content.lower(), (
                f"Landing page should contain '{section}'"
            )

    def test_feature_highlights_has_six_features(self):
        """Test that feature highlights has 6 features."""
        features = MARKETING_DIR / "feature-highlights.md"
        assert features.exists(), "feature-highlights.md should exist"

        content = features.read_text(encoding="utf-8")

        # Count feature sections (##)
        feature_count = content.count("## ")
        # Should have at least 6 features plus header
        assert feature_count >= 6, f"Should have 6+ features, found {feature_count}"

    def test_comparison_table_includes_competitors(self):
        """Test that comparison table includes all competitors."""
        comparison = MARKETING_DIR / "comparison-table.md"
        assert comparison.exists(), "comparison-table.md should exist"

        content = comparison.read_text(encoding="utf-8")

        competitors = ["LangGraph", "CrewAI", "AutoGen", "Langfuse"]

        for competitor in competitors:
            assert competitor in content, (
                f"Comparison should mention {competitor}"
            )

    def test_demo_script_has_timing(self):
        """Test that demo script has timing breakdown."""
        demo = MARKETING_DIR / "demo-script.md"
        assert demo.exists(), "demo-script.md should exist"

        content = demo.read_text(encoding="utf-8")

        # Should have time markers
        assert "0:00" in content, "Demo script should have timing markers"
        assert "Visual" in content or "visual" in content, "Demo should describe visuals"
        assert "Narration" in content or "Script" in content, "Demo should have narration"

    def test_all_marketing_files_are_markdown(self):
        """Test that all marketing files are valid markdown."""
        if not MARKETING_DIR.exists():
            pytest.skip("Marketing directory not created yet")

        md_files = list(MARKETING_DIR.glob("*.md"))
        assert len(md_files) >= 4, f"Should have at least 4 md files, found {len(md_files)}"

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            # Basic markdown validation - has headers
            assert "#" in content, f"{md_file.name} should contain markdown headers"
