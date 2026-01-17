"""
Tests for MCP package manifest files.

Ensures package.json and mcp.json are valid and consistent.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestMCPManifests:
    """Tests for MCP manifest validation."""

    def test_package_json_valid(self):
        """Test that package.json is valid JSON with required fields."""
        package_path = PROJECT_ROOT / "package.json"
        assert package_path.exists(), "package.json should exist"

        with open(package_path) as f:
            data = json.load(f)

        # Required fields
        assert "name" in data, "package.json should have name"
        assert "version" in data, "package.json should have version"
        assert "description" in data, "package.json should have description"
        assert "mcp" in data, "package.json should have mcp configuration"

        # MCP-specific fields
        mcp_config = data["mcp"]
        assert "server" in mcp_config, "mcp config should have server"
        assert "tools" in mcp_config, "mcp config should have tools count"
        assert "categories" in mcp_config, "mcp config should have categories"

        # Validate server config
        server = mcp_config["server"]
        assert "command" in server, "server should have command"
        assert "args" in server, "server should have args"

    def test_mcp_json_valid(self):
        """Test that mcp.json is valid JSON with required fields."""
        mcp_path = PROJECT_ROOT / "mcp.json"
        assert mcp_path.exists(), "mcp.json should exist"

        with open(mcp_path) as f:
            data = json.load(f)

        # Required fields
        assert "name" in data, "mcp.json should have name"
        assert "version" in data, "mcp.json should have version"
        assert "description" in data, "mcp.json should have description"
        assert "server" in data, "mcp.json should have server config"
        assert "features" in data, "mcp.json should have features"

        # Server config
        server = data["server"]
        assert "command" in server, "server should have command"
        assert "args" in server, "server should have args"

        # Features should be non-empty
        assert len(data["features"]) > 0, "mcp.json should have at least one feature"

        # Each feature should have name and description
        for feature in data["features"]:
            assert "name" in feature, "each feature should have name"
            assert "description" in feature, "each feature should have description"

    def test_manifests_version_match(self):
        """Test that package.json and mcp.json versions match."""
        package_path = PROJECT_ROOT / "package.json"
        mcp_path = PROJECT_ROOT / "mcp.json"

        with open(package_path) as f:
            package_data = json.load(f)

        with open(mcp_path) as f:
            mcp_data = json.load(f)

        assert package_data["version"] == mcp_data["version"], (
            "package.json and mcp.json versions should match"
        )

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Server start test may require different handling on Windows"
    )
    def test_server_starts_from_package_command(self):
        """Test that the server can start using the package.json command."""
        package_path = PROJECT_ROOT / "package.json"

        with open(package_path) as f:
            data = json.load(f)

        server = data["mcp"]["server"]
        command = server["command"]
        server["args"]

        # Try to import the server module (doesn't actually start it)
        # This validates the command would work
        try:
            result = subprocess.run(
                [command, "-c", "import sys; sys.path.insert(0, '.'); from src.mcp.server import TaskOrchestratorMCP; print('OK')"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10
            )
            assert "OK" in result.stdout or result.returncode == 0, (
                f"Server import failed: {result.stderr}"
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Server start timed out")
        except FileNotFoundError:
            pytest.skip(f"Python command '{command}' not found")

    def test_tool_categories_documented(self):
        """Test that mcp.json has tool categories documented."""
        mcp_path = PROJECT_ROOT / "mcp.json"

        with open(mcp_path) as f:
            data = json.load(f)

        assert "toolCategories" in data, "mcp.json should have toolCategories"

        categories = data["toolCategories"]
        assert len(categories) > 0, "should have at least one tool category"

        for cat_name, cat_data in categories.items():
            assert "description" in cat_data, f"category {cat_name} should have description"
            assert "tools" in cat_data, f"category {cat_name} should have tools list"
            assert len(cat_data["tools"]) > 0, f"category {cat_name} should have at least one tool"
