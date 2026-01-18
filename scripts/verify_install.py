#!/usr/bin/env python3
"""Verify Task Orchestrator installation is working correctly."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_python_version() -> bool:
    """Check Python version >= 3.10."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_dependencies() -> bool:
    """Check required packages are installed."""
    required = [
        "google.generativeai",
        "pydantic",
        "numpy",
        "sklearn",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace(".", "_") if "." in pkg else pkg)
        except ImportError:
            # Try the actual import
            try:
                parts = pkg.split(".")
                if len(parts) > 1:
                    exec(f"import {pkg}")
                else:
                    __import__(pkg)
            except ImportError:
                missing.append(pkg)

    if missing:
        print(f"  Missing: {', '.join(missing)}")
        return False
    print(f"  All {len(required)} required packages installed")
    return True


def check_env_file() -> bool:
    """Check .env.local exists with required keys."""
    env_file = project_root / ".env.local"
    if not env_file.exists():
        print("  .env.local not found - copy from .env.example")
        return False

    content = env_file.read_text()
    has_jwt = "JWT_SECRET_KEY=" in content and "REPLACE" not in content.split("JWT_SECRET_KEY=")[1].split("\n")[0]
    has_llm = any(
        key in content and "REPLACE" not in content.split(key)[1].split("\n")[0]
        for key in ["GOOGLE_API_KEY=", "OPENAI_API_KEY="]
    )

    if not has_jwt:
        print("  JWT_SECRET_KEY not configured in .env.local")
        return False
    if not has_llm:
        print("  No LLM API key configured (need GOOGLE_API_KEY or OPENAI_API_KEY)")
        return False

    print("  .env.local configured correctly")
    return True


def check_mcp_server() -> bool:
    """Check MCP server can be imported."""
    try:
        from src.mcp.server import TaskOrchestratorServer
        print("  MCP server module loads successfully")
        return True
    except ImportError as e:
        print(f"  MCP server import failed: {e}")
        return False


def check_immune_system() -> bool:
    """Check immune system can be initialized."""
    try:
        from src.evaluation.immune_system import ImmuneSystem
        immune = ImmuneSystem()
        print("  Immune system initializes successfully")
        return True
    except Exception as e:
        print(f"  Immune system failed: {e}")
        return False


def check_llm_providers() -> bool:
    """Check LLM providers can be imported."""
    try:
        from src.llm import GeminiProvider, OpenAIProvider, ModelRouter
        print("  LLM providers available (Gemini, OpenAI)")
        return True
    except ImportError as e:
        print(f"  LLM provider import failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 50)
    print("Task Orchestrator Installation Verification")
    print("=" * 50 + "\n")

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("MCP Server", check_mcp_server),
        ("Immune System", check_immune_system),
        ("LLM Providers", check_llm_providers),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\nTask Orchestrator is ready to use!")
        print("Add to Claude Code with: claude mcp add task-orchestrator python mcp_server.py")
        return 0
    else:
        print("\nSome checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
