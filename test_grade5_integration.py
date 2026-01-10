"""
Grade 5 Integration Test Suite

Tests all Grade 5 components working together:
1. Cross-Project Learning (pattern store)
2. Self-Healing (circuit breakers)
3. Observability (Langfuse tracing)
4. Prompt Optimization (DSPy feedback)
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path.home() / ".claude" / "grade5" / "cross-project"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test results tracker
results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "tests": []
}


def log_result(name: str, passed: bool, details: str = ""):
    """Log a test result."""
    status = "PASS" if passed else "FAIL"
    results["passed" if passed else "failed"] += 1
    results["tests"].append({
        "name": name,
        "passed": passed,
        "details": details
    })
    print(f"  [{status}] {name}")
    if details and not passed:
        print(f"         {details}")


def test_pattern_store():
    """Test 1: Pattern Store loads and queries correctly."""
    print("\n=== Test 1: Pattern Store ===")
    try:
        from pattern_store import PatternStore
        store = PatternStore()
        stats = store.get_stats()

        # Check patterns loaded
        has_patterns = stats["total_patterns"] > 0
        log_result("Patterns loaded", has_patterns,
                   f"Found {stats['total_patterns']} patterns")

        # Check query works
        patterns = store.query(group_ids=["global"], limit=5)
        can_query = len(patterns) > 0
        log_result("Query works", can_query,
                   f"Retrieved {len(patterns)} patterns")

        # Check validity scores
        has_valid_scores = all(0 <= p.validity_score <= 1 for p in patterns)
        log_result("Validity scores calculated", has_valid_scores)

        return has_patterns and can_query and has_valid_scores
    except Exception as e:
        log_result("Pattern store test", False, str(e))
        return False


async def test_cross_project_learning():
    """Test 2: CrossProjectLearning module works."""
    print("\n=== Test 2: Cross-Project Learning ===")
    try:
        from cross_project import CrossProjectLearning

        learning = CrossProjectLearning("task-orchestrator")
        stats = learning.get_stats()

        # Check initialization
        has_group_id = stats["group_id"] == "project_task_orchestrator"
        log_result("Group ID correct", has_group_id, stats["group_id"])

        # Check pattern query
        patterns = await learning.query_patterns("error", limit=3)
        can_query = len(patterns) > 0
        log_result("Can query patterns", can_query,
                   f"Found {len(patterns)} patterns")

        return has_group_id and can_query
    except Exception as e:
        log_result("CrossProjectLearning test", False, str(e))
        return False


def test_self_healing():
    """Test 3: Self-healing circuit breakers work."""
    print("\n=== Test 3: Self-Healing (Circuit Breakers) ===")
    try:
        from self_healing import CircuitBreaker, CircuitState, get_healing_status

        # Test circuit breaker creation
        breaker = CircuitBreaker.get("test_service")
        initial_state = breaker._state == CircuitState.CLOSED
        log_result("Circuit breaker starts CLOSED", initial_state)

        # Test recording success
        breaker.record_success()
        success_count = breaker._failure_count == 0
        log_result("Success resets failures", success_count)

        # Test availability check
        can_proceed, _ = breaker.is_available()
        log_result("Available when CLOSED", can_proceed)

        # Test failure recording
        for i in range(3):
            breaker.record_failure(Exception("test error"))
        is_open = breaker._state == CircuitState.OPEN
        log_result("Opens after 3 failures", is_open)

        # Test healing status
        status = get_healing_status()
        has_status = "circuit_breakers" in status
        log_result("Healing status available", has_status)

        # Clean up test breaker
        breaker._transition_to(CircuitState.CLOSED)

        return initial_state and can_proceed and is_open and has_status
    except Exception as e:
        log_result("Self-healing test", False, str(e))
        return False


def test_observability():
    """Test 4: Observability tracing setup."""
    print("\n=== Test 4: Observability ===")
    try:
        from observability import get_tracer, Tracer

        tracer = get_tracer()

        # Check tracer exists
        is_tracer = isinstance(tracer, Tracer)
        log_result("Tracer initialized", is_tracer)

        # Test span creation (mock since Langfuse may not be running)
        span = tracer.start_trace("test_trace")
        has_span = span is not None
        log_result("Can create spans", has_span)

        # Test span end
        span.end()
        log_result("Can end spans", True)

        return is_tracer and has_span
    except Exception as e:
        log_result("Observability test", False, str(e))
        return False


def test_dspy_feedback():
    """Test 5: DSPy feedback data exists."""
    print("\n=== Test 5: DSPy Feedback ===")
    try:
        dspy_dir = Path.home() / ".claude" / "grade5" / "dspy-data"

        # Check directory exists
        dir_exists = dspy_dir.exists()
        log_result("DSPy data dir exists", dir_exists)

        # Check examples file
        examples_file = dspy_dir / "examples.jsonl"
        has_examples = examples_file.exists()
        log_result("Examples file exists", has_examples)

        if has_examples:
            # Count examples
            with open(examples_file) as f:
                count = sum(1 for _ in f)
            log_result(f"Has {count} training examples", count > 0)

        # Check stats file
        stats_file = dspy_dir / "stats.json"
        has_stats = stats_file.exists()
        log_result("Stats file exists", has_stats)

        return dir_exists and has_examples
    except Exception as e:
        log_result("DSPy feedback test", False, str(e))
        return False


def test_mcp_server_handlers():
    """Test 6: MCP server handlers have decorators."""
    print("\n=== Test 6: MCP Server Integration ===")
    try:
        server_file = Path(__file__).parent / "src" / "mcp" / "server.py"

        with open(server_file) as f:
            content = f.read()

        # Check trace_operation decorators
        has_trace_decorators = content.count("@trace_operation") >= 5
        log_result("Has trace_operation decorators", has_trace_decorators,
                   f"Found {content.count('@trace_operation')} decorators")

        # Check circuit breaker imports
        has_cb_import = "CircuitBreaker" in content
        log_result("Has CircuitBreaker import", has_cb_import)

        # Check healing_status tool
        has_healing_tool = "healing_status" in content
        log_result("Has healing_status tool", has_healing_tool)

        return has_trace_decorators and has_cb_import
    except Exception as e:
        log_result("MCP server test", False, str(e))
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("GRADE 5 INTEGRATION TEST SUMMARY")
    print("=" * 60)

    total = results["passed"] + results["failed"]
    pct = (results["passed"] / total * 100) if total > 0 else 0

    print(f"\nPassed: {results['passed']}/{total} ({pct:.1f}%)")
    print(f"Failed: {results['failed']}")

    if results["failed"] > 0:
        print("\nFailed Tests:")
        for test in results["tests"]:
            if not test["passed"]:
                print(f"  - {test['name']}: {test['details']}")

    print("\n" + "=" * 60)
    status = "ALL TESTS PASSED" if results["failed"] == 0 else "SOME TESTS FAILED"
    print(f"STATUS: {status}")
    print("=" * 60)

    return results["failed"] == 0


async def main():
    """Run all integration tests."""
    print("=" * 60)
    print("GRADE 5 INTEGRATION TEST SUITE")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Run tests
    test_pattern_store()
    await test_cross_project_learning()
    test_self_healing()
    test_observability()
    test_dspy_feedback()
    test_mcp_server_handlers()

    # Summary
    success = print_summary()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
