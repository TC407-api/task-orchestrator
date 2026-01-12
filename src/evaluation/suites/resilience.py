"""
Resilience Evaluation Suite for task-orchestrator.

This module provides fault injection testing to verify that the system
handles failures gracefully, respects circuit breaker logic, and enforces
budget constraints.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from ...self_healing import CircuitBreaker, CircuitState, CircuitBreakerConfig
from ...core.cost_tracker import CostTracker, Provider

logger = logging.getLogger(__name__)


@dataclass
class ResilienceTestResult:
    """Result of a single resilience test."""
    test_name: str
    passed: bool
    details: str
    duration_ms: float


class ResilienceEvalSuite:
    """
    Evaluation suite for fault injection and resilience testing.

    This suite verifies that the task orchestrator handles failures gracefully,
    including circuit breaker behavior, budget enforcement, and error messages.

    Attributes:
        results (List[ResilienceTestResult]): Collected test results.
    """

    def __init__(self):
        """Initialize the resilience suite."""
        self.results: List[ResilienceTestResult] = []

    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all defined resilience tests.

        Returns:
            Dict containing summary of results and individual test details.
        """
        logger.info("Starting Resilience Evaluation Suite...")
        self.results = []

        tests = [
            self.test_circuit_breaker_trips_on_failures,
            self.test_circuit_breaker_recovers,
            self.test_semantic_failure_tracking,
            self.test_budget_exceeded_blocks,
            self.test_graceful_degradation_message,
        ]

        for test_func in tests:
            start_time = time.perf_counter()
            try:
                success, message = await test_func()
                result = ResilienceTestResult(
                    test_name=test_func.__name__,
                    passed=success,
                    details=message,
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )
            except Exception as e:
                logger.error(f"Test {test_func.__name__} failed: {e}")
                result = ResilienceTestResult(
                    test_name=test_func.__name__,
                    passed=False,
                    details=f"Exception: {str(e)}",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )

            self.results.append(result)

        passed_count = sum(1 for r in self.results if r.passed)

        return {
            "suite": "Resilience",
            "total_tests": len(tests),
            "passed": passed_count,
            "failed": len(tests) - passed_count,
            "results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "details": r.details,
                    "duration_ms": round(r.duration_ms, 2),
                }
                for r in self.results
            ],
        }

    # -------------------------------------------------------------------------
    # Fault Injection Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def create_timeout_mock() -> AsyncMock:
        """Create a mock that raises TimeoutError."""
        mock = AsyncMock()
        mock.side_effect = asyncio.TimeoutError("Simulated timeout")
        return mock

    @staticmethod
    def create_rate_limit_mock() -> AsyncMock:
        """Create a mock that simulates rate limiting."""
        mock = AsyncMock()
        mock.side_effect = Exception("429: Too Many Requests")
        return mock

    @staticmethod
    def create_malformed_response_mock() -> AsyncMock:
        """Create a mock that returns malformed data."""
        mock = AsyncMock()
        mock.return_value = MagicMock(
            content="{ invalid_json: true, ",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        return mock

    # -------------------------------------------------------------------------
    # Test Cases
    # -------------------------------------------------------------------------

    async def test_circuit_breaker_trips_on_failures(self) -> tuple[bool, str]:
        """
        Verify circuit breaker opens after failure threshold.

        Returns:
            Tuple of (passed, details message).
        """
        # Create isolated circuit breaker for testing
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=1.0)
        breaker = CircuitBreaker("test_trip", config)

        # Reset to known state
        breaker._state = CircuitState.CLOSED
        breaker._failure_count = 0

        # Record failures up to threshold
        for i in range(3):
            breaker.record_failure(Exception(f"Test failure {i}"))

        if breaker._state != CircuitState.OPEN:
            return False, f"Expected OPEN state, got {breaker._state.value}"

        # Verify requests are blocked
        can_proceed, retry_after = breaker.is_available()
        if can_proceed:
            return False, "Circuit should block requests when OPEN"

        return True, "Circuit breaker trips correctly after 3 failures"

    async def test_circuit_breaker_recovers(self) -> tuple[bool, str]:
        """
        Verify circuit breaker transitions OPEN -> HALF_OPEN -> CLOSED.

        Returns:
            Tuple of (passed, details message).
        """
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=0.1,  # Short timeout for testing
        )
        breaker = CircuitBreaker("test_recovery", config)

        # Force OPEN state
        breaker._state = CircuitState.OPEN
        breaker._last_transition_time = time.time() - 1  # Expired timeout

        # Check availability (should transition to HALF_OPEN)
        can_proceed, _ = breaker.is_available()

        if breaker._state != CircuitState.HALF_OPEN:
            return False, f"Expected HALF_OPEN, got {breaker._state.value}"

        # Record successful operations
        breaker.record_success()
        breaker.record_success()

        if breaker._state != CircuitState.CLOSED:
            return False, f"Expected CLOSED after successes, got {breaker._state.value}"

        return True, "Circuit breaker recovers correctly"

    async def test_semantic_failure_tracking(self) -> tuple[bool, str]:
        """
        Verify semantic failures are tracked and can trip circuit.

        Returns:
            Tuple of (passed, details message).
        """
        config = CircuitBreakerConfig(semantic_failure_threshold=3)
        breaker = CircuitBreaker("test_semantic", config)

        # Reset state
        breaker._state = CircuitState.CLOSED
        breaker._semantic_failures = {}

        # Record semantic failures
        breaker.record_semantic_failure("hallucination")
        breaker.record_semantic_failure("json_invalid")
        breaker.record_semantic_failure("empty_response")

        stats = breaker.get_stats()

        if stats["total_semantic_failures"] != 3:
            return False, f"Expected 3 semantic failures, got {stats['total_semantic_failures']}"

        if breaker._state != CircuitState.OPEN:
            return False, f"Expected OPEN after semantic threshold, got {breaker._state.value}"

        return True, "Semantic failures tracked and trip circuit correctly"

    async def test_budget_exceeded_blocks(self) -> tuple[bool, str]:
        """
        Verify requests are blocked when budget exceeded.

        Returns:
            Tuple of (passed, details message).
        """
        tracker = CostTracker()

        # Set very low budget
        tracker.set_budget(Provider.GOOGLE_GEMINI, daily_limit=0.001, monthly_limit=0.01)

        # Record usage that exceeds budget
        await tracker.record_usage(
            provider=Provider.GOOGLE_GEMINI,
            operation="test",
            input_tokens=100000,
            output_tokens=100000,
            model="test-model",
        )

        can_proceed, msg = tracker.check_can_proceed(Provider.GOOGLE_GEMINI)

        if can_proceed:
            return False, "Should block when budget exceeded"

        if "budget" not in msg.lower() and "exceeded" not in msg.lower():
            return False, f"Expected budget error message, got: {msg}"

        return True, "Budget enforcement blocks requests correctly"

    async def test_graceful_degradation_message(self) -> tuple[bool, str]:
        """
        Verify user-friendly error messages on failures.

        Returns:
            Tuple of (passed, details message).
        """
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test_graceful", config)

        # Trip the breaker
        breaker.record_failure(Exception("Internal error"))

        can_proceed, retry_after = breaker.is_available()

        if can_proceed:
            return False, "Breaker should be blocking"

        if retry_after is None or retry_after < 0:
            return False, "Should provide valid retry_after time"

        # Verify error info is usable for user-friendly message
        stats = breaker.get_stats()
        if "state" not in stats or "failure_count" not in stats:
            return False, "Stats should include state and failure info"

        return True, "Graceful degradation info available for error messages"


# =============================================================================
# Test Runner
# =============================================================================

async def run_resilience_suite() -> Dict[str, Any]:
    """
    Convenience function to run the full resilience suite.

    Returns:
        Dict with test summary and results.
    """
    suite = ResilienceEvalSuite()
    return await suite.run_all_tests()


__all__ = [
    "ResilienceEvalSuite",
    "ResilienceTestResult",
    "run_resilience_suite",
]
