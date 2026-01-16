"""
Tests for Self-Healing module.

Tests circuit breaker, retry logic, and graceful degradation.
"""

import pytest
import time

from src.self_healing import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    with_circuit_breaker,
    with_circuit_breaker_fallback,
    with_retry,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        CircuitBreaker._breakers = {}

    def test_initial_state_is_closed(self, tmp_path):
        """Test that circuit breaker starts in CLOSED state."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test_service", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        assert breaker._state == CircuitState.CLOSED

    def test_trips_after_threshold_failures(self, tmp_path):
        """Test circuit trips to OPEN after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test_threshold", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        # Record failures
        for _ in range(3):
            breaker.record_failure(Exception("test error"))

        assert breaker._state == CircuitState.OPEN

    def test_half_open_after_timeout(self, tmp_path):
        """Test circuit transitions to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        breaker = CircuitBreaker("test_timeout", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        # Trip the circuit
        breaker.record_failure(Exception("test"))
        assert breaker._state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Check availability triggers transition
        can_proceed, _ = breaker.is_available()
        assert breaker._state == CircuitState.HALF_OPEN
        assert can_proceed is True

    def test_closes_after_success_threshold(self, tmp_path):
        """Test circuit closes after success threshold in HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.01,
        )
        breaker = CircuitBreaker("test_close", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        # Trip and wait for HALF_OPEN
        breaker.record_failure(Exception("test"))
        time.sleep(0.02)
        breaker.is_available()  # Triggers transition to HALF_OPEN

        # Record successes
        breaker.record_success()
        assert breaker._state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker._state == CircuitState.CLOSED

    def test_semantic_failure_tracking(self, tmp_path):
        """Test semantic failures are tracked separately."""
        config = CircuitBreakerConfig(semantic_failure_threshold=3)
        breaker = CircuitBreaker("test_semantic_unique", config)
        # Set state path BEFORE any operations to avoid loading stale state
        breaker.state_path = tmp_path / "test_semantic_breaker.json"
        # Reset state to ensure clean start
        breaker._state = CircuitState.CLOSED
        breaker._semantic_failures = {}
        breaker._failure_count = 0

        # Record different types of semantic failures
        breaker.record_semantic_failure("hallucination")
        breaker.record_semantic_failure("json_invalid")
        assert breaker._state == CircuitState.CLOSED

        breaker.record_semantic_failure("empty_response")
        assert breaker._state == CircuitState.OPEN

        stats = breaker.get_stats()
        assert stats["total_semantic_failures"] == 3
        assert stats["semantic_failures"]["hallucination"] == 1


class TestCallWithFallback:
    """Tests for graceful degradation with fallback."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        CircuitBreaker._breakers = {}

    @pytest.mark.asyncio
    async def test_returns_result_when_closed(self, tmp_path):
        """Test normal operation returns function result."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test_fallback_closed", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        async def success_func():
            return {"status": "live", "data": [1, 2, 3]}

        fallback = {"status": "cached", "data": []}

        result = await breaker.call_with_fallback(success_func, fallback)

        assert result["status"] == "live"
        assert result["data"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_returns_fallback_when_open(self, tmp_path):
        """Test returns fallback when circuit is OPEN."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=30)
        breaker = CircuitBreaker("test_fallback_open", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        # Trip the circuit
        breaker.record_failure(Exception("test"))
        assert breaker._state == CircuitState.OPEN

        async def should_not_be_called():
            raise AssertionError("Function should not be called when circuit is OPEN")

        fallback = {"status": "cached", "data": []}

        result = await breaker.call_with_fallback(should_not_be_called, fallback)

        assert result["status"] == "cached"

    @pytest.mark.asyncio
    async def test_returns_fallback_on_exception(self, tmp_path):
        """Test returns fallback when function raises exception."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test_fallback_exception", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        async def failing_func():
            raise ConnectionError("Network error")

        fallback = {"status": "cached", "data": []}

        result = await breaker.call_with_fallback(failing_func, fallback)

        assert result["status"] == "cached"
        assert breaker._failure_count == 1

    @pytest.mark.asyncio
    async def test_records_success_on_successful_call(self, tmp_path):
        """Test success is recorded after successful call."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.01)
        breaker = CircuitBreaker("test_fallback_success", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        # Trip and enter HALF_OPEN
        breaker.record_failure(Exception("test"))
        time.sleep(0.02)
        breaker.is_available()

        assert breaker._state == CircuitState.HALF_OPEN
        assert breaker._success_count == 0

        async def success_func():
            return "ok"

        await breaker.call_with_fallback(success_func, "fallback")

        assert breaker._success_count == 1

    def test_sync_fallback_returns_result_when_closed(self, tmp_path):
        """Test sync version returns result when circuit closed."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test_sync_closed", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        def success_func():
            return "live_data"

        result = breaker.call_with_fallback_sync(success_func, "cached_data")

        assert result == "live_data"

    def test_sync_fallback_returns_fallback_when_open(self, tmp_path):
        """Test sync version returns fallback when circuit open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test_sync_open", config)
        breaker.state_path = tmp_path / "test_breaker.json"

        breaker.record_failure(Exception("test"))

        def should_not_be_called():
            raise AssertionError("Should not be called")

        result = breaker.call_with_fallback_sync(should_not_be_called, "cached_data")

        assert result == "cached_data"


class TestFallbackDecorator:
    """Tests for with_circuit_breaker_fallback decorator."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        CircuitBreaker._breakers = {}

    @pytest.mark.asyncio
    async def test_decorator_returns_result(self, tmp_path):
        """Test decorator returns function result normally."""
        fallback = {"status": "fallback"}

        @with_circuit_breaker_fallback("test_decorator_normal", fallback)
        async def my_func():
            return {"status": "live"}

        # Override state path for test
        result = await my_func()
        assert result["status"] == "live"

    @pytest.mark.asyncio
    async def test_decorator_returns_fallback_on_error(self, tmp_path):
        """Test decorator returns fallback on exception."""
        fallback = {"status": "fallback", "error": None}

        @with_circuit_breaker_fallback("test_decorator_error", fallback)
        async def failing_func():
            raise ValueError("Something went wrong")

        result = await failing_func()
        assert result["status"] == "fallback"


class TestWithRetry:
    """Tests for retry decorator with exponential backoff."""

    @pytest.mark.asyncio
    async def test_succeeds_without_retry(self):
        """Test function succeeds on first try."""
        call_count = 0

        @with_retry(max_retries=3)
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        """Test function retries on transient errors."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Test raises exception after max retries exhausted."""
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Always times out")

        with pytest.raises(TimeoutError):
            await always_fails()

        assert call_count == 3  # Initial + 2 retries


class TestWithCircuitBreaker:
    """Tests for with_circuit_breaker decorator."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        CircuitBreaker._breakers = {}

    @pytest.mark.asyncio
    async def test_raises_when_circuit_open(self, tmp_path):
        """Test decorator raises CircuitBreakerOpen when circuit is open."""
        # Use unique name to avoid state file conflicts
        service_name = f"test_open_raise_{time.time_ns()}"
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=60)

        @with_circuit_breaker(service_name, config)
        async def my_func():
            return "should not reach"

        # First call succeeds, which registers the breaker
        try:
            # Make the call fail to trip the breaker
            breaker = CircuitBreaker.get(service_name, config)
            breaker.state_path = tmp_path / f"{service_name}.json"
            breaker.record_failure(Exception("test error"))
        except Exception:
            pass

        # Now verify circuit is open and raises
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            await my_func()

        assert service_name in exc_info.value.service
