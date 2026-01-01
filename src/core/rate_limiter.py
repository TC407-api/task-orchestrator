"""Rate limiting and circuit breaker utilities."""
import asyncio
import time
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Callable, TypeVar, ParamSpec, Optional

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for API resilience.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        if breaker.can_proceed():
            try:
                result = await api_call()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0

    def can_proceed(self) -> tuple[bool, str]:
        """
        Check if a request can proceed.

        Returns:
            Tuple of (can_proceed, reason)
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True, ""

            elif self._state == CircuitState.OPEN:
                wait_time = self.recovery_timeout
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    wait_time = max(0, self.recovery_timeout - elapsed)
                return False, f"Circuit OPEN - too many failures. Retry in {wait_time:.0f}s"

            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True, "Circuit HALF_OPEN - testing recovery"
                return False, "Circuit HALF_OPEN - max test calls reached"

            return False, "Unknown state"

    def record_success(self) -> None:
        """Record a successful API call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    # Recovered - close circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    print("Circuit CLOSED - service recovered")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed API call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test - reopen
                self._state = CircuitState.OPEN
                print(f"Circuit OPEN - recovery failed")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    print(f"Circuit OPEN - {self._failure_count} consecutive failures")

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        with self._lock:
            self._check_state_transition()
            status = {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }
            if self._last_failure_time and self._state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                status["seconds_until_retry"] = max(0, self.recovery_timeout - elapsed)
            return status


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.rate = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.last_update = time.time()
        self.lock = Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / 60))
        self.last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens to wait for
        """
        while not self.acquire(tokens):
            time.sleep(0.1)

    async def async_wait(self, tokens: int = 1) -> None:
        """
        Async wait until tokens are available.

        Args:
            tokens: Number of tokens to wait for
        """
        while not self.acquire(tokens):
            await asyncio.sleep(0.1)

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get estimated wait time for tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Estimated seconds to wait
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / (self.rate / 60)


def rate_limited(limiter: RateLimiter):
    """
    Decorator to rate limit a function.

    Args:
        limiter: RateLimiter instance to use
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def async_rate_limited(limiter: RateLimiter):
    """
    Decorator to rate limit an async function.

    Args:
        limiter: RateLimiter instance to use
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            await limiter.async_wait()
            return await func(*args, **kwargs)
        return wrapper
    return decorator
