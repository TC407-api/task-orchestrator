"""
Grade 5 Self-Healing Integration for Task Orchestrator.

Provides circuit breaker, exponential backoff, and graceful degradation
for resilient MCP operations.

Usage:
    from .self_healing import with_circuit_breaker, with_retry

    @with_circuit_breaker("gmail_service")
    @with_retry(max_retries=3)
    async def fetch_emails():
        # ... your code
"""
import asyncio
import functools
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Awaitable, Callable, Optional, TypeVar

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Blocking requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 3        # Failures to trip circuit
    success_threshold: int = 2        # Successes in HALF_OPEN to close
    timeout_seconds: float = 30.0     # Time in OPEN before testing
    half_open_max_requests: int = 3   # Max concurrent requests in HALF_OPEN
    semantic_failure_threshold: int = 5  # Semantic failures to trip circuit


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""
    max_retries: int = 3
    base_delay: float = 2.0           # Base delay in seconds
    max_delay: float = 60.0           # Maximum delay
    multiplier: float = 2.0           # Exponential multiplier
    jitter: float = 0.1               # Random jitter factor (0-1)
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Integrates with Grade 5 infrastructure for observability and persistence.
    """

    _breakers: dict[str, "CircuitBreaker"] = {}
    _lock = Lock()

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state_path = Path.home() / ".claude" / "grade5" / "circuit-breakers" / f"{name}.json"
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_transition_time = time.time()
        self._half_open_requests = 0
        self._trip_count = 0
        self._semantic_failures: dict[str, int] = {}  # Track semantic failures by type
        self._state_lock = Lock()

        # Ensure directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    @classmethod
    def get(cls, name: str, config: Optional[CircuitBreakerConfig] = None) -> "CircuitBreaker":
        """Get or create a circuit breaker by name."""
        with cls._lock:
            if name not in cls._breakers:
                cls._breakers[name] = CircuitBreaker(name, config)
            return cls._breakers[name]

    def _load_state(self) -> None:
        """Load state from persistent storage."""
        try:
            if self.state_path.exists():
                with open(self.state_path) as f:
                    data = json.load(f)
                    self._state = CircuitState(data.get("state", "CLOSED"))
                    self._failure_count = data.get("failure_count", 0)
                    self._success_count = data.get("success_count", 0)
                    self._last_failure_time = data.get("last_failure_time")
                    self._last_transition_time = data.get("last_transition_time", time.time())
                    self._half_open_requests = data.get("half_open_requests", 0)
                    self._trip_count = data.get("trip_count", 0)
                    self._semantic_failures = data.get("semantic_failures", {})
        except Exception:
            pass

    def _save_state(self) -> None:
        """Save state to persistent storage."""
        try:
            with open(self.state_path, "w") as f:
                json.dump({
                    "state": self._state.value,
                    "failure_count": self._failure_count,
                    "success_count": self._success_count,
                    "last_failure_time": self._last_failure_time,
                    "last_transition_time": self._last_transition_time,
                    "half_open_requests": self._half_open_requests,
                    "trip_count": self._trip_count,
                    "semantic_failures": self._semantic_failures,
                    "updated_at": datetime.utcnow().isoformat(),
                }, f, indent=2)
        except Exception:
            pass

    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_transition_time
            if elapsed >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        self._state = new_state
        self._last_transition_time = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_requests = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.OPEN:
            self._trip_count += 1

        self._save_state()

    def _can_allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            return False

        # HALF_OPEN: allow limited requests
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_requests < self.config.half_open_max_requests:
                self._half_open_requests += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._state_lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

            self._save_state()

    def record_failure(self, error: Exception) -> None:
        """Record a failed operation."""
        with self._state_lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            self._save_state()

    def record_semantic_failure(self, failure_type: str) -> None:
        """
        Record a semantic failure (bad output quality, not a crash).

        Semantic failures track quality issues like hallucinations, wrong format,
        or invalid content that don't raise exceptions but indicate poor output.

        Args:
            failure_type: Category of semantic failure (e.g., 'hallucination',
                         'json_invalid', 'empty_response', 'eval_failed')
        """
        with self._state_lock:
            # Increment count for this failure type
            self._semantic_failures[failure_type] = self._semantic_failures.get(failure_type, 0) + 1
            self._last_failure_time = time.time()

            # Check if total semantic failures exceed threshold
            total_semantic = sum(self._semantic_failures.values())
            if total_semantic >= self.config.semantic_failure_threshold:
                self._transition_to(CircuitState.OPEN)

            self._save_state()

    def reset_semantic_failures(self) -> None:
        """Reset semantic failure counts (e.g., after successful recovery)."""
        with self._state_lock:
            self._semantic_failures = {}
            self._save_state()

    def is_available(self) -> tuple[bool, Optional[float]]:
        """
        Check if the circuit breaker allows requests.

        Returns:
            Tuple of (can_proceed, retry_after_seconds)
        """
        with self._state_lock:
            self._check_state_transition()

            if self._can_allow_request():
                return True, None

            retry_after = self.config.timeout_seconds - (time.time() - self._last_transition_time)
            return False, max(0, retry_after)

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._state_lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "trip_count": self._trip_count,
                "last_failure_time": self._last_failure_time,
                "semantic_failures": self._semantic_failures.copy(),
                "total_semantic_failures": sum(self._semantic_failures.values()),
            }

    async def call_with_fallback(
        self,
        func: Callable[..., Awaitable[T]],
        fallback: T,
        *args,
        **kwargs,
    ) -> T:
        """
        Call function with circuit breaker protection and fallback on failure.

        When circuit is OPEN, returns fallback immediately instead of raising.
        This enables graceful degradation - the system continues with cached/default
        values rather than failing hard.

        Pattern learned from: mind-health-flow (AI note generation with circuit breaker)

        Args:
            func: Async function to call
            fallback: Value to return if circuit is OPEN or call fails
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func, or fallback value if circuit is OPEN or call fails

        Example:
            breaker = CircuitBreaker.get("ai_service")
            cached_response = {"status": "cached", "data": last_known_good}

            result = await breaker.call_with_fallback(
                fetch_ai_response,
                fallback=cached_response,
                prompt="Generate summary"
            )
        """
        can_proceed, retry_after = self.is_available()

        if not can_proceed:
            # Circuit is OPEN - return fallback gracefully
            return fallback

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            # Return fallback instead of raising
            return fallback

    def call_with_fallback_sync(
        self,
        func: Callable[..., T],
        fallback: T,
        *args,
        **kwargs,
    ) -> T:
        """
        Synchronous version of call_with_fallback.

        Args:
            func: Sync function to call
            fallback: Value to return if circuit is OPEN or call fails
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func, or fallback value if circuit is OPEN or call fails
        """
        can_proceed, retry_after = self.is_available()

        if not can_proceed:
            return fallback

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            return fallback


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, service: str, retry_after: float):
        self.service = service
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker open for {service}. Retry after {retry_after:.1f}s")


def with_circuit_breaker(
    service_name: str,
    config: Optional[CircuitBreakerConfig] = None,
):
    """
    Decorator to wrap async function with circuit breaker protection.

    Example:
        @with_circuit_breaker("gmail")
        async def fetch_emails():
            # ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = CircuitBreaker.get(service_name, config)

            can_proceed, retry_after = breaker.is_available()
            if not can_proceed:
                raise CircuitBreakerOpen(service_name, retry_after or 0.0)

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                raise

        return wrapper
    return decorator


def with_circuit_breaker_fallback(
    service_name: str,
    fallback_value: object,
    config: Optional[CircuitBreakerConfig] = None,
):
    """
    Decorator to wrap async function with circuit breaker and graceful degradation.

    Unlike with_circuit_breaker which raises CircuitBreakerOpen, this decorator
    returns the fallback_value when the circuit is open or when the call fails.

    Pattern learned from: mind-health-flow (AI note generation with circuit breaker)

    Example:
        @with_circuit_breaker_fallback("ai_service", fallback_value={"cached": True})
        async def fetch_ai_response(prompt: str):
            # ...

        # When circuit is open, returns {"cached": True} instead of raising
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = CircuitBreaker.get(service_name, config)
            return await breaker.call_with_fallback(func, fallback_value, *args, **kwargs)
        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, asyncio.TimeoutError),
):
    """
    Decorator to add exponential backoff retry to async function.

    Example:
        @with_retry(max_retries=3, base_delay=2.0)
        async def make_api_call():
            # ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        break

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (multiplier ** attempt), max_delay)
                    delay = delay * (1 + random.uniform(-jitter, jitter))

                    await asyncio.sleep(delay)

            if last_exception is not None:
                raise last_exception
            # Should never reach here, but satisfy type checker
            raise RuntimeError("Retry exhausted with no exception captured")

        return wrapper
    return decorator


def get_healing_status() -> dict:
    """Get status of all circuit breakers and healing configuration."""
    status = {
        "circuit_breakers": {},
        "config_path": str(Path.home() / ".claude" / "healing" / "config.json"),
    }

    # Get all registered circuit breakers
    with CircuitBreaker._lock:
        for name, breaker in CircuitBreaker._breakers.items():
            status["circuit_breakers"][name] = breaker.get_stats()

    # Load healing config
    config_path = Path.home() / ".claude" / "healing" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                status["config"] = json.load(f)
        except Exception:
            status["config"] = {"error": "Failed to load config"}

    return status
