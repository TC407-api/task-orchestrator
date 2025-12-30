"""Rate limiting utilities."""
import asyncio
import time
from functools import wraps
from threading import Lock
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


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
