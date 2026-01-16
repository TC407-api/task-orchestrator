#!/usr/bin/env python3
"""
Self-Healing Agent Example

Demonstrates circuit breaker patterns and automatic retry with
exponential backoff for resilient agent execution.
"""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for resilient API calls.

    CLOSED → failures ≥ threshold → OPEN
    OPEN → timeout expires → HALF_OPEN
    HALF_OPEN → success → CLOSED / failure → OPEN
    """
    failure_threshold: int = 3
    recovery_timeout: float = 30.0  # seconds

    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_time: Optional[datetime] = None

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        state = self.state
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> Any:
    """
    Execute function with exponential backoff retry.

    Args:
        func: Async function to execute
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter: Add random jitter to delay

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            if attempt == max_retries:
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** attempt), max_delay)

            # Add jitter
            if jitter:
                delay = delay * (0.5 + random.random())

            print(f"  Retry {attempt + 1}/{max_retries} in {delay:.2f}s...")
            await asyncio.sleep(delay)

    raise last_exception


# Global circuit breaker
circuit = CircuitBreaker()


async def spawn_agent_resilient(
    prompt: str,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
) -> Dict:
    """
    Spawn an agent with self-healing capabilities.

    Uses circuit breaker and exponential backoff for resilience.

    Args:
        prompt: The task prompt
        model: Model to use
        max_retries: Maximum retry attempts

    Returns:
        Dict with response and status
    """
    # Check circuit breaker
    if not circuit.can_execute():
        return {
            "success": False,
            "error": "Circuit breaker OPEN - too many failures",
            "circuit_status": circuit.get_status(),
        }

    async def execute():
        """Inner execution function."""
        from google import genai

        client = genai.Client()

        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )

        return response.text

    try:
        result = await retry_with_backoff(
            execute,
            max_retries=max_retries,
        )

        circuit.record_success()

        return {
            "success": True,
            "response": result,
            "circuit_status": circuit.get_status(),
        }

    except Exception as e:
        circuit.record_failure()

        return {
            "success": False,
            "error": str(e),
            "circuit_status": circuit.get_status(),
        }


async def main():
    """Run self-healing agent example."""
    print("Self-Healing Agent Example")
    print("=" * 50)
    print("Circuit Breaker: failure_threshold=3, recovery_timeout=30s")
    print("Retry: max_retries=3, exponential backoff with jitter")
    print("=" * 50)

    tasks = [
        "What is the capital of France?",
        "Name a programming language",
        "What color is the sky?",
    ]

    for task in tasks:
        print(f"\nTask: {task}")

        result = await spawn_agent_resilient(task)

        if result["success"]:
            print(f"Response: {result['response'][:100]}...")
        else:
            print(f"Error: {result['error']}")

        status = result["circuit_status"]
        print(f"Circuit: {status['state']} (failures: {status['failure_count']}/{status['threshold']})")

    # Final status
    print("\n" + "=" * 50)
    print("CIRCUIT BREAKER STATUS")
    print("=" * 50)
    status = circuit.get_status()
    for key, value in status.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
