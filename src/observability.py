"""
Grade 5 Observability Integration for Task Orchestrator.

Provides Langfuse tracing and cost tracking for all MCP operations.

Usage:
    from .observability import trace_operation, get_tracer

    @trace_operation("sync_emails")
    async def tasks_sync_email(args):
        # ... your code
"""
import os
import functools
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Optional

# Try to import Langfuse, fall back to mock if not available
try:
    from langfuse import Langfuse
    from langfuse.types import TraceContext

    _langfuse_available = True
except ImportError:
    _langfuse_available = False
    Langfuse = None
    TraceContext = None


class Tracer:
    """
    Langfuse tracer for Grade 5 observability.

    Uses the new Langfuse v3 SDK API (start_span, start_generation, etc.)
    Gracefully degrades if Langfuse is not configured.
    """

    def __init__(self):
        self._client: Optional[Langfuse] = None
        self._current_trace_id: Optional[str] = None
        self._initialize()

    def _initialize(self):
        """Initialize Langfuse client from environment."""
        if not _langfuse_available:
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

        if public_key and secret_key:
            try:
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
            except Exception as e:
                print(f"[Grade5] Langfuse init failed: {e}")
                self._client = None

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._client is not None

    def start_trace(
        self,
        name: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        input: Optional[Any] = None,
    ):
        """Start a new trace using the v3 API."""
        if not self._client:
            return MockSpan(name)

        # Create a trace ID and use TraceContext
        trace_id = self._client.create_trace_id()
        self._current_trace_id = trace_id

        # Build trace context
        trace_context = TraceContext(trace_id=trace_id)

        # Combine metadata with user/session info
        full_metadata = {
            **(metadata or {}),
            "user_id": user_id or "mcp-server",
            "session_id": session_id or os.getenv("CLAUDE_SESSION_ID", "unknown"),
            "tags": tags or ["task-orchestrator", "mcp"],
        }

        # Use start_span with trace_context
        span = self._client.start_span(
            name=name,
            trace_context=trace_context,
            metadata=full_metadata,
            input=input,
        )
        return span

    def start_span(
        self,
        name: str,
        *,
        input: Optional[Any] = None,
        metadata: Optional[dict] = None,
        parent_span=None,
    ):
        """Create a span within the current trace."""
        if not self._client:
            return MockSpan(name)

        kwargs = {
            "name": name,
            "input": input,
            "metadata": metadata,
        }

        # Build trace context with parent if available
        if self._current_trace_id:
            trace_context = {"trace_id": self._current_trace_id}
            if parent_span and hasattr(parent_span, "id"):
                trace_context["parent_span_id"] = parent_span.id
            kwargs["trace_context"] = TraceContext(**trace_context)

        return self._client.start_span(**kwargs)

    def start_generation(
        self,
        name: str,
        *,
        model: str = "gemini-2.0-flash",
        input: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ):
        """Log an LLM generation."""
        if not self._client:
            return MockGeneration(name)

        kwargs = {
            "name": name,
            "model": model,
            "input": input,
            "metadata": metadata,
        }

        # Add trace context if available
        if self._current_trace_id:
            kwargs["trace_context"] = TraceContext(trace_id=self._current_trace_id)

        return self._client.start_generation(**kwargs)

    def score(
        self,
        name: str,
        value: float,
        *,
        comment: Optional[str] = None,
    ):
        """Add a score to the current trace."""
        if not self._client or not self._current_trace_id:
            return

        self._client.create_score(
            trace_id=self._current_trace_id,
            name=name,
            value=value,
            comment=comment,
        )

    def flush(self):
        """Flush pending traces."""
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass

    def shutdown(self):
        """Shutdown the tracer."""
        if self._client:
            try:
                self._client.shutdown()
            except Exception:
                pass

    # Backward compatibility aliases
    def trace(self, name: str, **kwargs):
        """Backward compatibility: alias for start_trace."""
        return self.start_trace(name, **kwargs)

    def span(self, name: str, **kwargs):
        """Backward compatibility: alias for start_span."""
        return self.start_span(name, **kwargs)

    def generation(self, name: str, **kwargs):
        """Backward compatibility: alias for start_generation."""
        return self.start_generation(name, **kwargs)


class MockSpan:
    """Mock span for when Langfuse is unavailable."""

    def __init__(self, name: str):
        self.name = name
        self.id = "mock_span"

    def update(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass


class MockGeneration:
    """Mock generation for when Langfuse is unavailable."""

    def __init__(self, name: str):
        self.name = name
        self.id = "mock_generation"

    def update(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def trace_operation(name: str, *, include_result: bool = False):
    """
    Decorator to trace an async function.

    Args:
        name: Name of the operation for the trace
        include_result: Whether to include the result in the trace output

    Example:
        @trace_operation("sync_emails")
        async def tasks_sync_email(args: dict) -> dict:
            # ... implementation
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()

            # Extract input from args
            input_data = None
            if args and len(args) > 1 and isinstance(args[1], dict):
                input_data = args[1]  # For methods, args[0] is self
            elif kwargs:
                input_data = kwargs

            # Create trace for this operation
            span = tracer.start_trace(
                name=f"task-orchestrator/{name}",
                metadata={
                    "function": func.__name__,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                tags=["mcp-tool", name],
                input=input_data,
            )

            try:
                start_time = datetime.utcnow()
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()

                # Update span with result
                span.update(
                    output=result if include_result else {"status": "success"},
                    metadata={"duration_seconds": duration},
                )
                span.end()

                # Add success score
                tracer.score("success", 1.0, comment="Operation completed successfully")

                return result

            except Exception as e:
                # Update span with error
                span.update(
                    output={"error": str(e)},
                    level="ERROR",
                    status_message=str(e),
                )
                span.end()

                # Add failure score
                tracer.score("success", 0.0, comment=f"Error: {str(e)}")

                raise

            finally:
                # Always flush to ensure traces are sent
                tracer.flush()

        return wrapper

    return decorator


@contextmanager
def trace_context(name: str, **metadata):
    """
    Context manager for tracing a block of code.

    Example:
        with trace_context("gmail_fetch", email_count=10):
            # ... fetch emails
    """
    tracer = get_tracer()
    span = tracer.start_trace(name=name, metadata=metadata)

    try:
        yield span
        span.update(output={"status": "success"})
    except Exception as e:
        span.update(output={"error": str(e)}, level="ERROR")
        raise
    finally:
        span.end()
        tracer.flush()
