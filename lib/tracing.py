"""
Grade 5 Langfuse Tracing - Auto-generated
==========================================
Usage:
    from lib.tracing import observe, langfuse

    @observe()
    def my_function():
        pass

    # Or for more control:
    @observe(name="custom_name")
    def another_function():
        pass

Langfuse Dashboard: http://localhost:3000
"""
import os
from functools import wraps

# Check if langfuse is available
import logging as _logging

_tracing_logger = _logging.getLogger(__name__)

def _noop_observe(name=None, **kwargs):
    """No-op decorator when langfuse is not available."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe as langfuse_observe

    _public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    _secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if _public_key and _secret_key:
        langfuse = Langfuse(
            public_key=_public_key,
            secret_key=_secret_key,
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
        observe = langfuse_observe
        TRACING_ENABLED = True
    else:
        _tracing_logger.warning(
            "LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY not set — tracing disabled"
        )
        langfuse = None
        TRACING_ENABLED = False
        observe = _noop_observe

except ImportError:
    langfuse = None
    TRACING_ENABLED = False
    observe = _noop_observe

def is_tracing_enabled() -> bool:
    """Check if Langfuse tracing is active."""
    return TRACING_ENABLED

def flush_traces():
    """Flush any pending traces to Langfuse."""
    if langfuse:
        langfuse.flush()

__all__ = ["langfuse", "observe", "is_tracing_enabled", "flush_traces", "TRACING_ENABLED"]
