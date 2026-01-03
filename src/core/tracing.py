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
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe as langfuse_observe

    # Initialize Langfuse client
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-e4acdb77-1e22-4a75-ac49-f44dc85c6ba7"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-ecb05847-f7c6-4ed5-9298-72ab4252c096"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    )

    # Re-export observe decorator
    observe = langfuse_observe

    TRACING_ENABLED = True

except ImportError:
    # Langfuse not installed - provide no-op fallback
    langfuse = None
    TRACING_ENABLED = False

    def observe(name=None, **kwargs):
        """No-op decorator when langfuse is not installed."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

def is_tracing_enabled() -> bool:
    """Check if Langfuse tracing is active."""
    return TRACING_ENABLED

def flush_traces():
    """Flush any pending traces to Langfuse."""
    if langfuse:
        langfuse.flush()

__all__ = ["langfuse", "observe", "is_tracing_enabled", "flush_traces", "TRACING_ENABLED"]
