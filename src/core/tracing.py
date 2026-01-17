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

    # SECURITY: Require env vars - no hardcoded fallback secrets
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if public_key and secret_key:
        # Initialize Langfuse client only if credentials provided
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        # Re-export observe decorator
        observe = langfuse_observe
        TRACING_ENABLED = True
    else:
        # Missing credentials - disable tracing
        langfuse = None
        TRACING_ENABLED = False

        def observe(name=None, **kwargs):
            """No-op decorator when credentials not configured."""
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper
            return decorator

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
