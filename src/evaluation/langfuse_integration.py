"""
Phase 8.1: Enhanced Langfuse Integration for Evaluation System.

This module provides deep integration with Langfuse for observability,
including trial tracing, grader span logging, and cost tracking.
"""

import os
import logging
import functools
import time
from typing import Optional, Dict, Any, Callable, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .trial import Trial, GraderResult


class EvaluationTracer:
    """
    Singleton-style manager for Langfuse evaluation tracing.

    Wraps the Langfuse SDK to provide domain-specific logging for the
    evaluation system and immune system.
    """
    _instance: Optional["EvaluationTracer"] = None

    def __new__(cls) -> "EvaluationTracer":
        if cls._instance is None:
            cls._instance = super(EvaluationTracer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._init_client()
        self._initialized = True

    def _init_client(self):
        """Initialize the Langfuse client if credentials are available."""
        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            self.enabled = True
            logger.info("Langfuse evaluation tracer initialized successfully.")
        except ImportError:
            logger.warning("langfuse package not installed. Tracing disabled.")
            self.client = None
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}. Tracing disabled.")
            self.client = None
            self.enabled = False

    def create_trial_trace(
        self,
        trial: "Trial",
        run_name: str = "Evaluation Run",
    ) -> Optional[str]:
        """
        Creates a root trace for a specific Trial execution.

        Args:
            trial: The Trial object to trace
            run_name: Name for the trace run

        Returns:
            The trace ID if successful, None otherwise
        """
        if not self.enabled or not self.client:
            return None

        try:
            trace = self.client.trace(
                name=run_name,
                input=trial.input_prompt,
                output=str(trial.output) if trial.output else None,
                metadata={
                    "trial_id": trial.id,
                    "operation": trial.operation,
                    "model": trial.model,
                    "timestamp": trial.created_at.isoformat(),
                },
                tags=[trial.operation, trial.model] if trial.model else [trial.operation],
            )
            return trace.id
        except Exception as e:
            logger.error(f"Error creating trial trace: {e}")
            return None

    def log_grader_result(
        self,
        trace_id: str,
        result: "GraderResult",
        model_name: Optional[str] = None,
    ) -> None:
        """
        Logs a specific grader's execution.

        If it's a model-based grader, logs as a Generation.
        If it's a heuristic grader, logs as a Span.
        Also pushes the score to the trace.

        Args:
            trace_id: The parent trace ID
            result: The GraderResult to log
            model_name: Model name if this is a model-based grader
        """
        if not self.enabled or not self.client or not trace_id:
            return

        try:
            if model_name:
                # Model-based grader - log as generation
                token_usage = result.metadata.get("token_usage", {})
                self.client.generation(
                    trace_id=trace_id,
                    name=f"Grader: {result.name}",
                    model=model_name,
                    input=result.metadata.get("grader_prompt", "N/A"),
                    output=result.reason,
                    usage={
                        "input": token_usage.get("input", 0),
                        "output": token_usage.get("output", 0),
                        "total": token_usage.get("input", 0) + token_usage.get("output", 0),
                    },
                    metadata=result.metadata,
                )
            else:
                # Code/Heuristic grader - log as span
                self.client.span(
                    trace_id=trace_id,
                    name=f"Grader: {result.name}",
                    input={"reasoning": result.reason},
                    output={"score": result.score, "passed": result.passed},
                    metadata=result.metadata,
                )

            # Attach the score to the trace
            self.client.score(
                trace_id=trace_id,
                name=f"{result.name}-score",
                value=result.score,
                comment=result.reason[:500] if result.reason else None,
            )
        except Exception as e:
            logger.error(f"Error logging grader result: {e}")

    def log_trial_outcome(
        self,
        trace_id: str,
        trial: "Trial",
    ) -> None:
        """
        Log the final trial outcome and aggregate scores.

        Args:
            trace_id: The trace ID to update
            trial: The completed Trial object
        """
        if not self.enabled or not self.client or not trace_id:
            return

        try:
            # Calculate aggregate metrics
            if trial.grader_results:
                passed_count = sum(1 for r in trial.grader_results if r.passed)
                pass_rate = passed_count / len(trial.grader_results)
            else:
                pass_rate = 1.0

            # Log aggregate score
            self.client.score(
                trace_id=trace_id,
                name="trial_pass_rate",
                value=pass_rate,
                comment=f"Passed {passed_count}/{len(trial.grader_results)} graders" if trial.grader_results else "No graders",
            )

            # Log cost if available
            if trial.cost_usd > 0:
                self.client.score(
                    trace_id=trace_id,
                    name="trial_cost_usd",
                    value=trial.cost_usd,
                    comment=f"Estimated cost: ${trial.cost_usd:.6f}",
                )
        except Exception as e:
            logger.error(f"Error logging trial outcome: {e}")

    def flush(self) -> None:
        """Flush any pending events to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")


# --- Cost Calculation Helpers ---

# Gemini pricing (approximate as of 2026)
GEMINI_PRICING = {
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},  # per 1M tokens
    "gemini-3-pro-preview": {"input": 1.25, "output": 5.00},     # per 1M tokens
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},        # per 1M tokens
}


def calculate_gemini_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate cost for Gemini API calls.

    Args:
        model: The model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    pricing = GEMINI_PRICING.get(model, GEMINI_PRICING["gemini-3-flash-preview"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# --- Decorators ---

def trace_trial(run_name: str = "Evaluation Run"):
    """
    Decorator for the main evaluation loop processing a single trial.

    Injects trace_id into the Trial object and handles cleanup.

    Args:
        run_name: Name for the trace run
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = EvaluationTracer()

            # Extract trial from args
            trial = kwargs.get('trial')
            if not trial and args:
                for arg in args:
                    if hasattr(arg, 'input_prompt') and hasattr(arg, 'grader_results'):
                        trial = arg
                        break

            trace_id = None
            if trial:
                trace_id = tracer.create_trial_trace(trial, run_name)
                trial.langfuse_trace_id = trace_id or ""

            try:
                result = await func(*args, **kwargs)

                # Log final outcome
                if trial and trace_id:
                    tracer.log_trial_outcome(trace_id, trial)

                return result
            except Exception as e:
                if trace_id and tracer.enabled and tracer.client:
                    try:
                        tracer.client.trace(id=trace_id).update(
                            level="ERROR",
                            status_message=str(e),
                        )
                    except Exception:
                        pass
                raise
            finally:
                tracer.flush()

        return wrapper
    return decorator


def trace_grader(model_name: Optional[str] = None):
    """
    Decorator for Grader.grade() methods.

    Captures timing and enriches the result metadata with cost info
    for model-based graders.

    Args:
        model_name: The model name if this is a model-based grader
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = await func(self, *args, **kwargs)
            duration = time.time() - start_time

            # Add timing to metadata
            result.metadata["duration_ms"] = duration * 1000

            # Calculate cost for model graders
            if model_name:
                token_usage = result.metadata.get("token_usage", {})
                if token_usage and result.metadata.get("cost", 0) == 0:
                    in_tokens = token_usage.get("input", 0)
                    out_tokens = token_usage.get("output", 0)
                    cost = calculate_gemini_cost(model_name, in_tokens, out_tokens)
                    result.metadata["cost"] = cost
                    result.metadata["model"] = model_name

            return result

        return wrapper
    return decorator


# --- Helper Functions ---

def get_tracer() -> EvaluationTracer:
    """Get the singleton EvaluationTracer instance."""
    return EvaluationTracer()


def reset_tracer() -> None:
    """Reset the singleton tracer (for testing)."""
    EvaluationTracer._instance = None


__all__ = [
    "EvaluationTracer",
    "get_tracer",
    "reset_tracer",
    "trace_trial",
    "trace_grader",
    "calculate_gemini_cost",
    "GEMINI_PRICING",
]
