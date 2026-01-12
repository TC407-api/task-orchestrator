"""
Langfuse Integration Module.

This module handles the transmission of evaluation results, scores, and traces
to the observability backend (Langfuse). It serves as the bridge between
local evaluation trials and remote monitoring.
"""

import logging
from typing import Optional, Any, Dict
from contextlib import nullcontext

from ..observability import get_tracer
from .trial import Trial, GraderResult

logger = logging.getLogger(__name__)


async def score_trial(trial: Trial) -> None:
    """
    Push all grader results from a completed trial to Langfuse.

    This function aggregates individual grader results and calculates/pushes
    an overall pass/fail score based on the trial's final determination.

    Args:
        trial (Trial): The completed trial object containing grader results.
    """
    try:
        tracer = get_tracer()
        if not tracer or not tracer.enabled:
            logger.debug("Tracer not enabled. Skipping score submission.")
            return

        # Score each specific grader result
        for result in trial.grader_results:
            try:
                tracer.score(
                    name=f"eval/{result.name}",
                    value=result.score,
                    comment=result.reason[:1000] if result.reason else None,
                )
            except Exception as e:
                logger.error(f"Failed to push score for grader '{result.name}': {e}")

        # Overall pass/fail score
        passed_count = len([r for r in trial.grader_results if r.passed])
        total_count = len(trial.grader_results)

        tracer.score(
            name="eval/overall",
            value=1.0 if trial.pass_fail else 0.0,
            comment=f"{passed_count}/{total_count} graders passed",
        )

        logger.debug(f"Scored trial {trial.id} in Langfuse.")

    except Exception as e:
        logger.error(f"Error in score_trial for trial {trial.id}: {e}", exc_info=True)


async def score_grader_result(
    result: GraderResult,
    trace_id: Optional[str] = None
) -> None:
    """
    Push a single grader result to Langfuse immediately.

    Args:
        result (GraderResult): The result from a single grader.
        trace_id (Optional[str]): Optional ID to link score to a specific trace.
    """
    try:
        tracer = get_tracer()
        if not tracer or not tracer.enabled:
            return

        tracer.score(
            name=f"eval/{result.name}",
            value=result.score,
            comment=result.reason,
        )
    except Exception as e:
        logger.error(f"Failed to push single grader score: {e}")


def create_eval_span(trial: Trial) -> Any:
    """
    Create a span for evaluation tracking.

    This should be used as a context manager or explicitly ended by the caller.

    Args:
        trial (Trial): The trial object to create a span for.

    Returns:
        Any: The span object from the tracer (type depends on observability implementation).
    """
    try:
        tracer = get_tracer()
        if not tracer or not tracer.enabled:
            return nullcontext()

        metadata: Dict[str, Any] = {
            "model": trial.model,
            "circuit_breaker_state": trial.circuit_breaker_state,
        }

        if trial.metadata:
            metadata.update(trial.metadata)

        return tracer.start_span(
            name=f"eval/{trial.operation}",
            input={"prompt": trial.input_prompt[:2000] if trial.input_prompt else ""},
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Failed to create eval span: {e}")
        return nullcontext()
