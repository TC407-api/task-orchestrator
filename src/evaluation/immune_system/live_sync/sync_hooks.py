"""
Sync Event Hooks for Graphiti Federation.

Provides a middleware-style hook system for sync lifecycle events
with support for validation, transformation, and error handling.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HookEventType(Enum):
    """Defines the lifecycle events for the Sync Engine."""
    BEFORE_PUSH = "before_push"
    AFTER_PUSH = "after_push"
    BEFORE_PULL = "before_pull"
    AFTER_PULL = "after_pull"
    ON_ERROR = "on_error"


@dataclass
class SyncContext:
    """
    Context object passed through the hook chain.

    Attributes:
        event_type: The current lifecycle event.
        project_id: The identifier of the project being synced.
        payload: The data being synced (patterns). Mutable for transformation.
        metadata: Arbitrary data passed between hooks.
        errors: List of exceptions encountered during the chain.
        _aborted: Internal flag to stop propagation.
    """
    event_type: HookEventType
    project_id: str
    payload: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Exception] = field(default_factory=list)
    _aborted: bool = field(default=False, init=False)

    def abort(self, reason: str = "Aborted by hook") -> None:
        """Stops execution of subsequent hooks and marks sync as cancelled."""
        self._aborted = True
        self.metadata["abort_reason"] = reason
        logger.info(f"Sync chain aborted for project {self.project_id}: {reason}")

    @property
    def is_aborted(self) -> bool:
        return self._aborted


HookCallback = Callable[[SyncContext], Awaitable[None]]


class SyncHooks:
    """
    Registry and executor for Sync Engine hooks.

    Supports middleware-style chaining where hooks can validate,
    transform data, or handle errors.
    """

    def __init__(self):
        self._hooks: Dict[HookEventType, List[HookCallback]] = {
            event: [] for event in HookEventType
        }

    def register(
        self,
        event_type: HookEventType,
        callback: HookCallback,
        priority: int = 100
    ) -> None:
        """
        Register a new hook callback.

        Args:
            event_type: The lifecycle event to attach to.
            callback: The async function to execute.
            priority: Execution order (lower numbers run first).
        """
        setattr(callback, "_priority", priority)
        self._hooks[event_type].append(callback)
        self._hooks[event_type].sort(key=lambda f: getattr(f, "_priority", 100))

        logger.debug(
            f"Registered hook '{callback.__name__}' for '{event_type.value}' "
            f"with priority {priority}"
        )

    def before_push(self, priority: int = 100):
        """Decorator for registering pre-push hooks."""
        def decorator(func: HookCallback):
            self.register(HookEventType.BEFORE_PUSH, func, priority)
            return func
        return decorator

    def after_push(self, priority: int = 100):
        """Decorator for registering post-push hooks."""
        def decorator(func: HookCallback):
            self.register(HookEventType.AFTER_PUSH, func, priority)
            return func
        return decorator

    def before_pull(self, priority: int = 100):
        """Decorator for registering pre-pull hooks."""
        def decorator(func: HookCallback):
            self.register(HookEventType.BEFORE_PULL, func, priority)
            return func
        return decorator

    def after_pull(self, priority: int = 100):
        """Decorator for registering post-pull hooks."""
        def decorator(func: HookCallback):
            self.register(HookEventType.AFTER_PULL, func, priority)
            return func
        return decorator

    def on_error(self, priority: int = 100):
        """Decorator for registering error handlers."""
        def decorator(func: HookCallback):
            self.register(HookEventType.ON_ERROR, func, priority)
            return func
        return decorator

    async def emit(
        self,
        event_type: HookEventType,
        context: SyncContext
    ) -> SyncContext:
        """
        Trigger all hooks registered for a specific event.

        If an error occurs in a hook:
        1. The exception is caught
        2. The ON_ERROR chain is triggered
        3. The original chain is aborted

        Args:
            event_type: The event to trigger.
            context: The metadata and payload context.

        Returns:
            The potentially modified context.
        """
        if context.event_type != event_type and event_type != HookEventType.ON_ERROR:
            context.event_type = event_type

        hooks = self._hooks.get(event_type, [])

        if not hooks:
            return context

        logger.debug(f"Emitting event {event_type.value} with {len(hooks)} hooks.")

        for hook in hooks:
            if context.is_aborted and event_type != HookEventType.ON_ERROR:
                logger.debug("Context aborted, skipping remaining hooks.")
                break

            try:
                await hook(context)
            except Exception as exc:
                logger.error(f"Error in hook '{hook.__name__}': {exc}", exc_info=True)
                context.errors.append(exc)

                if event_type != HookEventType.ON_ERROR:
                    context.abort(reason=f"Exception in {hook.__name__}: {str(exc)}")
                    await self.emit(HookEventType.ON_ERROR, context)

        return context

    def clear(self, event_type: Optional[HookEventType] = None) -> None:
        """Clear registered hooks for an event type or all events."""
        if event_type:
            self._hooks[event_type] = []
        else:
            for event in HookEventType:
                self._hooks[event] = []


__all__ = [
    "SyncHooks",
    "SyncContext",
    "HookEventType",
    "HookCallback",
]
