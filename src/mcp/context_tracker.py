"""Context window tracking for dynamic tool loading.

Monitors context window usage and triggers dynamic mode when threshold is reached.
"""
from typing import Dict, Any, Optional, Callable
import json


class ContextTracker:
    """
    Tracks context window usage to determine when to switch to dynamic tool loading.

    When remaining context drops below threshold (default 10%), the system
    switches to dynamic mode with only core tools exposed.
    """

    def __init__(
        self,
        max_context: int = 200000,
        threshold_pct: float = 0.1,
        token_estimator: Optional[Callable[[Dict[str, Any]], int]] = None
    ):
        """
        Initialize the context tracker.

        Args:
            max_context: Maximum context window size (default: 200,000 tokens)
            threshold_pct: Percentage threshold for dynamic mode (default: 0.1 = 10%)
            token_estimator: Optional custom token estimator function
        """
        self.max_context = max_context
        self.threshold_pct = threshold_pct
        self.current_usage = 0
        self._estimator = token_estimator or self._default_estimator
        self._cache: Dict[str, int] = {}

    def _default_estimator(self, tool_def: Dict[str, Any]) -> int:
        """Default token estimator using chars/4 heuristic."""
        return len(str(tool_def)) // 4

    def update_usage(self, tokens: int) -> None:
        """Update current token usage by accumulating tokens."""
        self.current_usage += tokens

    def remaining_pct(self) -> float:
        """Return remaining context as percentage (0.0 to 1.0)."""
        if self.max_context <= 0:
            return 0.0
        return (self.max_context - self.current_usage) / self.max_context

    def below_threshold(self) -> bool:
        """Check if remaining context is below threshold."""
        return self.remaining_pct() < self.threshold_pct

    def estimate_tool_tokens(self, tool_def: Dict[str, Any]) -> int:
        """Estimate tokens for a tool definition with caching."""
        # Create cache key from tool name
        cache_key = tool_def.get("name", str(id(tool_def)))
        if cache_key in self._cache:
            return self._cache[cache_key]

        tokens = self._estimator(tool_def)
        self._cache[cache_key] = tokens
        return tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get context tracking statistics."""
        remaining = self.max_context - self.current_usage
        usage_pct = self.current_usage / self.max_context if self.max_context > 0 else 0.0
        return {
            "max_context": self.max_context,
            "current_usage": self.current_usage,
            "remaining_tokens": remaining,
            "usage_pct": usage_pct,
            "threshold_warning": self.below_threshold(),
        }
