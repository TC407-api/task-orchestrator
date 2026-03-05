"""Cost and healing handler methods for MCP server."""
from ..core.cost_tracker import Provider
from ..observability import trace_operation
from ..self_healing import get_healing_status


class CostHandlers:
    """Mixin providing cost tracking and healing handler methods."""

    @trace_operation("cost_summary")
    async def _handle_cost_summary(self, args: dict) -> dict:
        """Get cost summary."""
        summary = self.cost_tracker.get_summary()

        provider_filter = args.get("provider", "all")
        if provider_filter != "all":
            return {
                "generated_at": summary["generated_at"],
                "providers": {provider_filter: summary["providers"].get(provider_filter, {})},
                "totals": summary["totals"],
            }

        return summary

    @trace_operation("cost_set_budget")
    async def _handle_cost_set_budget(self, args: dict) -> dict:
        """Set budget limits."""
        provider = Provider(args["provider"])

        self.cost_tracker.set_budget(
            provider,
            daily_limit=args.get("daily_limit"),
            monthly_limit=args.get("monthly_limit"),
        )

        return {
            "success": True,
            "provider": provider.value,
            "new_limits": {
                "daily": self.cost_tracker.budgets[provider].daily_limit_usd,
                "monthly": self.cost_tracker.budgets[provider].monthly_limit_usd,
            },
        }

    @trace_operation("healing_status")
    async def _handle_healing_status(self, args: dict) -> dict:
        """Get self-healing system status."""
        status = get_healing_status()

        status["circuit_breakers"]["gmail_service"] = self._gmail_breaker.get_stats()
        status["circuit_breakers"]["calendar_service"] = self._calendar_breaker.get_stats()
        status["circuit_breakers"]["gemini_service"] = self._gemini_breaker.get_stats()

        return {
            "success": True,
            "healing_status": status,
        }
