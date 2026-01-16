#!/usr/bin/env python3
"""
Cost-Controlled Agent Example

Demonstrates how to manage API costs with budget limits and tracking.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class CostRecord:
    """Record of a single API call cost."""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class BudgetTracker:
    """Tracks costs and enforces budget limits."""
    daily_limit_usd: float = 1.0
    session_limit_usd: float = 0.50
    _records: List[CostRecord] = field(default_factory=list)

    # Pricing per 1M tokens (approximate)
    PRICING = {
        "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
        "gemini-3-pro-preview": {"input": 1.25, "output": 5.00},
    }

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for token usage."""
        pricing = self.PRICING.get(model, self.PRICING["gemini-3-flash-preview"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostRecord:
        """Record API usage and return the cost record."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        record = CostRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self._records.append(record)
        return record

    @property
    def session_cost(self) -> float:
        """Total cost for this session."""
        return sum(r.cost_usd for r in self._records)

    @property
    def daily_cost(self) -> float:
        """Total cost for today."""
        today = datetime.now().date()
        return sum(
            r.cost_usd for r in self._records
            if r.timestamp.date() == today
        )

    def can_proceed(self) -> bool:
        """Check if we're within budget limits."""
        return (
            self.session_cost < self.session_limit_usd and
            self.daily_cost < self.daily_limit_usd
        )

    def get_summary(self) -> Dict:
        """Get cost summary."""
        return {
            "session_cost_usd": round(self.session_cost, 6),
            "daily_cost_usd": round(self.daily_cost, 6),
            "session_limit_usd": self.session_limit_usd,
            "daily_limit_usd": self.daily_limit_usd,
            "session_remaining_usd": round(self.session_limit_usd - self.session_cost, 6),
            "daily_remaining_usd": round(self.daily_limit_usd - self.daily_cost, 6),
            "total_calls": len(self._records),
        }


# Global tracker instance
budget_tracker = BudgetTracker()


async def spawn_agent_with_budget(
    prompt: str,
    model: str = "gemini-3-flash-preview",
) -> Dict:
    """
    Spawn an agent with budget checking.

    Args:
        prompt: The task prompt
        model: Model to use

    Returns:
        Dict with response and cost info
    """
    # Check budget before proceeding
    if not budget_tracker.can_proceed():
        return {
            "success": False,
            "error": "Budget limit exceeded",
            "cost_summary": budget_tracker.get_summary(),
        }

    try:
        from google import genai

        client = genai.Client()

        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )

        # Extract token usage (approximate if not available)
        input_tokens = len(prompt) // 4  # Rough estimate
        output_tokens = len(response.text) // 4

        # Record the cost
        cost_record = budget_tracker.record_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return {
            "success": True,
            "response": response.text,
            "cost_usd": cost_record.cost_usd,
            "cost_summary": budget_tracker.get_summary(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cost_summary": budget_tracker.get_summary(),
        }


async def main():
    """Run cost-controlled agent tasks."""
    print("Cost-Controlled Agent Example")
    print("=" * 50)
    print(f"Session limit: ${budget_tracker.session_limit_usd:.2f}")
    print(f"Daily limit: ${budget_tracker.daily_limit_usd:.2f}")
    print("=" * 50)

    # Run a few tasks and track costs
    tasks = [
        "What is 2 + 2?",
        "Write a one-line Python hello world",
        "List 3 Python web frameworks",
    ]

    for task in tasks:
        print(f"\nTask: {task}")

        result = await spawn_agent_with_budget(task)

        if result["success"]:
            print(f"Response: {result['response'][:100]}...")
            print(f"Cost: ${result['cost_usd']:.6f}")
        else:
            print(f"Error: {result['error']}")

        print(f"Session total: ${result['cost_summary']['session_cost_usd']:.6f}")

    # Print final summary
    print("\n" + "=" * 50)
    print("COST SUMMARY")
    print("=" * 50)
    summary = budget_tracker.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
