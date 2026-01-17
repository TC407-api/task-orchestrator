#!/usr/bin/env python3
"""
Basic Agent Example

A simple example showing how to spawn a Gemini agent using task-orchestrator.
"""

import asyncio
from typing import Optional


async def spawn_agent(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gemini-3-flash-preview",
) -> dict:
    """
    Spawn a Gemini agent to execute a task.

    Args:
        prompt: The task prompt for the agent
        system_prompt: Optional system prompt to guide behavior
        model: Model to use (default: gemini-3-flash-preview)

    Returns:
        Dict with agent response
    """
    # Import the MCP client or use direct API
    try:
        from google import genai

        client = genai.Client()

        config = {"system_instruction": system_prompt} if system_prompt else {}

        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        return {
            "success": True,
            "response": response.text,
            "model": model,
        }

    except ImportError:
        return {
            "success": False,
            "error": "google-generativeai not installed. Run: pip install google-generativeai",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def main():
    """Run a simple agent task."""
    print("Basic Agent Example")
    print("=" * 40)

    # Example task
    result = await spawn_agent(
        prompt="Write a haiku about coding",
        system_prompt="You are a creative assistant. Be concise and creative.",
    )

    if result["success"]:
        print(f"\nAgent Response:\n{result['response']}")
    else:
        print(f"\nError: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
