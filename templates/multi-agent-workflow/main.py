#!/usr/bin/env python3
"""
Multi-Agent Workflow Example

Demonstrates the architect → builder → QC agent pattern for
coordinated task execution.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentRole(Enum):
    """Agent roles in the workflow."""
    ARCHITECT = "architect"
    BUILDER = "builder"
    QC = "qc"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    role: AgentRole
    success: bool
    output: str
    metadata: Dict[str, Any]


async def spawn_agent(
    role: AgentRole,
    prompt: str,
    context: Optional[str] = None,
) -> AgentResult:
    """
    Spawn an agent with a specific role.

    Args:
        role: The agent's role in the workflow
        prompt: The task prompt
        context: Optional context from previous agents

    Returns:
        AgentResult with the output
    """
    # System prompts for each role
    system_prompts = {
        AgentRole.ARCHITECT: (
            "You are a software architect. Plan the implementation, "
            "identify components, and define interfaces. Be thorough but concise."
        ),
        AgentRole.BUILDER: (
            "You are a software builder. Implement the code based on the "
            "architect's plan. Follow best practices and write clean code."
        ),
        AgentRole.QC: (
            "You are a QC engineer. Review the implementation, identify "
            "issues, and verify correctness. Be critical but constructive."
        ),
    }

    full_prompt = prompt
    if context:
        full_prompt = f"Context from previous agent:\n{context}\n\nTask:\n{prompt}"

    try:
        from google import genai

        client = genai.Client()

        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=full_prompt,
            config={"system_instruction": system_prompts[role]},
        )

        return AgentResult(
            role=role,
            success=True,
            output=response.text,
            metadata={"model": "gemini-3-flash-preview"},
        )

    except Exception as e:
        return AgentResult(
            role=role,
            success=False,
            output=str(e),
            metadata={"error": True},
        )


async def run_workflow(task: str) -> List[AgentResult]:
    """
    Run the architect → builder → QC workflow.

    Args:
        task: The high-level task description

    Returns:
        List of AgentResult from each stage
    """
    results = []

    # Stage 1: Architect plans
    print("\n[ARCHITECT] Planning...")
    architect_result = await spawn_agent(
        role=AgentRole.ARCHITECT,
        prompt=f"Plan the implementation for: {task}",
    )
    results.append(architect_result)

    if not architect_result.success:
        print(f"[ARCHITECT] Failed: {architect_result.output}")
        return results

    print(f"[ARCHITECT] Plan:\n{architect_result.output[:500]}...")

    # Stage 2: Builder implements
    print("\n[BUILDER] Implementing...")
    builder_result = await spawn_agent(
        role=AgentRole.BUILDER,
        prompt="Implement based on the architect's plan",
        context=architect_result.output,
    )
    results.append(builder_result)

    if not builder_result.success:
        print(f"[BUILDER] Failed: {builder_result.output}")
        return results

    print(f"[BUILDER] Implementation:\n{builder_result.output[:500]}...")

    # Stage 3: QC validates
    print("\n[QC] Validating...")
    qc_result = await spawn_agent(
        role=AgentRole.QC,
        prompt="Review the implementation and identify any issues",
        context=f"Plan:\n{architect_result.output}\n\nImplementation:\n{builder_result.output}",
    )
    results.append(qc_result)

    if qc_result.success:
        print(f"[QC] Review:\n{qc_result.output[:500]}...")

    return results


async def main():
    """Run an example workflow."""
    print("Multi-Agent Workflow Example")
    print("=" * 50)

    task = "Create a function that validates email addresses"

    results = await run_workflow(task)

    print("\n" + "=" * 50)
    print("WORKFLOW SUMMARY")
    print("=" * 50)

    for result in results:
        status = "✓" if result.success else "✗"
        print(f"{status} {result.role.value.upper()}: {'Success' if result.success else 'Failed'}")


if __name__ == "__main__":
    asyncio.run(main())
