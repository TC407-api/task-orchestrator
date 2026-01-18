# Twitter/X Thread

## Tweet 1 (Hook)
Just open-sourced Task Orchestrator - production safety for Claude Code agents.

AI agents don't crash. They confidently give wrong answers.

This fixes that. MIT licensed, 680+ tests.

Thread on what it does and why I built it:

github.com/TC407-api/task-orchestrator

## Tweet 2 (Problem)
The problem:

<1 in 3 teams are satisfied with their AI agent guardrails (Cleanlab 2025)

At 95% per-step reliability, a 20-step workflow has 36% success rate.

The math doesn't lie. Every step that could fail, will.

## Tweet 3 (My experience)
What I kept seeing with Claude Code:

- Hallucinated file paths
- "Tests pass" (they didn't run)
- Same errors recurring across sessions
- No way to catch semantic failures

Tools exist for crashes. Nothing for "confident but wrong."

## Tweet 4 (Solution)
Task Orchestrator adds an "immune system":

- Detects semantic failures (not just crashes)
- Learns from mistakes
- Blocks similar errors before they happen
- Human-in-the-loop for high-risk ops

## Tweet 5 (Technical)
Technical bits:

- ML-powered risk prediction (O(1) lookup)
- Self-healing circuit breakers
- Cost tracking across Gemini/OpenAI
- 680+ tests
- Plugin architecture

Works with Gemini free tier.

## Tweet 6 (Quick start)
Get started in 3 commands:

```
git clone github.com/TC407-api/task-orchestrator
cd task-orchestrator && pip install -r requirements.txt
claude mcp add task-orchestrator python mcp_server.py
```

## Tweet 7 (CTA)
If you're deploying AI agents and want them to fail less:

- Star the repo
- Try it out
- Tell me what guardrails you wish existed

This is the infrastructure layer that's been missing.

github.com/TC407-api/task-orchestrator

---

## Alt: Single Tweet Version

Just open-sourced Task Orchestrator - an "immune system" for Claude Code agents.

Catches semantic failures (hallucinations, not just crashes), learns from mistakes, prevents recurrence.

MIT licensed. 680+ tests. Works with Gemini or OpenAI.

github.com/TC407-api/task-orchestrator
