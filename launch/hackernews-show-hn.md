# Hacker News - Show HN

## Title (80 chars max)
Show HN: Task Orchestrator - Production safety for Claude Code agents (MIT)

## URL
https://github.com/TC407-api/task-orchestrator

## Text (if self-post, otherwise leave for comments)

I built an MCP server that adds an "immune system" to Claude Code agents.

**The problem:** AI agents fail silently. They don't crash - they confidently give wrong answers. At 95% per-step reliability, a 20-step workflow has 36% overall success rate.

**The solution:** Semantic failure detection that catches hallucinations, not just crashes. The system learns from mistakes and prevents recurrence.

Key features:
- Immune system that learns failure patterns and blocks similar prompts
- ML-powered risk prediction (O(1) lookup, not similarity search)
- Human-in-the-loop controls for high-risk operations
- Cost tracking across providers (Gemini, OpenAI)
- Self-healing circuit breakers
- 680+ tests, MIT licensed

Works with Gemini (free tier) or OpenAI. Designed for Claude Code but the patterns apply to any agent framework.

Quick start:
```
git clone https://github.com/TC407-api/task-orchestrator.git
cd task-orchestrator && pip install -r requirements.txt
claude mcp add task-orchestrator python mcp_server.py
```

Looking for feedback on what guardrails you wish existed for AI agents.

---

## First Comment (post immediately after)

Some context on why I built this:

I've been using Claude Code heavily for months. It's incredible for velocity, but I kept hitting the same problems:

1. Agents hallucinating file paths
2. "Tests pass" claims without running tests
3. Same errors recurring across sessions
4. No way to catch semantic failures

The Cleanlab report says <1 in 3 teams are satisfied with their AI guardrails. That matched my experience.

The immune system approach is inspired by how biological systems work - detect, remember, prevent. Each failure becomes training data for future prevention.

Technical note: The risk prediction uses a TTT (Test-Time Training) memory layer for O(1) lookups instead of O(N) similarity search. Pattern matching stays constant-time regardless of how many failures you've stored.

Happy to answer questions about the architecture or specific failure patterns I've encountered.
