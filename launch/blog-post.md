# Why Your AI Agents Fail in Production (And How to Fix It)

**TL;DR:** I built an open-source MCP server that adds an "immune system" to Claude Code agents. It catches semantic failures (hallucinations, wrong answers) not just crashes, learns from mistakes, and prevents recurrence. MIT licensed, 680+ tests, works with Gemini or OpenAI.

**GitHub:** https://github.com/TC407-api/task-orchestrator

---

## The Problem Nobody Talks About

Here's a stat that should terrify anyone deploying AI agents:

> "Less than 1 in 3 teams are satisfied with their AI agent guardrails and observability" - [Cleanlab AI Agents Report 2025](https://cleanlab.ai/ai-agents-in-production-2025/)

And it gets worse. Only 5% of companies have AI agents in production. Why? Because when an agent fails, it doesn't crash with a nice stack trace. It confidently gives you the wrong answer and moves on.

I've been building with Claude Code for months now. It's incredible for development velocity. But here's what I noticed:

- Agents would hallucinate file paths that don't exist
- They'd suggest fixes that introduce new bugs
- They'd confidently claim "tests pass" when they didn't run them
- Same errors would happen again and again across sessions

The tools exist to catch crashes. Nothing exists to catch *semantic* failures.

## What I Built

**Task Orchestrator** is an MCP server that adds production safety to Claude Code agents. Think of it as an immune system for your AI workflows.

### Key Features

**1. Semantic Failure Detection**

Not just "did it crash?" but "did it actually do the right thing?"

```
mcp__task-orchestrator__immune_check
> Analyzing prompt for risk patterns...
> Risk Score: 0.73 (HIGH)
> Detected: File path hallucination pattern
> Suggestion: Verify paths before operations
```

**2. ML-Powered Learning**

The system learns from failures. When an agent makes a mistake, that pattern gets stored. Next time a similar prompt comes in, you get a warning *before* execution.

```
mcp__task-orchestrator__immune_dashboard
> Failure Patterns: 23 stored
> Prevention Rate: 87%
> Last 24h: 3 blocked, 45 passed
```

**3. Human-in-the-Loop Controls**

Some operations shouldn't be automated. The inbox system queues high-risk actions for human approval:

```
mcp__task-orchestrator__inbox_status
> Pending Approvals: 2
> - [HIGH] Delete production database backup
> - [MEDIUM] Push to main branch
```

**4. Cost Tracking Across Providers**

Know exactly what you're spending:

```
mcp__task-orchestrator__cost_summary
> Today: $0.45 (Gemini: $0.32, OpenAI: $0.13)
> This Month: $12.30
> Budget Remaining: $37.70
```

**5. Self-Healing Circuit Breakers**

When external services fail, the system backs off automatically instead of hammering APIs:

```
CLOSED (normal) -> 3 failures -> OPEN (blocked) -> 30s -> HALF_OPEN (test) -> success -> CLOSED
```

## Why This Matters

At 95% reliability per step, a 20-step agent workflow has only a **36% success rate** overall. That's not a bug - it's math.

The only way to make AI agents production-ready is to:
1. Detect failures *semantically*, not just crashes
2. Learn from every failure
3. Prevent the same failure from happening twice
4. Keep humans in the loop for high-stakes decisions

## Technical Details

- **680+ tests** - This isn't a prototype
- **Multi-provider** - Works with Gemini (free tier!), OpenAI, or bring your own
- **MCP native** - Designed specifically for Claude Code
- **Plugin architecture** - Free core, optional enterprise features
- **MIT licensed** - Use it however you want

### Architecture

```
src/
├── mcp/           # MCP server and plugin system
├── evaluation/    # Graders, immune system, alerting
├── llm/           # Multi-provider abstraction (Gemini, OpenAI, custom)
├── governance/    # TTT memory, cost tracking
├── agents/        # Agent archetypes and coordination
└── self_healing/  # Circuit breakers, retry logic
```

## Getting Started

```bash
# Clone and install (2 minutes)
git clone https://github.com/TC407-api/task-orchestrator.git
cd task-orchestrator && pip install -r requirements.txt

# Configure (add your API key)
cp .env.example .env.local
# Edit .env.local: Add GOOGLE_API_KEY or OPENAI_API_KEY

# Add to Claude Code
claude mcp add task-orchestrator python mcp_server.py
```

That's it. Restart Claude Code and you have production safety.

## What's Next

The core is free and always will be. I'm working on enterprise features (cross-project pattern sharing, content automation) for teams that need them.

But more importantly, I want feedback:
- What failures are you seeing with AI agents?
- What guardrails do you wish existed?
- What would make this useful for your workflow?

**GitHub:** https://github.com/TC407-api/task-orchestrator

**Star it** if you think AI agents need better safety infrastructure.

---

*Built by someone who got tired of AI agents failing silently.*
