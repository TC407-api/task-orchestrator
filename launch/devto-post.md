---
title: Why Your AI Agents Fail in Production (And How to Fix It)
published: true
description: I built an open-source MCP server that adds an immune system to Claude Code agents. Catches hallucinations, learns from mistakes, prevents recurrence.
tags: ai, opensource, python, productivity
cover_image: # Add a cover image URL if you have one
canonical_url: https://github.com/TC407-api/task-orchestrator
---

## TL;DR

I built **Task Orchestrator**, an open-source MCP server that adds production safety to Claude Code agents. It catches semantic failures (hallucinations, wrong answers) not just crashes, learns from mistakes, and prevents recurrence.

**GitHub:** [github.com/TC407-api/task-orchestrator](https://github.com/TC407-api/task-orchestrator)

- MIT licensed
- 680+ tests
- Works with Gemini (free tier!) or OpenAI

---

## The Problem

Here's a stat that should terrify anyone deploying AI agents:

> "Less than 1 in 3 teams are satisfied with their AI agent guardrails and observability" - Cleanlab AI Agents Report 2025

I've been building with Claude Code for months. It's incredible for development velocity. But here's what I noticed:

- Agents hallucinate file paths that don't exist
- They suggest fixes that introduce new bugs
- They claim "tests pass" without running them
- Same errors happen again and again

**The tools exist to catch crashes. Nothing exists to catch *semantic* failures.**

## The Math Problem

At 95% reliability per step, a 20-step agent workflow has only a **36% success rate** overall.

```
0.95^20 = 0.358 = 35.8%
```

That's not a bug - it's compound probability. Every step that could fail, will eventually fail.

## What I Built

**Task Orchestrator** is an MCP server that adds an immune system to Claude Code:

### 1. Semantic Failure Detection

Not "did it crash?" but "did it actually do the right thing?"

```python
mcp__task-orchestrator__immune_check(prompt="...")
# Risk Score: 0.73 (HIGH)
# Detected: File path hallucination pattern
# Suggestion: Verify paths before operations
```

### 2. ML-Powered Learning

The system learns from failures. Pattern stored -> warning before similar prompts.

```python
mcp__task-orchestrator__immune_dashboard
# Failure Patterns: 23 stored
# Prevention Rate: 87%
# Last 24h: 3 blocked, 45 passed
```

### 3. Human-in-the-Loop

High-risk operations queue for human approval:

```python
mcp__task-orchestrator__inbox_status
# Pending: Delete production database backup [HIGH]
```

### 4. Cost Tracking

Know what you're spending across providers:

```python
mcp__task-orchestrator__cost_summary
# Today: $0.45 (Gemini: $0.32, OpenAI: $0.13)
```

### 5. Self-Healing

Circuit breakers that back off automatically:

```
CLOSED -> 3 failures -> OPEN -> 30s -> HALF_OPEN -> success -> CLOSED
```

## Getting Started

```bash
# Clone and install
git clone https://github.com/TC407-api/task-orchestrator.git
cd task-orchestrator && pip install -r requirements.txt

# Configure
cp .env.example .env.local
# Add GOOGLE_API_KEY or OPENAI_API_KEY

# Add to Claude Code
claude mcp add task-orchestrator python mcp_server.py
```

Restart Claude Code. Done.

## Architecture

```
src/
├── mcp/           # MCP server and plugins
├── evaluation/    # Graders, immune system
├── llm/           # Gemini, OpenAI, custom providers
├── governance/    # Cost tracking, TTT memory
└── self_healing/  # Circuit breakers
```

680+ tests. Not a prototype.

## What's Next

Core is free forever. Working on enterprise features for teams.

**I want your feedback:**
- What failures do you see with AI agents?
- What guardrails do you wish existed?

**GitHub:** [github.com/TC407-api/task-orchestrator](https://github.com/TC407-api/task-orchestrator)

Star if you think AI agents need better safety.

---

*Built by someone tired of AI agents failing silently.*
