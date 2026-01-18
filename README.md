# Task Orchestrator MCP Server

[![CI](https://github.com/TC407-api/task-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/TC407-api/task-orchestrator/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-680%2B-brightgreen.svg)](tests/)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

**Production safety for Claude Code agents** - catches failures before your users do, including hallucinations, not just crashes.

## The Problem

> "Less than 1 in 3 teams are satisfied with their AI agent guardrails and observability" - [Cleanlab AI Agents Report 2025](https://cleanlab.ai/ai-agents-in-production-2025/)

Most AI agents fail silently in production. Task Orchestrator adds an **immune system** to Claude Code that:
- Detects **semantic failures** (hallucinations, wrong answers) not just crashes
- **Learns from mistakes** and prevents the same error twice
- Provides **human-in-the-loop** controls for sensitive operations
- Works with **any LLM provider** (Gemini, OpenAI, or bring your own)

## Features

- **Multi-Model Agent Spawning** - Spawn Gemini agents with automatic model selection
- **Parallel Agent Execution** - Run multiple agents concurrently for complex tasks
- **Self-Healing System** - Circuit breakers, retry logic, and automatic recovery
- **Cost Management** - Real-time budget tracking and limits per provider
- **Task Management** - Create, schedule, and track tasks from email/calendar
- **Evaluation System** - Quality gates, failure detection, and training data export
- **Immune System** - Learn from failures and prevent recurrence
- **Dynamic Tool Loading** - Lazy load tool categories to reduce context window usage (88% reduction)
- **Human-in-the-Loop Controls** - Operation classification for safe, approval-required, and blocked actions

## Quick Start (5 minutes)

```bash
# 1. Clone and install
git clone https://github.com/TC407-api/task-orchestrator.git
cd task-orchestrator && pip install -r requirements.txt

# 2. Configure (add your API key)
cp .env.example .env.local
# Edit .env.local: Add GOOGLE_API_KEY or OPENAI_API_KEY

# 3. Add to Claude Code
claude mcp add task-orchestrator python mcp_server.py

# 4. Restart Claude Code and verify
# Try: mcp__task-orchestrator__healing_status
```

**Need detailed setup?** See [Claude Code Setup Guide](docs/CLAUDE_CODE_SETUP.md)

### LLM Provider Options

Task Orchestrator works with multiple providers:

| Provider | Environment Variable | Notes |
|----------|---------------------|-------|
| **Gemini** (Recommended) | `GOOGLE_API_KEY` | Free tier available, default |
| OpenAI | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini |
| Custom | Implement `LLMProvider` | Any provider you want |

## Installation (Development)

```bash
# Clone repository
git clone https://github.com/TC407-api/task-orchestrator.git
cd task-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env.local
# Edit .env.local with your keys

# Run tests
JWT_SECRET_KEY=test123 python -m pytest tests/ -v
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `spawn_agent` | Spawn a Gemini agent for a task |
| `spawn_parallel_agents` | Run multiple agents concurrently |
| `tasks_list` | List tasks by priority |
| `tasks_add` | Create a new task |
| `tasks_complete` | Mark task as completed |
| `cost_summary` | View API costs across providers |
| `healing_status` | Check self-healing system status |
| `immune_status` | Check immune system health |
| `immune_dashboard` | View failure patterns and trends |
| `request_tool` | Dynamically load tool categories (for context optimization) |

## Evaluation System

The evaluation system provides quality gates for agent outputs, catching semantic failures (not just crashes).

### Components

```
src/
├── mcp/
│   ├── server.py         # MCP server with 41 tools
│   ├── tool_router.py    # Dynamic tool loading
│   └── context_tracker.py# Context window monitoring
├── agents/
│   ├── operation_classifier.py # HITL operation classification
│   ├── inbox.py          # Universal approval queue
│   └── ...
├── evaluation/
│   ├── trial.py          # Trial schema and lifecycle
│   ├── graders/          # Code and model-based validators
│   │   ├── code.py       # JSON, regex, length validators
│   │   └── model.py      # LLM-as-judge graders
│   ├── immune_system/    # Self-learning failure prevention
│   │   ├── core.py       # ImmuneSystem singleton
│   │   ├── failure_store.py  # Pattern storage
│   │   ├── pattern_matcher.py# Similarity matching
│   │   ├── guardrails.py # Prompt protection
│   │   ├── dashboard.py  # Health metrics
│   │   └── federation.py # Cross-project sharing
│   ├── alerting/         # High-risk pattern detection
│   │   ├── manager.py    # Alert coordination
│   │   ├── rules.py      # Alert rule types
│   │   └── notifiers.py  # Console/Webhook/Slack
│   ├── prediction/       # ML-based failure prediction
│   │   ├── features.py   # TF-IDF + meta features
│   │   ├── classifier.py # RandomForest predictor
│   │   └── training.py   # Model training pipeline
│   ├── langfuse_integration.py # Observability tracing
│   └── export.py         # Training data export
└── ...
```

### Graders

**Code-Based Graders:**
- `NonEmptyGrader` - Verify non-empty output
- `JSONValidGrader` - Validate JSON structure
- `JSONSchemaGrader` - Validate against JSON schema
- `RegexGrader` - Pattern matching
- `LengthGrader` - Output length bounds
- `ContainsGrader` / `NotContainsGrader` - Text matching

**Model-Based Graders (LLM-as-judge):**
- `RelevanceGrader` - Is response relevant to prompt?
- `CompletenessGrader` - Does it fully address the request?
- `AccuracyGrader` - Is the information correct?
- `FormatGrader` - Does it follow formatting requirements?
- `CodeQualityGrader` - Code best practices
- `SafetyGrader` - Security vulnerability detection
- `PerformanceGrader` - Algorithm efficiency

### Immune System

The immune system learns from failures and prevents recurrence:

```python
from src.evaluation import get_immune_system

immune = get_immune_system()

# Pre-spawn check (before agent execution)
response = await immune.pre_spawn_check(prompt, "spawn_agent")
if response.should_proceed:
    result = await spawn_agent(response.processed_prompt)

# Record failures for learning
if not result.passed:
    await immune.record_failure(
        operation="spawn_agent",
        prompt=prompt,
        output=result.output,
        grader_results=result.grader_results,
    )
```

### Alerting System

Four alert rules for detecting high-risk patterns:

| Rule | Description |
|------|-------------|
| `HighRiskThreshold` | Triggers when risk score exceeds threshold |
| `FrequencySpike` | Detects unusual failure frequency |
| `NewPatternDetected` | Alerts on new failure patterns |
| `ConsecutiveFailures` | Triggers after N consecutive failures |

Three notification channels:
- `ConsoleNotifier` - Log to console
- `WebhookNotifier` - POST to webhook URL
- `SlackNotifier` - Send to Slack channel

### Cross-Project Federation

Share failure patterns across projects:

```python
from src.evaluation import PatternFederation

federation = PatternFederation(
    graphiti_client=client,
    local_group_id="project_task_orchestrator",
)

# Subscribe to another project's patterns
await federation.subscribe_to_project("project_other")

# Search patterns across subscribed projects
patterns = await federation.search_global_patterns("timeout handling")

# Import a useful pattern with lineage tracking
await federation.import_pattern(pattern_id, source_group_id)
```

### ML Prediction

Proactively predict failures before they occur:

```python
from src.evaluation import FailurePredictor, ModelTrainer

# Train on historical data
trainer = ModelTrainer()
training_results = trainer.train_from_jsonl("training_data.jsonl")

# Use predictor
predictor = FailurePredictor(model_dir="models/")
result = predictor.predict(prompt, tool="spawn_agent")

if result.is_high_risk:
    print(f"Warning: {result.risk_score:.0%} failure probability")
```

## Dynamic Tool Loading

Reduce context window usage by 88% through lazy loading of tool categories. When context is low, only core tools are exposed; other categories are loaded on demand via `request_tool`.

### Tool Categories

| Category | Tools | Use Case |
|----------|-------|----------|
| `core` | tasks_list, tasks_add, spawn_agent, healing_status, request_tool | Always available |
| `task` | tasks_sync_email, tasks_schedule, tasks_complete, tasks_analyze, tasks_briefing | Task management |
| `agent` | spawn_parallel_agents, spawn_archetype_agent, inbox_status, approve_action | Agent coordination |
| `immune` | immune_status, immune_check, immune_failures, immune_dashboard, immune_sync | Health monitoring |
| `federation` | federation_status, federation_subscribe, federation_search, federation_decay | Cross-project patterns |
| `sync` | sync_status, sync_trigger, sync_alerts | Real-time sync |
| `workflow` | trigger_workflow, list_workflows, validate_code, run_with_error_capture | Workflow automation |
| `cost` | cost_summary, cost_set_budget | Budget management |

### Usage

```python
# Load a tool category dynamically
result = await mcp_server.handle_tool_call("request_tool", {
    "category": "immune",
    "reason": "Need to check system health"
})

# Result includes loaded tools
# {"success": true, "tools_loaded": ["immune_status", "immune_check", ...]}
```

### Context Tracking

The `ContextTracker` monitors context window usage:
- Default max: 200,000 tokens
- Threshold: 10% remaining triggers dynamic mode
- Token estimation with caching for tool definitions

### Operation Classification (HITL)

The `OperationClassifier` categorizes operations for human-in-the-loop controls:

| Category | Behavior | Examples |
|----------|----------|----------|
| SAFE | Auto-execute | tasks_list, file_read, cost_summary |
| REQUIRES_APPROVAL | Wait for human | file_delete, email_send, deployment |
| BLOCKED | Never execute | rm -rf /, DROP DATABASE, force push main |

## Self-Healing System

Automatic recovery with circuit breakers and exponential backoff:

```
CLOSED (normal) → 3 failures → OPEN (blocked) → 30s → HALF_OPEN (test) → success → CLOSED
```

Configuration in `src/self_healing.py`:
- Base retry delay: 2 seconds
- Max retry delay: 60 seconds
- Max retries: 5
- Circuit breaker failure threshold: 3

## Cost Management

Track and limit API costs:

```python
# View cost summary
result = await mcp_server.handle_tool_call("cost_summary", {})

# Set budget limits
await mcp_server.handle_tool_call("cost_set_budget", {
    "provider": "google_gemini",
    "daily_limit": 5.00,
    "monthly_limit": 50.00,
})
```

## Testing

```bash
# Run all tests
JWT_SECRET_KEY=test123 python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_evaluation.py -v
python -m pytest tests/test_immune_system.py -v
python -m pytest tests/test_phase8.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

**Test Counts:** 680+ tests across all modules

## CI/CD

GitHub Actions workflow in `.github/workflows/evaluation.yml`:
- Runs on push to master and PRs
- Executes full test suite
- Enforces 70%+ code coverage
- Validates type hints

## Configuration

Environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `JWT_SECRET_KEY` | Secret for JWT tokens | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `LANGFUSE_SECRET_KEY` | Langfuse observability | No |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | No |
| `LANGFUSE_HOST` | Langfuse server URL | No (default: localhost:3000) |

## Why Task Orchestrator?

| Feature | Task Orchestrator | LangGraph | CrewAI | AutoGen |
|---------|------------------|-----------|--------|---------|
| Semantic failure detection | **Yes** | No | No | No |
| ML-powered learning | **Yes** | No | No | No |
| Cross-project federation | **Yes** | No | No | No |
| MCP native | **Yes** | No | No | No |
| Human-in-the-loop | **Yes** | Partial | Partial | Partial |
| Cost tracking | **Yes** | No | Enterprise | No |
| Self-healing | **Yes** | No | No | No |
| Multi-provider LLM | **Yes** | Partial | Partial | Yes |
| TTT Memory (O(1) lookup) | **Yes** | No | No | No |

**Key differentiator:** Task Orchestrator catches **semantic failures** (hallucinations, wrong answers) using an immune system that learns from mistakes - not just crashes and exceptions.

## Documentation

| Document | Description |
|----------|-------------|
| [Claude Code Setup](docs/CLAUDE_CODE_SETUP.md) | Quick start guide for Claude Code users |
| [Phase 10 Observability](docs/phase10-observability.md) | Langfuse + Graphiti integration architecture |
| [PRD: Eval System](docs/PRD-eval-system.md) | Product requirements for evaluation system |

## Architecture

```
                         Claude Code
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Graphiti   │   │    Task     │   │   Memory    │
    │ MCP Server  │   │Orchestrator │   │ MCP Server  │
    │             │   │ MCP Server  │   │             │
    └──────┬──────┘   └──────┬──────┘   └─────────────┘
           │                 │
           ▼                 ▼
    ┌──────────────┐  ┌─────────────────────────────────┐
    │    Neo4j     │  │        Evaluation System        │
    │  Graph DB    │  │  Graders │ Immune │ Alerting   │
    │  :7687       │  │          │ System │            │
    └──────────────┘  └──────────────┬──────────────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │                             │
                      ▼                             ▼
               ┌─────────────┐              ┌─────────────┐
               │  Langfuse   │              │ Self-Healing│
               │   (SDK)     │              │   System    │
               │ :3000       │              │             │
               └─────────────┘              └──────┬──────┘
                                                   │
                                                   ▼
                                            ┌─────────────┐
                                            │ Gemini API  │
                                            └─────────────┘
```

**Integration Methods:**
- **Langfuse**: Python SDK with `@trace_*` decorators (automatic tracing)
- **Graphiti**: MCP tools from Claude Code (pattern storage/retrieval)

See [Phase 10 Observability](docs/phase10-observability.md) for detailed architecture.

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- Built with [MCP Protocol](https://modelcontextprotocol.io/)
- Observability via [Langfuse](https://langfuse.com/)
- Knowledge graph via [Graphiti](https://github.com/getzep/graphiti)
