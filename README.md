# Task Orchestrator MCP Server

A production-grade MCP (Model Context Protocol) server for orchestrating AI agents with comprehensive evaluation, self-healing, and cost management capabilities.

## Features

- **Multi-Model Agent Spawning** - Spawn Gemini agents with automatic model selection
- **Parallel Agent Execution** - Run multiple agents concurrently for complex tasks
- **Self-Healing System** - Circuit breakers, retry logic, and automatic recovery
- **Cost Management** - Real-time budget tracking and limits per provider
- **Task Management** - Create, schedule, and track tasks from email/calendar
- **Evaluation System** - Quality gates, failure detection, and training data export
- **Immune System** - Learn from failures and prevent recurrence

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/task-orchestrator.git
cd task-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export JWT_SECRET_KEY=your-secret-key
export GEMINI_API_KEY=your-gemini-key
```

## Quick Start

```bash
# Run the MCP server
python -m src.mcp.server

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

## Evaluation System

The evaluation system provides quality gates for agent outputs, catching semantic failures (not just crashes).

### Components

```
src/evaluation/
├── trial.py              # Trial schema and lifecycle
├── graders/              # Code and model-based validators
│   ├── code.py           # JSON, regex, length validators
│   └── model.py          # LLM-as-judge graders
├── immune_system/        # Self-learning failure prevention
│   ├── core.py           # ImmuneSystem singleton
│   ├── failure_store.py  # Pattern storage
│   ├── pattern_matcher.py# Similarity matching
│   ├── guardrails.py     # Prompt protection
│   ├── dashboard.py      # Health metrics
│   └── federation.py     # Cross-project sharing
├── alerting/             # High-risk pattern detection
│   ├── manager.py        # Alert coordination
│   ├── rules.py          # Alert rule types
│   └── notifiers.py      # Console/Webhook/Slack
├── prediction/           # ML-based failure prediction
│   ├── features.py       # TF-IDF + meta features
│   ├── classifier.py     # RandomForest predictor
│   └── training.py       # Model training pipeline
├── langfuse_integration.py # Observability tracing
└── export.py             # Training data export
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

**Test Counts:**
- Evaluation tests: 37
- Immune system tests: 22
- CI/CD integration tests: 19
- Phase 8 tests: 25
- **Total: 103 tests**

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
| `GRAPHITI_URI` | Neo4j connection for Graphiti | No |

## Architecture

```
Agent Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    MCP Server                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ spawn_agent │  │  parallel   │  │ task management │  │
│  │   handler   │  │   agents    │  │     tools       │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                   │           │
│         ▼                ▼                   ▼           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Evaluation System                   │    │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────────────┐ │    │
│  │  │ Graders │  │ Immune  │  │    Alerting      │ │    │
│  │  │         │  │ System  │  │                  │ │    │
│  │  └────┬────┘  └────┬────┘  └────────┬─────────┘ │    │
│  │       │            │                 │           │    │
│  │       ▼            ▼                 ▼           │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │         Langfuse Observability          │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
│                          │                               │
│  ┌───────────────────────┴───────────────────────────┐  │
│  │              Self-Healing System                   │  │
│  │  Circuit Breaker │ Retry Logic │ Cost Management   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                   Gemini API / Graphiti
```

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
