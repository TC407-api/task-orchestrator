# Setting Up Task Orchestrator with Claude Code

This guide walks you through adding Task Orchestrator to your Claude Code setup. **No complex hooks required** - the MCP server works standalone.

## Prerequisites

- Python 3.10 or higher
- Claude Code CLI installed
- **At least one LLM API key**:
  - `GOOGLE_API_KEY` - Gemini (**Recommended**: Free tier available)
  - `OPENAI_API_KEY` - GPT-4o, GPT-4o-mini (alternative)

## Quick Start (4 Commands)

```bash
# 1. Clone the repository
git clone https://github.com/TC407-api/task-orchestrator.git

# 2. Install dependencies
cd task-orchestrator && pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env.local
# Edit .env.local and add your API keys (see below)

# 4. Add to Claude Code
claude mcp add task-orchestrator python mcp_server.py
```

## Step-by-Step Setup

### Step 1: Clone and Install

```bash
git clone https://github.com/TC407-api/task-orchestrator.git
cd task-orchestrator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

```bash
# Create local config from template
cp .env.example .env.local
```

Edit `.env.local` and add your keys:

```bash
# REQUIRED: Generate a JWT secret
# Run: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=your-generated-secret-here

# REQUIRED: At least ONE LLM provider key
# Option A: Gemini (Recommended - has free tier)
GOOGLE_API_KEY=your-gemini-api-key

# Option B: OpenAI (alternative)
OPENAI_API_KEY=your-openai-api-key

# OPTIONAL: Langfuse for observability
# LANGFUSE_PUBLIC_KEY=your-key
# LANGFUSE_SECRET_KEY=your-key
```

**Getting API Keys:**
- Gemini: https://aistudio.google.com/apikey (free tier available)
- OpenAI: https://platform.openai.com/api-keys

### Step 3: Add MCP Server to Claude Code

**Option A: Using CLI (Recommended)**

```bash
# Run from the task-orchestrator directory
claude mcp add task-orchestrator python mcp_server.py
```

**Option B: Manual Configuration**

Add to your Claude Code MCP configuration file:

```json
{
  "mcpServers": {
    "task-orchestrator": {
      "command": "python",
      "args": ["/absolute/path/to/task-orchestrator/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/task-orchestrator"
      }
    }
  }
}
```

### Step 4: Verify Installation

Restart Claude Code, then test these commands:

```
# Check if the server is responding
mcp__task-orchestrator__healing_status

# Check immune system
mcp__task-orchestrator__immune_status
```

Or run the verification script:

```bash
python scripts/verify_install.py
```

## Available Tools

Once installed, you'll have access to these MCP tools:

### Core Tools (Always Available)
| Tool | Description |
|------|-------------|
| `spawn_agent` | Spawn an AI agent safely |
| `spawn_parallel_agents` | Run multiple agents concurrently |
| `tasks_list` | List tasks by priority |
| `tasks_add` | Create a new task |
| `healing_status` | Check self-healing system |
| `request_tool` | Load additional tool categories |

### Immune System Tools
| Tool | Description |
|------|-------------|
| `immune_status` | Check immune system health |
| `immune_check` | Pre-check a prompt for risks |
| `immune_failures` | List recent failure patterns |
| `immune_dashboard` | View health metrics |

### Cost Management
| Tool | Description |
|------|-------------|
| `cost_summary` | View API costs |
| `cost_set_budget` | Set daily/monthly limits |

### Human-in-the-Loop
| Tool | Description |
|------|-------------|
| `inbox_status` | View pending approvals |
| `approve_action` | Approve or reject actions |

## Multi-Provider Support

Task Orchestrator supports multiple LLM providers. The `ModelRouter` automatically selects the best model based on task type:

| Task Type | Default Provider | Model |
|-----------|------------------|-------|
| Fast tasks | Gemini | gemini-2.5-flash |
| Complex reasoning | Gemini | gemini-3-pro-preview |
| Code generation | Gemini | gemini-3-flash-preview |
| Fallback | OpenAI | gpt-4o-mini |

To use a specific provider:

```python
# The spawn_agent tool accepts a model parameter
mcp__task-orchestrator__spawn_agent(
    prompt="Your task here",
    model="gemini-3-flash-preview"  # or "gpt-4o"
)
```

## Troubleshooting

### "Module not found" errors

Ensure PYTHONPATH is set correctly in your MCP config:

```json
{
  "env": {
    "PYTHONPATH": "/absolute/path/to/task-orchestrator"
  }
}
```

### "API key not found" errors

1. Check that `.env.local` exists and has your keys
2. Verify the file is in the project root directory
3. Make sure there are no typos in key names

### Server not responding

1. Check if Python is available: `python --version`
2. Test the server directly: `python mcp_server.py`
3. Check for import errors in the output

### Context window issues

Task Orchestrator includes dynamic tool loading to reduce context usage:

```python
# Load only the tools you need
mcp__task-orchestrator__request_tool(
    category="immune",
    reason="Need to check health"
)
```

## Optional: Advanced Configuration

### Langfuse Observability

For production tracing, add Langfuse credentials to `.env.local`:

```bash
LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

### Custom Hooks (Optional)

Task Orchestrator works without hooks, but advanced users can add:
- Pre/post verification hooks
- Automatic cost tracking
- Custom approval workflows

See `examples/hooks/` for examples.

## Need Help?

- [GitHub Issues](https://github.com/TC407-api/task-orchestrator/issues)
- [Documentation](https://github.com/TC407-api/task-orchestrator/tree/main/docs)
- [Contributing Guide](../CONTRIBUTING.md)
