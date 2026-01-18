# Contributing to Task Orchestrator

Thank you for your interest in contributing to Task Orchestrator! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

1. **Check existing issues** - Search the [issue tracker](https://github.com/TC407-api/task-orchestrator/issues) to avoid duplicates.
2. **Create a detailed report** including:
   - Python version and OS
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the use case and expected behavior
3. Explain why this would benefit other users

### Pull Requests

1. **Fork the repository** and create a feature branch from `main`
2. **Write tests** for new functionality (we aim for 80%+ coverage)
3. **Follow code style** - we use `ruff` for linting and `pyright` for type checking
4. **Update documentation** if needed
5. **Submit a PR** with a clear description of changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/task-orchestrator.git
cd task-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Set up environment
cp .env.example .env.local
# Edit .env.local with your API keys

# Run tests
JWT_SECRET_KEY=test123 python -m pytest tests/ -v
```

## Code Style

- **Type hints**: All functions should have type annotations
- **Docstrings**: Use Google-style docstrings for public functions
- **Line length**: 100 characters max
- **Imports**: Use absolute imports, sorted with `isort`

### Before Submitting

```bash
# Run linter
ruff check src/ tests/

# Run type checker
pyright src/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Architecture Overview

```
src/
├── mcp/           # MCP server and tool definitions
├── evaluation/    # Graders, immune system, alerting
├── llm/           # Multi-provider LLM abstraction
├── governance/    # TTT memory, compliance, cost tracking
├── agents/        # Agent archetypes and coordination
└── core/          # Config, auth, rate limiting
```

Key design principles:
- **Provider-agnostic**: LLM providers implement the `LLMProvider` ABC
- **Plugin architecture**: Tools are organized into loadable categories
- **Self-healing**: Circuit breakers and retry logic throughout
- **Observable**: Langfuse integration for tracing

## Test Categories

| Directory | Purpose |
|-----------|---------|
| `tests/test_evaluation.py` | Graders and trial lifecycle |
| `tests/test_immune_system.py` | Failure detection and prevention |
| `tests/test_mcp_*.py` | MCP server and tools |
| `tests/test_llm.py` | Provider implementations |

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing docs in the `docs/` directory

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
