# Multi-Agent Workflow Template

Demonstrates coordinated multi-agent execution using the architect → builder → QC pattern.

## Features

- Sequential agent pipeline with context passing
- Role-based system prompts
- Structured result tracking
- Error handling at each stage

## Workflow

```
[ARCHITECT] → Plans implementation
     ↓
[BUILDER]  → Implements code
     ↓
[QC]       → Validates and reviews
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY=your_api_key

# Run
python main.py
```

## Customization

- Modify `spawn_agent()` to change agent behavior
- Add more roles (e.g., TESTER, REVIEWER)
- Implement parallel execution for independent tasks
- Add retry logic for failed stages

## Next Steps

- Add the `self-healing` template for automatic retries
- Add the `cost-controlled` template for budget limits
