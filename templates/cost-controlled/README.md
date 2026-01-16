# Cost-Controlled Agent Template

Demonstrates budget management and cost tracking for AI agent operations.

## Features

- Session and daily budget limits
- Per-call cost tracking
- Automatic budget enforcement
- Cost summaries and reporting

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY=your_api_key

# Run
python main.py
```

## Configuration

Edit the `BudgetTracker` initialization:

```python
budget_tracker = BudgetTracker(
    daily_limit_usd=5.0,    # $5/day max
    session_limit_usd=1.0,  # $1/session max
)
```

## Pricing

Default pricing (per 1M tokens):

| Model | Input | Output |
|-------|-------|--------|
| gemini-3-flash-preview | $0.075 | $0.30 |
| gemini-3-pro-preview | $1.25 | $5.00 |

## Next Steps

- Combine with `self-healing` template for resilient budget enforcement
- Add alerting when approaching limits
- Implement per-user budget tracking
