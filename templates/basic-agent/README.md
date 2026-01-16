# Basic Agent Template

A minimal example demonstrating how to spawn a single Gemini agent.

## Features

- Simple agent spawning with prompt
- Optional system prompt support
- Error handling

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

Edit `main.py` to customize:
- The prompt sent to the agent
- The system prompt for behavior guidance
- The model used (default: gemini-3-flash-preview)

## Next Steps

- Try the `multi-agent-workflow` template for coordinated agents
- Try the `cost-controlled` template for budget management
- Try the `self-healing` template for resilient execution
