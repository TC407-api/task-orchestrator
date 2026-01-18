# Reddit Posts

---

## r/ClaudeAI

### Title
I built an open-source MCP server that adds "production safety" to Claude Code - catches hallucinations, learns from mistakes [MIT License]

### Body
Hey r/ClaudeAI,

I've been using Claude Code heavily for the past few months and kept running into the same problems:

- Agent hallucinates file paths that don't exist
- Claims "tests pass" without actually running them
- Same errors keep happening across sessions
- No way to catch failures that aren't crashes

So I built **Task Orchestrator** - an MCP server that adds an "immune system" to Claude Code.

**What it does:**
- **Semantic failure detection** - catches hallucinations, not just crashes
- **ML-powered learning** - remembers failure patterns and warns before similar prompts
- **Human-in-the-loop** - queues high-risk operations for approval
- **Cost tracking** - see exactly what you're spending on Gemini/OpenAI
- **Self-healing** - circuit breakers that back off when things fail

**Quick start:**
```bash
git clone https://github.com/TC407-api/task-orchestrator.git
cd task-orchestrator && pip install -r requirements.txt
claude mcp add task-orchestrator python mcp_server.py
```

Works with Gemini (free tier) or OpenAI. MIT licensed, 680+ tests.

**GitHub:** https://github.com/TC407-api/task-orchestrator

Would love feedback on what guardrails you wish existed for Claude Code. What failures are you hitting?

---

## r/MachineLearning

### Title
[P] Task Orchestrator: Open-source semantic failure detection for LLM agents (immune system approach, 680+ tests, MIT)

### Body
**GitHub:** https://github.com/TC407-api/task-orchestrator

**Problem:** LLM agents fail silently. They don't crash - they confidently produce wrong outputs. At 95% per-step reliability, a 20-step workflow has 36% overall success rate.

**Solution:** An "immune system" that detects semantic failures, learns from them, and prevents recurrence.

**Key technical details:**

1. **Semantic Graders** - Custom evaluation functions that check output quality beyond "did it run"
2. **TTT Memory Layer** - O(1) risk prediction using test-time training concept (pattern matching stays constant-time regardless of failure count)
3. **Failure Store** - Persistent storage of failure patterns with signature matching
4. **Circuit Breakers** - Self-healing with exponential backoff (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)

**Architecture:**
```
src/
├── evaluation/      # Graders, immune system
│   ├── graders/     # JSONSchema, Regex, Semantic, LLM-as-judge
│   └── immune_system/
│       ├── core.py          # Main immune system logic
│       ├── failure_store.py # Pattern persistence
│       └── guardrails.py    # Pre-execution checks
├── llm/             # Multi-provider (Gemini, OpenAI, custom)
├── governance/      # TTT memory, cost tracking
└── self_healing/    # Circuit breakers, retry logic
```

**Stats:**
- 680+ tests
- Multi-provider support (Gemini free tier, OpenAI)
- MCP (Model Context Protocol) native for Claude Code integration

Looking for feedback on the approach. The "immune system" metaphor maps surprisingly well - detect, remember, prevent.

---

## r/LocalLLaMA

### Title
Open-sourced my agent safety infrastructure - works with any LLM provider (MIT license)

### Body
Built Task Orchestrator as an MCP server for Claude Code, but the core concepts work with any LLM.

**The problem I was solving:** Agents fail silently. They don't crash - they just give wrong answers confidently. Traditional error handling doesn't help because there's no error to catch.

**Key features:**
- Semantic failure detection (graders that check output quality)
- Pattern learning (remembers failures, warns on similar inputs)
- Human-in-the-loop for high-risk ops
- Circuit breakers for external service failures
- Cost tracking across providers

**Multi-provider support:**
The LLM abstraction (`src/llm/`) is provider-agnostic. Currently supports:
- Gemini (default, has free tier)
- OpenAI

But the `LLMProvider` interface is simple to implement for any provider:

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass
```

Would work with Ollama, llama.cpp, vLLM, etc. with a simple wrapper.

**GitHub:** https://github.com/TC407-api/task-orchestrator

MIT licensed, 680+ tests. Contributions welcome if you want to add your preferred provider.
