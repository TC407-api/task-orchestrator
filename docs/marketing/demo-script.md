# Demo Video Script

**Title:** Building a Self-Healing Agent in 3 Minutes
**Duration:** 180 Seconds
**Target Audience:** Technical Lead / Senior Developer

---

## Script Breakdown

### Opening (0:00 - 0:20)

**Visual:** Animation of a "Failed" agent log scrolling rapidly, error messages accumulating.

**Narration:**
> "We've all been there. You build a great agent, deploy it, and wake up to a $200 bill and a stack of 'API Timeout' errors. The 95% failure rate of AI agents is real."

**Key Points:**
- Establish the pain point
- Relatable scenario
- Create urgency

---

### Introduction (0:20 - 0:45)

**Visual:** Transition to Task-Orchestrator Dashboard. Show the 44 MCP tools list scrolling.

**Narration:**
> "Meet Task-Orchestrator. It's the first MCP-native platform that treats agents like production software, not science experiments. With 44 built-in tools, you're ready to go in seconds."

**Key Points:**
- Introduce the solution
- Highlight MCP-native advantage
- Show scale of tool library

---

### Configuration (0:45 - 1:15)

**Visual:** Code editor showing a simple agent definition:
```python
agent = spawn_archetype_agent(
    archetype="builder",
    budget_limit=5.00,
    retry_policy="exponential"
)
```

**Narration:**
> "Configuration is simple. Define your budget, set your archetypes—like Architect or QC—and let our self-healing infrastructure handle the rest. No more manual 'try-except' blocks for your LLM calls."

**Key Points:**
- Simplicity of setup
- Budget controls visible
- Archetype concept introduced

---

### The Magic Moment (1:15 - 1:45)

**Visual:** A tool call fails (simulated). Show the "Circuit Breaker" tripping animation. The "Immune System" panel lights up, showing pattern detection.

**Narration:**
> "Watch what happens when a tool fails. Instead of looping infinitely, the circuit breaker trips. The Agent Immune System analyzes the failure, adapts the prompt, and retries successfully. That's self-healing in action."

**Key Points:**
- **This is the demo's climax**
- Show failure → recovery flow
- Visualize the immune system working

---

### Observability & Federation (1:45 - 2:15)

**Visual:** Split screen - Langfuse integration on left, Federation pattern sharing on right.

**Narration:**
> "With Langfuse baked in, you get total observability. And with Federation, your agents share what they learn across your whole team, building a collective intelligence for your company."

**Key Points:**
- Observability integration
- Cross-project learning
- Team-wide benefits

---

### Archetypes in Action (2:15 - 2:45)

**Visual:** Rapid-fire shots of the 4 Archetypes icons with brief workflow animations:
- Architect: Blueprint drawing
- Builder: Code writing
- QC: Checkmark verification
- Researcher: Document searching

**Narration:**
> "Architects design. Builders code. QC verifies. Researchers find facts. Task-Orchestrator makes them work as a high-performance team."

**Key Points:**
- Showcase specialization
- Quick visual impact
- Team metaphor

---

### Closing CTA (2:45 - 3:00)

**Visual:** Task-Orchestrator logo centered, URL below, "Start Building Reliable Agents Today" tagline.

**Narration:**
> "Stop the failure rate. Start building for production. Get started with Task-Orchestrator today."

**Key Points:**
- Clear call to action
- Reinforce main message
- Easy next step

---

## Technical Demo Points

During the demo, highlight these technical features:

1. **Circuit Breaker States:**
   - CLOSED → OPEN → HALF_OPEN → CLOSED

2. **Cost Tracking:**
   - Real-time token usage
   - Budget enforcement

3. **Archetype System:**
   - Tool filtering per role
   - System prompt optimization

4. **Immune System:**
   - Pattern detection
   - Automatic adaptation

---

## Production Notes

- **Music:** Upbeat, tech-forward background track
- **Pacing:** Fast cuts during archetype section, slower during magic moment
- **Graphics:** Clean, minimalist design consistent with developer tools
- **Tone:** Confident, technical but accessible
