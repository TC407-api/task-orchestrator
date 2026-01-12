# Product Requirement Document: Task-Orchestrator Evaluation System

**Status:** Active
**Date:** 2026-01-12
**Owner:** Engineering Team

## 1. Problem Statement

Current reliability mechanisms in the task-orchestrator (specifically Circuit Breakers) are designed to detect infrastructure failures (timeouts, crashes, API errors). However, they lack the context to detect **semantic failures**.

A model returning a well-formatted JSON response that contains hallucinated data, biased reasoning, or unsafe content is currently treated as a "Success" by the circuit breaker. We need a dedicated evaluation layer to assess the *quality* of the output, not just the *availability* of the system.

**Key Insight:** Circuit breakers detect crashes, NOT semantic failures. An agent can return `{"success": true}` with hallucinated output.

## 2. Goals

1. **Semantic Reliability:** Catch 80% of logic/safety failures in pre-production or staging environments.
2. **Debug Efficiency:** Reduce time-to-root-cause for AI logic errors by 40-60% using detailed trace artifacts.
3. **Observability:** Correlate cost and latency with output quality scores via Langfuse.
4. **Training Data:** Generate labeled datasets for fine-tuning from production evaluations.
5. **Guardrails:** Prevent regression by blocking deployments that fail critical evaluation trials.

## 3. Architecture Overview

### 3.1 Core Components

* **Trial Object:** A standardized data structure that wraps an agent execution. Captures inputs, outputs, model metadata, costs, and evaluation results.
* **Graders:** Specialized functions (heuristic or LLM-based) that accept a Trial and return a GraderResult (Pass/Fail + Score + Reasoning).
* **Langfuse Integration:** Push all Trial data to Langfuse for visualization, dataset management, and historical tracking.
* **Training Data Export:** Export labeled trials to D:\Research\training-data\ for fine-tuning.

### 3.2 Workflow

```
1. Orchestration: The agent performs a task
2. Encapsulation: The result is wrapped in a Trial object
3. Grading: Graders run against the Trial (code-based, then model-based)
4. Aggregation: Results aggregated; Trial marked Pass/Fail
5. Telemetry: Data pushed to Langfuse Scores API
6. Export: Labeled trials exported for training (async)
```

### 3.3 Design Decisions

1. **Grader Scope:** Code graders only in Phase 1 (JSON, regex, assertions). Model-based graders deferred to Phase 3.
2. **Failure Mode:** Log-only (non-blocking). Evaluation failures recorded in Langfuse but don't block agent responses.
3. **Location:** Core eval module in task-orchestrator. Training data exports to D: drive.

## 4. Success Metrics

* **Precision:** >90% accuracy in automated graders (low false positives)
* **Latency Overhead:** Evaluation logic adds <50ms to the synchronous request path
* **Coverage:** 100% of critical user paths have at least one deterministic grader
* **Training Data:** 1000+ labeled examples exported per week

## 5. Non-Goals

* **Real-time Blocking:** Not building an inline firewall for this phase. Evaluations are non-blocking.
* **Unit Test Replacement:** This evaluates stochastic AI behavior, not deterministic code logic.
* **Replacing Langfuse:** We extend Langfuse with Scores API, not replace it.
* **Testing Gemini itself:** We test our scaffold, not the model.

## 6. Timeline

### Phase 1: Foundation (30 Days)
* Define Trial and GraderResult schemas
* Implement basic heuristic graders (JSON validity, Regex matching, Length checks)
* Setup Langfuse Scores API connection
* Create training data export pipeline

### Phase 2: Advanced Evaluation (60 Days)
* Implement Resilience Suite (fault injection testing)
* Create Golden Dataset of 50 curated examples
* CI/CD gating with Promptfoo integration
* Semantic failure tracking in circuit breakers

### Phase 3: Intelligence (90 Days)
* Implement LLM-as-a-Judge graders
* Graphiti "Immune System Memory" integration
* Shadow Agent (Critic) pattern
* Automated failure → learning pipeline

## 7. Module Structure

```
src/evaluation/
├── __init__.py           # Module exports
├── trial.py              # Trial and GraderResult schemas
├── integration.py        # Langfuse Scores API wrapper
├── export.py             # Training data exporter
├── graders/
│   ├── __init__.py       # Grader exports
│   ├── base.py           # Grader ABC, GraderPipeline
│   ├── code.py           # Code-based graders
│   └── model.py          # LLM-as-judge (Phase 3)
└── suites/
    ├── __init__.py       # Suite exports
    ├── unit.py           # Individual tool validation
    └── resilience.py     # Fault injection tests
```

## 8. Key Files to Modify

| File | Change |
|------|--------|
| `src/mcp/server.py` | Add Trial wrapping to spawn_agent handlers |
| `src/self_healing.py` | Add semantic failure tracking method |
| `src/observability.py` | Add evaluation scoring helpers |

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Grader drift | Medium | High | Weekly calibration against Golden Dataset |
| Performance overhead | Low | Medium | Async grading, sampling in production |
| False positives | Medium | Medium | Start with conservative thresholds |
| Integration catastrophe | Low | Critical | Sandbox Gate for Gmail/Calendar before production |

## 10. Success Criteria for Launch

- [ ] Trial schema captures all execution context
- [ ] Code graders validate JSON/regex patterns
- [ ] Langfuse shows evaluation scores on traces
- [ ] Semantic failures tracked in circuit breaker stats
- [ ] spawn_agent response includes evaluation results
- [ ] Training data exports to D:\Research\training-data\
- [ ] Unit tests pass for all graders
