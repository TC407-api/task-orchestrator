# Self-Healing Agent Template

Demonstrates resilient agent execution with circuit breaker patterns and exponential backoff.

## Features

- Circuit breaker pattern (CLOSED → OPEN → HALF_OPEN)
- Exponential backoff with jitter
- Automatic recovery testing
- Failure counting and thresholds

## Circuit Breaker States

```
CLOSED (normal) → 3 failures → OPEN (blocking)
                                    ↓
                              30s timeout
                                    ↓
                            HALF_OPEN (testing)
                                    ↓
                success → CLOSED / failure → OPEN
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

## Configuration

Customize the circuit breaker:

```python
circuit = CircuitBreaker(
    failure_threshold=5,    # Failures before opening
    recovery_timeout=60.0,  # Seconds before testing recovery
)
```

Customize retry behavior:

```python
await retry_with_backoff(
    func,
    max_retries=5,
    base_delay=2.0,
    max_delay=120.0,
    jitter=True,
)
```

## Next Steps

- Add health checks for proactive circuit opening
- Implement bulkhead pattern for isolation
- Add metrics and alerting
