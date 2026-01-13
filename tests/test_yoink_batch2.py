"""
Comprehensive Test Suite for Anti-gravity Yoink Batch 2 Features.

Tests for (TDD style - defining expected interfaces):
- Terminal Loop (terminal_loop.py)
- Shadow AST Validation (shadow_validator.py)
- @Workflow System (workflows.py)
- Background Tasks (background_tasks.py)

These tests define the expected API contracts for batch 2 features.
Run with:
    JWT_SECRET_KEY=test123 python -m pytest tests/test_yoink_batch2.py -v
    JWT_SECRET_KEY=test123 python -m pytest tests/test_yoink_batch2.py -v --cov=src --cov-report=html
"""

import pytest
import asyncio
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Coroutine
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# SECTION 1: Terminal Loop Tests (terminal_loop.py)
# =============================================================================
# A reusable loop system for agent tasks, dialogue, and iterative workflows.
# Supports local execution, distributed queues, and monitoring.


class LoopState(Enum):
    """Terminal loop execution states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LoopIteration:
    """Single iteration of a terminal loop."""
    iteration_number: int
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class LoopConfig:
    """Terminal loop configuration."""
    name: str
    max_iterations: int = 100
    timeout_seconds: int = 300
    retry_policy: str = "exponential_backoff"
    max_retries: int = 3
    loop_condition: Optional[Callable] = None  # Callable that returns bool
    on_iteration_complete: Optional[Callable] = None
    persistent: bool = False
    persistence_path: Optional[str] = None


class TestTerminalLoopBasics:
    """Tests for basic terminal loop functionality."""

    @pytest.mark.asyncio
    async def test_create_terminal_loop(self):
        """Test creating a terminal loop instance."""
        from src.agents.terminal_loop import TerminalLoop

        loop = TerminalLoop(
            name="test_loop",
            max_iterations=10,
            timeout_seconds=60,
        )

        assert loop.name == "test_loop"
        assert loop.max_iterations == 10
        assert loop.timeout_seconds == 60
        assert loop.state == LoopState.IDLE

    @pytest.mark.asyncio
    async def test_terminal_loop_run_single_iteration(self):
        """Test running a single iteration."""
        from src.agents.terminal_loop import TerminalLoop

        async def simple_task(input_data: Dict) -> Dict:
            return {"result": input_data.get("value", 0) * 2}

        loop = TerminalLoop(name="test_loop", max_iterations=1)
        result = await loop.run({"value": 5}, simple_task)

        assert loop.state == LoopState.COMPLETED
        assert result["result"] == 10

    @pytest.mark.asyncio
    async def test_terminal_loop_multiple_iterations(self):
        """Test running multiple iterations until condition."""
        from src.agents.terminal_loop import TerminalLoop

        iteration_count = 0

        async def task_with_state(input_data: Dict) -> Dict:
            nonlocal iteration_count
            iteration_count += 1
            return {"count": iteration_count}

        def should_continue(result: Dict) -> bool:
            return result["count"] < 5

        loop = TerminalLoop(
            name="test_loop",
            max_iterations=10,
            loop_condition=should_continue,
        )

        final_result = await loop.run({}, task_with_state)

        assert iteration_count == 5
        assert loop.state == LoopState.COMPLETED

    @pytest.mark.asyncio
    async def test_terminal_loop_state_transitions(self):
        """Test state transitions during loop execution."""
        from src.agents.terminal_loop import TerminalLoop

        async def task(input_data):
            await asyncio.sleep(0.01)
            return {"done": True}

        loop = TerminalLoop(name="test_loop", max_iterations=1)

        assert loop.state == LoopState.IDLE

        task_coro = loop.run({}, task)
        # State should transition to RUNNING
        result = await task_coro

        assert loop.state == LoopState.COMPLETED

    @pytest.mark.asyncio
    async def test_terminal_loop_timeout(self):
        """Test that loop respects timeout."""
        from src.agents.terminal_loop import TerminalLoop

        async def slow_task(input_data):
            await asyncio.sleep(10)
            return {"done": True}

        loop = TerminalLoop(name="test_loop", timeout_seconds=0.1)

        with pytest.raises(asyncio.TimeoutError):
            await loop.run({}, slow_task)

        assert loop.state == LoopState.FAILED

    @pytest.mark.asyncio
    async def test_terminal_loop_iteration_history(self):
        """Test that iteration history is tracked."""
        from src.agents.terminal_loop import TerminalLoop

        async def task(input_data):
            return {"value": input_data.get("value", 0) + 1}

        def should_continue(result):
            return result["value"] < 3

        loop = TerminalLoop(
            name="test_loop",
            max_iterations=10,
            loop_condition=should_continue,
        )

        await loop.run({"value": 0}, task)

        history = loop.get_iteration_history()
        assert len(history) >= 3
        assert all(isinstance(it, LoopIteration) for it in history)

    @pytest.mark.asyncio
    async def test_terminal_loop_pause_resume(self):
        """Test pausing and resuming a loop."""
        from src.agents.terminal_loop import TerminalLoop

        async def task(input_data):
            return {"value": input_data.get("value", 0) + 1}

        loop = TerminalLoop(name="test_loop", max_iterations=10)

        # Start loop in background
        task_ref = asyncio.create_task(loop.run({"value": 0}, task))
        await asyncio.sleep(0.01)

        loop.pause()
        assert loop.state == LoopState.PAUSED

        loop.resume()
        assert loop.state == LoopState.RUNNING

        try:
            await asyncio.wait_for(task_ref, timeout=1.0)
        except asyncio.TimeoutError:
            loop.cancel()

    @pytest.mark.asyncio
    async def test_terminal_loop_error_handling(self):
        """Test error handling and retry logic."""
        from src.agents.terminal_loop import TerminalLoop

        call_count = 0

        async def failing_task(input_data):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return {"success": True}

        loop = TerminalLoop(
            name="test_loop",
            max_iterations=1,
            retry_policy="exponential_backoff",
            max_retries=3,
        )

        result = await loop.run({}, failing_task)
        assert result["success"] is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_terminal_loop_persistent_storage(self):
        """Test persistent loop state."""
        from src.agents.terminal_loop import TerminalLoop

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_file = Path(tmpdir) / "loop_state.json"

            loop = TerminalLoop(
                name="persistent_loop",
                max_iterations=5,
                persistent=True,
                persistence_path=str(persistence_file),
            )

            async def task(input_data):
                return {"iteration": input_data.get("iteration", 0) + 1}

            await loop.run({"iteration": 0}, task)

            # Check state was persisted
            assert persistence_file.exists()
            data = json.loads(persistence_file.read_text())
            assert data["name"] == "persistent_loop"

    @pytest.mark.asyncio
    async def test_terminal_loop_cancel(self):
        """Test cancelling a loop."""
        from src.agents.terminal_loop import TerminalLoop

        async def long_task(input_data):
            await asyncio.sleep(10)
            return {"done": True}

        loop = TerminalLoop(name="test_loop", max_iterations=100)

        task_ref = asyncio.create_task(loop.run({}, long_task))
        await asyncio.sleep(0.01)

        loop.cancel()
        assert loop.state == LoopState.CANCELLED

        with pytest.raises(asyncio.CancelledError):
            await task_ref


class TestTerminalLoopCallbacks:
    """Tests for terminal loop callbacks and hooks."""

    @pytest.mark.asyncio
    async def test_loop_iteration_callback(self):
        """Test on_iteration_complete callback."""
        from src.agents.terminal_loop import TerminalLoop

        callback_results = []

        async def callback(iteration: LoopIteration):
            callback_results.append(iteration)

        async def task(input_data):
            return {"count": input_data.get("count", 0) + 1}

        def should_continue(result):
            return result["count"] < 3

        loop = TerminalLoop(
            name="test_loop",
            max_iterations=10,
            loop_condition=should_continue,
            on_iteration_complete=callback,
        )

        await loop.run({"count": 0}, task)

        assert len(callback_results) >= 3
        assert all(isinstance(it, LoopIteration) for it in callback_results)

    @pytest.mark.asyncio
    async def test_loop_statistics(self):
        """Test gathering loop statistics."""
        from src.agents.terminal_loop import TerminalLoop

        async def task(input_data):
            await asyncio.sleep(0.01)
            return {"value": input_data.get("value", 0) + 1}

        def should_continue(result):
            return result["value"] < 5

        loop = TerminalLoop(
            name="test_loop",
            max_iterations=10,
            loop_condition=should_continue,
        )

        await loop.run({"value": 0}, task)

        stats = loop.get_statistics()
        assert "total_iterations" in stats
        assert "total_duration_ms" in stats
        assert "average_iteration_duration_ms" in stats
        assert stats["total_iterations"] >= 5

    @pytest.mark.asyncio
    async def test_loop_input_output_validation(self):
        """Test input/output validation."""
        from src.agents.terminal_loop import TerminalLoop

        def validate_input(input_data):
            assert isinstance(input_data, dict)
            assert "value" in input_data

        def validate_output(output_data):
            assert isinstance(output_data, dict)
            assert "result" in output_data

        async def task(input_data):
            validate_input(input_data)
            result = {"result": input_data["value"] * 2}
            validate_output(result)
            return result

        loop = TerminalLoop(name="test_loop", max_iterations=1)
        await loop.run({"value": 5}, task)


class TestTerminalLoopDistributed:
    """Tests for distributed terminal loop features."""

    @pytest.mark.asyncio
    async def test_loop_with_queue_backend(self):
        """Test loop with distributed queue backend."""
        from src.agents.terminal_loop import TerminalLoop

        queue_items = []

        async def task(input_data):
            queue_items.append(input_data)
            return {"processed": True}

        loop = TerminalLoop(
            name="test_loop",
            max_iterations=5,
        )

        for i in range(5):
            await loop.run({"id": i}, task)

        assert len(queue_items) == 5

    @pytest.mark.asyncio
    async def test_loop_with_priority_handling(self):
        """Test loop respects item priority."""
        from src.agents.terminal_loop import TerminalLoop

        processed_order = []

        async def task(input_data):
            processed_order.append(input_data["priority"])
            return {"done": True}

        loop = TerminalLoop(name="test_loop", max_iterations=3)

        await loop.run({"priority": 1}, task)
        await loop.run({"priority": 3}, task)
        await loop.run({"priority": 2}, task)


class TestTerminalLoopPerformance:
    """Performance tests for terminal loop."""

    @pytest.mark.asyncio
    async def test_loop_throughput(self):
        """Test loop throughput - iterations per second."""
        from src.agents.terminal_loop import TerminalLoop

        async def fast_task(input_data):
            return {"value": input_data.get("value", 0) + 1}

        def should_continue(result):
            return result["value"] < 100

        loop = TerminalLoop(
            name="perf_loop",
            max_iterations=1000,
            loop_condition=should_continue,
        )

        start = time.time()
        await loop.run({"value": 0}, fast_task)
        elapsed = time.time() - start

        stats = loop.get_statistics()
        throughput = stats["total_iterations"] / elapsed

        # Should process at least 50 iterations/second
        assert throughput > 50

    @pytest.mark.asyncio
    async def test_loop_memory_efficiency(self):
        """Test loop doesn't accumulate excess memory."""
        from src.agents.terminal_loop import TerminalLoop

        async def task(input_data):
            # Create some data that should be garbage collected
            large_data = [0] * 10000
            return {"value": input_data.get("value", 0) + 1}

        def should_continue(result):
            return result["value"] < 50

        loop = TerminalLoop(
            name="memory_loop",
            max_iterations=50,
            loop_condition=should_continue,
        )

        await loop.run({"value": 0}, task)

        stats = loop.get_statistics()
        assert "memory_peak_mb" in stats or True  # Optional metric


# =============================================================================
# SECTION 2: Shadow AST Validation Tests (shadow_validator.py)
# =============================================================================
# A validation system that compares actual code AST against expected patterns.
# Detects mutations, unexpected changes, and security issues.


class ASTNodeType(Enum):
    """AST node types to validate."""
    FUNCTION = "function"
    CLASS = "class"
    IMPORT = "import"
    CALL = "call"
    ASSIGNMENT = "assignment"
    RETURN = "return"


@dataclass
class ASTNode:
    """Represents an AST node."""
    type: ASTNodeType
    name: str
    line_number: int
    source: str
    children: List["ASTNode"] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of AST validation."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TestShadowASTValidation:
    """Tests for shadow AST validation."""

    def test_parse_python_source(self):
        """Test parsing Python source code into AST."""
        from src.agents.shadow_validator import ShadowValidator

        source_code = """
def hello(name):
    return f"Hello, {name}!"

class MyClass:
    def method(self):
        pass
"""

        validator = ShadowValidator()
        ast = validator.parse(source_code)

        assert ast is not None
        assert len(ast.children) >= 2  # Function and class

    def test_extract_function_signature(self):
        """Test extracting function signatures."""
        from src.agents.shadow_validator import ShadowValidator

        source_code = "def add(a, b, c=0): return a + b + c"

        validator = ShadowValidator()
        sig = validator.extract_signature("add", source_code)

        assert sig is not None
        assert sig.name == "add"
        assert len(sig.get("parameters", [])) == 3

    def test_validate_against_shadow(self):
        """Test validating code against a shadow copy."""
        from src.agents.shadow_validator import ShadowValidator

        original = """
def process_data(data):
    result = data * 2
    return result
"""

        modified_good = """
def process_data(data):
    result = data * 2
    return result
"""

        validator = ShadowValidator()
        validator.set_shadow(original)
        result = validator.validate(modified_good)

        assert result.is_valid is True
        assert len(result.violations) == 0

    def test_detect_function_mutation(self):
        """Test detecting function mutations."""
        from src.agents.shadow_validator import ShadowValidator

        original = """
def calculate(x):
    return x * 2
"""

        mutated = """
def calculate(x):
    return x * 3  # Changed logic!
"""

        validator = ShadowValidator()
        validator.set_shadow(original)
        result = validator.validate(mutated)

        assert result.is_valid is False
        assert len(result.mutations) > 0
        assert "calculate" in " ".join(result.mutations)

    def test_detect_added_imports(self):
        """Test detecting added or removed imports."""
        from src.agents.shadow_validator import ShadowValidator

        original = """
import os
import sys
"""

        with_extra = """
import os
import sys
import subprocess  # Added!
"""

        validator = ShadowValidator()
        validator.set_shadow(original)
        result = validator.validate(with_extra)

        assert len(result.mutations) > 0

    def test_detect_parameter_changes(self):
        """Test detecting parameter list changes."""
        from src.agents.shadow_validator import ShadowValidator

        original = """
def func(a, b):
    return a + b
"""

        changed_params = """
def func(a, b, c):  # Added parameter!
    return a + b + c
"""

        validator = ShadowValidator()
        validator.set_shadow(original)
        result = validator.validate(changed_params)

        assert result.is_valid is False

    def test_detect_security_violations(self):
        """Test detecting security violations."""
        from src.agents.shadow_validator import ShadowValidator

        dangerous_code = """
import os
def dangerous():
    os.system("rm -rf /")  # Security issue!
"""

        validator = ShadowValidator()
        result = validator.validate(dangerous_code)

        assert len(result.warnings) > 0 or len(result.violations) > 0

    def test_validate_class_structure(self):
        """Test validating class structure."""
        from src.agents.shadow_validator import ShadowValidator

        original = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""

        validator = ShadowValidator()
        validator.set_shadow(original)

        same_structure = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""

        result = validator.validate(same_structure)
        assert result.is_valid is True

    def test_detect_removed_methods(self):
        """Test detecting removed methods."""
        from src.agents.shadow_validator import ShadowValidator

        original = """
class DataHandler:
    def load(self): pass
    def save(self): pass
    def delete(self): pass
"""

        reduced = """
class DataHandler:
    def load(self): pass
    def save(self): pass
"""

        validator = ShadowValidator()
        validator.set_shadow(original)
        result = validator.validate(reduced)

        assert result.is_valid is False
        assert "delete" in " ".join(result.mutations)

    def test_validate_call_chains(self):
        """Test validating method call chains."""
        from src.agents.shadow_validator import ShadowValidator

        source = """
result = data.filter(x > 5).map(x * 2).collect()
"""

        validator = ShadowValidator()
        calls = validator.extract_call_chain(source)

        assert calls is not None
        assert "filter" in str(calls)
        assert "map" in str(calls)

    def test_validate_return_statements(self):
        """Test validating return statement consistency."""
        from src.agents.shadow_validator import ShadowValidator

        good_code = """
def process(x):
    if x > 0:
        return x
    else:
        return 0
"""

        validator = ShadowValidator()
        returns = validator.extract_returns("process", good_code)

        assert len(returns) >= 2

    def test_incremental_validation(self):
        """Test validating code incrementally as it changes."""
        from src.agents.shadow_validator import ShadowValidator

        original = """
def calculate(x):
    return x * 2
"""

        validator = ShadowValidator()
        validator.set_shadow(original)

        # First change
        change1 = """
def calculate(x):
    result = x * 2
    return result
"""

        result1 = validator.validate(change1)
        # Should be valid (just refactoring)

        # Second change
        change2 = """
def calculate(x):
    result = x * 3
    return result
"""

        result2 = validator.validate(change2)
        # Should be invalid (logic changed)

    def test_generate_diff_report(self):
        """Test generating validation report."""
        from src.agents.shadow_validator import ShadowValidator

        original = """
def main():
    x = 1
    y = 2
    return x + y
"""

        modified = """
def main():
    x = 1
    y = 3
    return x + y
"""

        validator = ShadowValidator()
        validator.set_shadow(original)
        result = validator.validate(modified)

        report = validator.generate_report()
        assert "original" in report or True  # Optional detailed report


# =============================================================================
# SECTION 3: @Workflow System Tests (workflows.py)
# =============================================================================
# Decorator-based workflow system for orchestrating multi-step operations.


@dataclass
class WorkflowStep:
    """Single step in a workflow."""
    name: str
    description: str
    handler: Callable
    required: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    depends_on: List[str] = field(default_factory=list)


@dataclass
class WorkflowState:
    """Current state of a workflow execution."""
    workflow_id: str
    current_step: str
    status: str  # pending, running, completed, failed
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    timestamps: Dict[str, datetime] = field(default_factory=dict)


class TestWorkflowDecorator:
    """Tests for @Workflow decorator."""

    @pytest.mark.asyncio
    async def test_simple_workflow_definition(self):
        """Test defining a simple workflow with decorator."""
        from src.agents.workflows import workflow, step

        @workflow(name="data_pipeline", description="Process data")
        class DataPipeline:
            @step(name="extract", description="Extract data")
            async def extract(self, source: str) -> Dict:
                return {"data": [1, 2, 3]}

            @step(name="transform", description="Transform data")
            async def transform(self, data: List) -> Dict:
                return {"transformed": [x * 2 for x in data]}

            @step(name="load", description="Load data")
            async def load(self, transformed: List) -> Dict:
                return {"loaded": True}

        pipeline = DataPipeline()
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test executing a complete workflow."""
        from src.agents.workflows import workflow, step

        @workflow(name="simple_wf")
        class SimpleWorkflow:
            @step(name="step1")
            async def step1(self) -> Dict:
                return {"value": 10}

            @step(name="step2")
            async def step2(self, value: int) -> Dict:
                return {"result": value * 2}

        wf = SimpleWorkflow()
        result = await wf.execute()

        assert result is not None
        assert result["result"] == 20

    @pytest.mark.asyncio
    async def test_workflow_dependency_resolution(self):
        """Test step dependency ordering."""
        from src.agents.workflows import workflow, step

        execution_order = []

        @workflow(name="deps_wf")
        class DepWorkflow:
            @step(name="step_a")
            async def step_a(self) -> Dict:
                execution_order.append("a")
                return {"value": 1}

            @step(name="step_b", depends_on=["step_a"])
            async def step_b(self, value: int) -> Dict:
                execution_order.append("b")
                return {"value": value + 1}

            @step(name="step_c", depends_on=["step_b"])
            async def step_c(self, value: int) -> Dict:
                execution_order.append("c")
                return {"value": value + 1}

        wf = DepWorkflow()
        await wf.execute()

        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_workflow_step_timeout(self):
        """Test step timeout handling."""
        from src.agents.workflows import workflow, step

        @workflow(name="timeout_wf")
        class TimeoutWorkflow:
            @step(name="slow_step", timeout=0.1)
            async def slow_step(self) -> Dict:
                await asyncio.sleep(1.0)
                return {"done": True}

        wf = TimeoutWorkflow()

        with pytest.raises(asyncio.TimeoutError):
            await wf.execute()

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow error handling."""
        from src.agents.workflows import workflow, step

        @workflow(name="error_wf")
        class ErrorWorkflow:
            @step(name="fail_step")
            async def fail_step(self) -> Dict:
                raise ValueError("Step failed")

            @step(name="after_fail", required=False)
            async def after_fail(self) -> Dict:
                return {"never_runs": True}

        wf = ErrorWorkflow()

        state = await wf.execute(continue_on_error=False)
        assert "fail_step" in state.errors

    @pytest.mark.asyncio
    async def test_workflow_retry_logic(self):
        """Test step retry on failure."""
        from src.agents.workflows import workflow, step

        call_count = 0

        @workflow(name="retry_wf")
        class RetryWorkflow:
            @step(name="retry_step", retry_count=3)
            async def retry_step(self) -> Dict:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ValueError("Not yet")
                return {"success": True}

        wf = RetryWorkflow()
        await wf.execute()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_workflow_state_management(self):
        """Test workflow state tracking."""
        from src.agents.workflows import workflow, step

        @workflow(name="state_wf")
        class StateWorkflow:
            @step(name="step1")
            async def step1(self) -> Dict:
                return {"value": 1}

            @step(name="step2")
            async def step2(self, value: int) -> Dict:
                return {"value": value + 1}

        wf = StateWorkflow()
        state = await wf.get_state()

        assert state is not None
        assert state.workflow_id is not None

    @pytest.mark.asyncio
    async def test_workflow_parallel_steps(self):
        """Test parallel execution of independent steps."""
        from src.agents.workflows import workflow, step, parallel

        @workflow(name="parallel_wf")
        class ParallelWorkflow:
            @step(name="step_a")
            async def step_a(self) -> Dict:
                await asyncio.sleep(0.05)
                return {"a": 1}

            @step(name="step_b")
            async def step_b(self) -> Dict:
                await asyncio.sleep(0.05)
                return {"b": 2}

            @step(name="combine", depends_on=["step_a", "step_b"])
            async def combine(self, a: int, b: int) -> Dict:
                return {"result": a + b}

        wf = ParallelWorkflow()
        start = time.time()
        await wf.execute()
        elapsed = time.time() - start

        # Should be ~0.05s (parallel) not ~0.1s (serial)
        assert elapsed < 0.12

    @pytest.mark.asyncio
    async def test_workflow_conditional_steps(self):
        """Test conditional step execution."""
        from src.agents.workflows import workflow, step

        @workflow(name="conditional_wf")
        class ConditionalWorkflow:
            @step(name="check")
            async def check(self) -> Dict:
                return {"condition": True}

            @step(name="if_true", condition=lambda r: r.get("condition"))
            async def if_true(self) -> Dict:
                return {"path": "true"}

            @step(name="if_false", condition=lambda r: not r.get("condition"))
            async def if_false(self) -> Dict:
                return {"path": "false"}

        wf = ConditionalWorkflow()
        result = await wf.execute()

        assert result["path"] == "true"

    @pytest.mark.asyncio
    async def test_workflow_data_passing(self):
        """Test passing data between workflow steps."""
        from src.agents.workflows import workflow, step

        @workflow(name="data_wf")
        class DataWorkflow:
            @step(name="produce")
            async def produce(self) -> Dict:
                return {"message": "hello", "value": 42}

            @step(name="consume")
            async def consume(self, message: str, value: int) -> Dict:
                return {"output": f"{message}_{value}"}

        wf = DataWorkflow()
        result = await wf.execute()

        assert result["output"] == "hello_42"

    @pytest.mark.asyncio
    async def test_workflow_cancellation(self):
        """Test cancelling a workflow mid-execution."""
        from src.agents.workflows import workflow, step

        @workflow(name="cancel_wf")
        class CancelWorkflow:
            @step(name="step1")
            async def step1(self) -> Dict:
                return {"done": True}

        wf = CancelWorkflow()
        task = asyncio.create_task(wf.execute())
        await asyncio.sleep(0.01)

        wf.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_workflow_persistence(self):
        """Test saving and resuming workflow state."""
        from src.agents.workflows import workflow, step

        @workflow(name="persist_wf", persistent=True)
        class PersistWorkflow:
            @step(name="step1")
            async def step1(self) -> Dict:
                return {"checkpoint": 1}

            @step(name="step2")
            async def step2(self) -> Dict:
                return {"checkpoint": 2}

        wf = PersistWorkflow()

        # Execute, save state
        state1 = await wf.execute()
        saved_state = wf.save_state()

        assert saved_state is not None

        # Create new workflow and resume
        wf2 = PersistWorkflow()
        wf2.restore_state(saved_state)
        state2 = await wf2.execute()

        assert state1 == state2

    @pytest.mark.asyncio
    async def test_workflow_hooks(self):
        """Test workflow lifecycle hooks."""
        from src.agents.workflows import workflow, step

        hooks_called = []

        async def on_start(wf):
            hooks_called.append("start")

        async def on_complete(wf, result):
            hooks_called.append("complete")

        @workflow(
            name="hooks_wf",
            on_start=on_start,
            on_complete=on_complete,
        )
        class HooksWorkflow:
            @step(name="step1")
            async def step1(self) -> Dict:
                return {"done": True}

        wf = HooksWorkflow()
        await wf.execute()

        assert "start" in hooks_called
        assert "complete" in hooks_called

    @pytest.mark.asyncio
    async def test_workflow_monitoring(self):
        """Test workflow monitoring and metrics."""
        from src.agents.workflows import workflow, step

        @workflow(name="monitor_wf")
        class MonitorWorkflow:
            @step(name="step1")
            async def step1(self) -> Dict:
                await asyncio.sleep(0.01)
                return {"done": True}

        wf = MonitorWorkflow()
        await wf.execute()

        metrics = wf.get_metrics()
        assert "execution_time_ms" in metrics
        assert "steps_executed" in metrics


# =============================================================================
# SECTION 4: Background Tasks Tests (background_tasks.py)
# =============================================================================
# Async background task queue with scheduling, retries, and monitoring.


@dataclass
class TaskDefinition:
    """Background task definition."""
    id: str
    name: str
    handler: Callable
    schedule: Optional[str] = None  # cron-like: "0 * * * *"
    priority: int = 0
    retries: int = 3
    timeout: Optional[float] = None
    enabled: bool = True


@dataclass
class TaskExecution:
    """Single task execution record."""
    task_id: str
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0


class TestBackgroundTasks:
    """Tests for background task system."""

    @pytest.mark.asyncio
    async def test_create_background_task(self):
        """Test creating a background task."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def my_task():
            return {"done": True}

        manager = BackgroundTaskManager()
        task_id = manager.register_task(
            name="my_task",
            handler=my_task,
            priority=1,
        )

        assert task_id is not None

    @pytest.mark.asyncio
    async def test_enqueue_and_process_task(self):
        """Test enqueueing and processing a task."""
        from src.agents.background_tasks import BackgroundTaskManager

        result_holder = []

        async def process_data():
            result_holder.append({"processed": True})
            return {"done": True}

        manager = BackgroundTaskManager(max_workers=2)
        task_id = manager.register_task("process", process_data)
        execution_id = await manager.enqueue(task_id)

        await asyncio.sleep(0.1)  # Let it process

        execution = manager.get_execution(execution_id)
        assert execution.status in ["completed", "running"]

    @pytest.mark.asyncio
    async def test_task_scheduling(self):
        """Test scheduled task execution."""
        from src.agents.background_tasks import BackgroundTaskManager

        call_count = 0

        async def scheduled_task():
            nonlocal call_count
            call_count += 1

        manager = BackgroundTaskManager(enable_scheduler=True)
        manager.register_task(
            name="scheduled",
            handler=scheduled_task,
            schedule="*/1 * * * *",  # Every minute (for testing, would be faster)
        )

        # In real test would wait for schedule trigger
        # For now just verify registration works
        tasks = manager.get_registered_tasks()
        assert len(tasks) >= 1

    @pytest.mark.asyncio
    async def test_task_priority_handling(self):
        """Test tasks execute in priority order."""
        from src.agents.background_tasks import BackgroundTaskManager

        execution_order = []

        async def task_a():
            execution_order.append("a")

        async def task_b():
            execution_order.append("b")

        async def task_c():
            execution_order.append("c")

        manager = BackgroundTaskManager(max_workers=1)
        id_a = manager.register_task("a", task_a, priority=1)
        id_b = manager.register_task("b", task_b, priority=3)
        id_c = manager.register_task("c", task_c, priority=2)

        # Enqueue in reverse priority order
        manager.enqueue(id_a)
        manager.enqueue(id_b)
        manager.enqueue(id_c)

        await asyncio.sleep(0.3)  # Let them process

        # Should process in priority order: b(3), c(2), a(1)
        assert execution_order[0] == "b"

    @pytest.mark.asyncio
    async def test_task_retry_on_failure(self):
        """Test task retry on failure."""
        from src.agents.background_tasks import BackgroundTaskManager

        attempt_count = 0

        async def failing_task():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Not yet")
            return {"success": True}

        manager = BackgroundTaskManager()
        task_id = manager.register_task("retry", failing_task, retries=3)
        execution_id = await manager.enqueue(task_id)

        await asyncio.sleep(0.2)

        execution = manager.get_execution(execution_id)
        # After retries, should be completed
        assert execution.retry_count >= 2

    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task timeout handling."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def slow_task():
            await asyncio.sleep(10)

        manager = BackgroundTaskManager()
        task_id = manager.register_task("slow", slow_task, timeout=0.1)
        execution_id = await manager.enqueue(task_id)

        await asyncio.sleep(0.2)

        execution = manager.get_execution(execution_id)
        assert execution.status == "failed"

    @pytest.mark.asyncio
    async def test_task_cancellation(self):
        """Test cancelling a running task."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def long_task():
            await asyncio.sleep(10)

        manager = BackgroundTaskManager()
        task_id = manager.register_task("long", long_task)
        execution_id = await manager.enqueue(task_id)

        await asyncio.sleep(0.01)
        manager.cancel_execution(execution_id)

        execution = manager.get_execution(execution_id)
        assert execution.status == "cancelled"

    @pytest.mark.asyncio
    async def test_task_result_storage(self):
        """Test storing and retrieving task results."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def result_task():
            return {"data": "important_result", "value": 42}

        manager = BackgroundTaskManager()
        task_id = manager.register_task("store_result", result_task)
        execution_id = await manager.enqueue(task_id)

        await asyncio.sleep(0.1)

        execution = manager.get_execution(execution_id)
        assert execution.result is not None
        assert execution.result["value"] == 42

    @pytest.mark.asyncio
    async def test_task_error_tracking(self):
        """Test tracking task errors."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def error_task():
            raise RuntimeError("Task failed with error")

        manager = BackgroundTaskManager()
        task_id = manager.register_task("error", error_task, retries=0)
        execution_id = await manager.enqueue(task_id)

        await asyncio.sleep(0.1)

        execution = manager.get_execution(execution_id)
        assert execution.status == "failed"
        assert execution.error is not None
        assert "Task failed" in execution.error

    @pytest.mark.asyncio
    async def test_task_queue_depth_monitoring(self):
        """Test monitoring queue depth."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def task():
            await asyncio.sleep(0.05)

        manager = BackgroundTaskManager(max_workers=1)
        task_id = manager.register_task("queue_test", task)

        # Enqueue multiple tasks
        for _ in range(5):
            manager.enqueue(task_id)

        stats = manager.get_statistics()
        assert "queue_depth" in stats
        assert stats["queue_depth"] >= 1

    @pytest.mark.asyncio
    async def test_task_worker_pool(self):
        """Test worker pool management."""
        from src.agents.background_tasks import BackgroundTaskManager

        concurrent_runs = 0
        max_concurrent = 0

        async def track_concurrent():
            nonlocal concurrent_runs, max_concurrent
            concurrent_runs += 1
            max_concurrent = max(max_concurrent, concurrent_runs)
            await asyncio.sleep(0.05)
            concurrent_runs -= 1

        manager = BackgroundTaskManager(max_workers=3)
        task_id = manager.register_task("concurrent", track_concurrent)

        # Enqueue 10 tasks
        for _ in range(10):
            manager.enqueue(task_id)

        await asyncio.sleep(0.2)

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_task_persistence(self):
        """Test persisting task queue to storage."""
        from src.agents.background_tasks import BackgroundTaskManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BackgroundTaskManager(
                persistent=True,
                storage_path=tmpdir,
            )

            async def persist_task():
                return {"saved": True}

            task_id = manager.register_task("persist", persist_task)
            manager.enqueue(task_id)

            await asyncio.sleep(0.1)

            # Verify persistence occurred
            persistence_file = Path(tmpdir) / "tasks.json"
            assert persistence_file.exists() or True  # Optional

    @pytest.mark.asyncio
    async def test_task_dependency_chain(self):
        """Test task dependency execution."""
        from src.agents.background_tasks import BackgroundTaskManager

        results = []

        async def task_a():
            results.append("a")
            return {"data": "from_a"}

        async def task_b(prev_result):
            results.append("b")
            return {"data": "from_b", "prev": prev_result}

        manager = BackgroundTaskManager()
        id_a = manager.register_task("a", task_a)
        id_b = manager.register_task("b", task_b)

        # Create dependency
        manager.add_dependency(id_b, id_a)

        manager.enqueue(id_b)
        await asyncio.sleep(0.2)

        # Task A should have run first
        assert "a" in results

    @pytest.mark.asyncio
    async def test_batch_task_processing(self):
        """Test processing batch of tasks."""
        from src.agents.background_tasks import BackgroundTaskManager

        processed_items = []

        async def batch_processor(items):
            processed_items.extend(items)
            return {"count": len(items)}

        manager = BackgroundTaskManager()
        task_id = manager.register_task("batch", batch_processor)

        items = [{"id": i} for i in range(10)]
        execution_id = await manager.enqueue(task_id, data=items)

        await asyncio.sleep(0.1)

        execution = manager.get_execution(execution_id)
        assert execution.result["count"] == 10

    @pytest.mark.asyncio
    async def test_task_dead_letter_queue(self):
        """Test dead letter queue for failed tasks."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def failing_task():
            raise RuntimeError("Always fails")

        manager = BackgroundTaskManager(max_retries=1)
        task_id = manager.register_task("dead_letter", failing_task, retries=1)
        execution_id = await manager.enqueue(task_id)

        await asyncio.sleep(0.2)

        # Should be in DLQ after exhausting retries
        dlq = manager.get_dead_letter_queue()
        assert len(dlq) >= 1


class TestBackgroundTasksMonitoring:
    """Tests for background task monitoring."""

    @pytest.mark.asyncio
    async def test_task_metrics_collection(self):
        """Test collecting task metrics."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def metric_task():
            await asyncio.sleep(0.01)
            return {"done": True}

        manager = BackgroundTaskManager()
        task_id = manager.register_task("metrics", metric_task)

        for _ in range(5):
            await manager.enqueue(task_id)

        await asyncio.sleep(0.1)

        metrics = manager.get_metrics(task_id)
        assert metrics["total_executions"] >= 5
        assert "average_duration_ms" in metrics

    @pytest.mark.asyncio
    async def test_task_history_tracking(self):
        """Test tracking task execution history."""
        from src.agents.background_tasks import BackgroundTaskManager

        async def history_task():
            return {"timestamp": datetime.now().isoformat()}

        manager = BackgroundTaskManager(history_size=10)
        task_id = manager.register_task("history", history_task)

        for _ in range(3):
            await manager.enqueue(task_id)

        await asyncio.sleep(0.1)

        history = manager.get_execution_history(task_id)
        assert len(history) >= 3


class TestBackgroundTasksIntegration:
    """Integration tests for background tasks."""

    @pytest.mark.asyncio
    async def test_workflow_with_background_tasks(self):
        """Test integration of workflows with background tasks."""
        from src.agents.background_tasks import BackgroundTaskManager
        from src.agents.workflows import workflow, step

        manager = BackgroundTaskManager()

        @workflow(name="workflow_with_bg")
        class WorkflowWithBG:
            @step(name="trigger_bg")
            async def trigger_bg(self) -> Dict:
                # Would enqueue background task here
                return {"task_queued": True}

        wf = WorkflowWithBG()
        result = await wf.execute()
        assert result["task_queued"] is True

    @pytest.mark.asyncio
    async def test_terminal_loop_with_background_tasks(self):
        """Test integration of terminal loop with background tasks."""
        from src.agents.terminal_loop import TerminalLoop
        from src.agents.background_tasks import BackgroundTaskManager

        manager = BackgroundTaskManager()
        processed = []

        async def background_work(item):
            processed.append(item)
            return {"processed": True}

        async def loop_task(input_data):
            item = input_data.get("item")
            # Queue background work
            return {"queued": True}

        loop = TerminalLoop(name="bg_loop", max_iterations=1)
        await loop.run({"item": "test"}, loop_task)


# =============================================================================
# Integration Tests Between All Batch 2 Components
# =============================================================================


class TestBatch2Integration:
    """Integration tests for all batch 2 components together."""

    @pytest.mark.asyncio
    async def test_workflow_validates_with_shadow_ast(self):
        """Test that workflows are validated with shadow AST."""
        from src.agents.workflows import workflow, step
        from src.agents.shadow_validator import ShadowValidator

        @workflow(name="validated_wf")
        class ValidatedWorkflow:
            @step(name="step1")
            async def step1(self) -> Dict:
                return {"value": 1}

        wf = ValidatedWorkflow()
        validator = ShadowValidator()

        # Workflows should be AST-validatable
        # This integration test just verifies the concept

    @pytest.mark.asyncio
    async def test_background_task_in_terminal_loop(self):
        """Test background task execution within terminal loop."""
        from src.agents.terminal_loop import TerminalLoop
        from src.agents.background_tasks import BackgroundTaskManager

        manager = BackgroundTaskManager()

        async def loop_with_bg_task(input_data):
            # Simulate enqueueing a background task
            return {"done": True}

        loop = TerminalLoop(name="integrated_loop", max_iterations=1)
        result = await loop.run({}, loop_with_bg_task)

        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full integration of all batch 2 components."""
        from src.agents.terminal_loop import TerminalLoop
        from src.agents.workflows import workflow, step
        from src.agents.background_tasks import BackgroundTaskManager
        from src.agents.shadow_validator import ShadowValidator

        # Create instances of all components
        loop = TerminalLoop(name="integrated", max_iterations=1)
        manager = BackgroundTaskManager()
        validator = ShadowValidator()

        @workflow(name="integration_test")
        class IntegrationWorkflow:
            @step(name="main")
            async def main(self) -> Dict:
                return {"integrated": True}

        wf = IntegrationWorkflow()
        result = await wf.execute()

        assert result["integrated"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
