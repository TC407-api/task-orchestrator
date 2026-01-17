"""Tests for background task scheduling system."""
import asyncio
import pytest
from datetime import datetime, timedelta

from .background_tasks import (
    BackgroundTaskScheduler,
    ScheduledTask,
    TaskScheduleType,
    TaskStatus,
)
from .inbox import UniversalInbox


@pytest.fixture
def inbox():
    """Create a test inbox."""
    return UniversalInbox()


@pytest.fixture
def scheduler(inbox):
    """Create a test scheduler."""
    return BackgroundTaskScheduler(inbox=inbox, max_workers=2)


@pytest.mark.asyncio
async def test_schedule_one_time_task(scheduler):
    """Test scheduling a one-time task."""
    executed = False

    async def test_func():
        nonlocal executed
        executed = True
        return "done"

    task = ScheduledTask(
        name="test_one_time",
        func=test_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
    )

    task_id = await scheduler.schedule_task(task)
    assert task_id == task.task_id

    # Start scheduler
    await scheduler.start()

    # Wait for task to execute
    await asyncio.sleep(1.5)

    # Check status
    status = await scheduler.get_task_status(task_id)
    assert status is not None
    assert status["status"] == TaskStatus.COMPLETED.value

    await scheduler.stop()


@pytest.mark.asyncio
async def test_schedule_recurring_task(scheduler):
    """Test scheduling a recurring task."""
    call_count = 0

    async def test_func():
        nonlocal call_count
        call_count += 1
        return f"execution_{call_count}"

    task = ScheduledTask(
        name="test_recurring",
        func=test_func,
        schedule_type=TaskScheduleType.RECURRING,
        interval_seconds=1,
    )

    task_id = await scheduler.schedule_task(task)
    assert task_id == task.task_id

    # Start scheduler
    await scheduler.start()

    # Wait for multiple executions
    await asyncio.sleep(3.5)

    # Check that task ran multiple times
    status = await scheduler.get_task_status(task_id)
    assert status is not None
    assert call_count >= 2, f"Expected at least 2 executions, got {call_count}"

    await scheduler.stop()


@pytest.mark.asyncio
async def test_schedule_deferred_task(scheduler):
    """Test scheduling a deferred task."""
    executed = False

    async def test_func():
        nonlocal executed
        executed = True
        return "deferred_done"

    task = ScheduledTask(
        name="test_deferred",
        func=test_func,
        schedule_type=TaskScheduleType.DEFERRED,
    )

    task_id = await scheduler.schedule_task(task)
    assert task_id == task.task_id

    # Start scheduler
    await scheduler.start()

    # Wait for task to execute
    await asyncio.sleep(1.5)

    # Check status
    status = await scheduler.get_task_status(task_id)
    assert status is not None
    assert status["status"] == TaskStatus.COMPLETED.value

    await scheduler.stop()


@pytest.mark.asyncio
async def test_cancel_task(scheduler):
    """Test cancelling a scheduled task."""
    call_count = 0

    async def test_func():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)
        return "should_not_complete"

    task = ScheduledTask(
        name="test_cancel",
        func=test_func,
        schedule_type=TaskScheduleType.RECURRING,
        interval_seconds=1,
    )

    task_id = await scheduler.schedule_task(task)

    # Start scheduler
    await scheduler.start()

    # Wait a bit
    await asyncio.sleep(1.5)

    # Cancel task
    cancelled = await scheduler.cancel_task(task_id)
    assert cancelled is True

    call_count_before_cancel = call_count

    # Wait to verify no more executions
    await asyncio.sleep(2)

    assert call_count == call_count_before_cancel

    await scheduler.stop()


@pytest.mark.asyncio
async def test_task_with_sync_function(scheduler):
    """Test scheduling a synchronous function."""
    executed = False

    def sync_func():
        nonlocal executed
        executed = True
        return "sync_result"

    task = ScheduledTask(
        name="test_sync",
        func=sync_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
    )

    task_id = await scheduler.schedule_task(task)

    await scheduler.start()
    await asyncio.sleep(1.5)

    status = await scheduler.get_task_status(task_id)
    assert status is not None
    assert status["status"] == TaskStatus.COMPLETED.value
    assert "sync_result" in (status["result"]["output"] or "")

    await scheduler.stop()


@pytest.mark.asyncio
async def test_task_with_args_and_kwargs(scheduler):
    """Test scheduling a task with arguments."""
    received_args = None
    received_kwargs = None

    async def test_func(a, b, c=None):
        nonlocal received_args, received_kwargs
        received_args = (a, b)
        received_kwargs = {"c": c}
        return f"{a}_{b}_{c}"

    task = ScheduledTask(
        name="test_args",
        func=test_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
        args=(1, 2),
        kwargs={"c": 3},
    )

    task_id = await scheduler.schedule_task(task)

    await scheduler.start()
    await asyncio.sleep(1.5)

    status = await scheduler.get_task_status(task_id)
    assert status is not None
    assert status["status"] == TaskStatus.COMPLETED.value

    await scheduler.stop()


@pytest.mark.asyncio
async def test_task_timeout(scheduler):
    """Test task timeout handling."""
    async def slow_func():
        await asyncio.sleep(10)
        return "should_timeout"

    task = ScheduledTask(
        name="test_timeout",
        func=slow_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
        timeout_seconds=0.1,
        max_retries=1,
    )

    task_id = await scheduler.schedule_task(task)

    await scheduler.start()
    await asyncio.sleep(2.5)  # Need to wait for scheduler loop + timeout + retry

    status = await scheduler.get_task_status(task_id)
    assert status is not None
    assert status["result"] is not None, f"Result was None, status: {status}"
    assert status["result"]["status"] == TaskStatus.FAILED.value
    assert "timed out" in (status["result"]["error"] or "").lower()

    await scheduler.stop()


@pytest.mark.asyncio
async def test_task_exception_handling(scheduler):
    """Test task exception handling."""
    async def failing_func():
        raise ValueError("Test error")

    task = ScheduledTask(
        name="test_exception",
        func=failing_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
        max_retries=1,
    )

    task_id = await scheduler.schedule_task(task)

    await scheduler.start()
    await asyncio.sleep(1.5)

    status = await scheduler.get_task_status(task_id)
    assert status is not None
    assert status["result"]["status"] == TaskStatus.FAILED.value
    assert "Test error" in (status["result"]["error"] or "")

    await scheduler.stop()


@pytest.mark.asyncio
async def test_list_scheduled_tasks(scheduler):
    """Test listing scheduled tasks."""
    async def dummy_func():
        return "done"

    task1 = ScheduledTask(
        name="task1",
        func=dummy_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(seconds=10),
    )

    task2 = ScheduledTask(
        name="task2",
        func=dummy_func,
        schedule_type=TaskScheduleType.RECURRING,
        interval_seconds=5,
    )

    await scheduler.schedule_task(task1)
    await scheduler.schedule_task(task2)

    tasks = await scheduler.list_scheduled_tasks()
    assert len(tasks) == 2
    assert any(t["name"] == "task1" for t in tasks)
    assert any(t["name"] == "task2" for t in tasks)


@pytest.mark.asyncio
async def test_list_scheduled_tasks_by_type(scheduler):
    """Test listing scheduled tasks by type."""
    async def dummy_func():
        return "done"

    task1 = ScheduledTask(
        name="recurring_task",
        func=dummy_func,
        schedule_type=TaskScheduleType.RECURRING,
        interval_seconds=5,
    )

    task2 = ScheduledTask(
        name="one_time_task",
        func=dummy_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(seconds=10),
    )

    await scheduler.schedule_task(task1)
    await scheduler.schedule_task(task2)

    recurring_tasks = await scheduler.list_scheduled_tasks(
        schedule_type=TaskScheduleType.RECURRING
    )
    assert len(recurring_tasks) == 1
    assert recurring_tasks[0]["name"] == "recurring_task"


@pytest.mark.asyncio
async def test_get_statistics(scheduler):
    """Test getting scheduler statistics."""
    async def dummy_func():
        return "done"

    task = ScheduledTask(
        name="test_stats",
        func=dummy_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
    )

    await scheduler.schedule_task(task)

    await scheduler.start()
    await asyncio.sleep(1.5)

    stats = await scheduler.get_statistics()
    assert stats["total_scheduled_tasks"] == 1
    assert stats["active_tasks"] == 1
    assert stats["completed_executions"] == 1

    await scheduler.stop()


@pytest.mark.asyncio
async def test_task_history(scheduler):
    """Test task history tracking."""
    execution_count = 0

    async def dummy_func():
        nonlocal execution_count
        execution_count += 1
        return f"result_{execution_count}"

    task = ScheduledTask(
        name="test_history",
        func=dummy_func,
        schedule_type=TaskScheduleType.RECURRING,
        interval_seconds=1,  # Match scheduler loop interval
    )

    task_id = await scheduler.schedule_task(task)

    await scheduler.start()
    await asyncio.sleep(4)  # Need at least 3-4s for 2 executions with 1s scheduler loop

    history = scheduler.get_task_history(task_id)
    assert len(history) >= 2, f"Expected at least 2 history entries, got {len(history)}"

    await scheduler.stop()


@pytest.mark.asyncio
async def test_inbox_integration(inbox):
    """Test that events are published to inbox."""
    scheduler = BackgroundTaskScheduler(inbox=inbox, max_workers=2)

    async def dummy_func():
        return "done"

    task = ScheduledTask(
        name="test_inbox",
        func=dummy_func,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
    )

    # Subscribe to events
    events = []

    async def collect_events():
        async for event in inbox.subscribe():
            events.append(event)
            if len(events) >= 2:  # Scheduled + completed
                break

    await scheduler.schedule_task(task)

    await scheduler.start()

    # Collect events with timeout
    try:
        await asyncio.wait_for(collect_events(), timeout=2)
    except asyncio.TimeoutError:
        pass

    await scheduler.stop()

    # Check that we got scheduling and completion events
    assert len(events) >= 1
    assert events[0].agent_name == "BackgroundTaskScheduler"


@pytest.mark.asyncio
async def test_invalid_one_time_task(scheduler):
    """Test that ONE_TIME task without run_at raises error."""
    async def dummy_func():
        return "done"

    task = ScheduledTask(
        name="invalid_task",
        func=dummy_func,
        schedule_type=TaskScheduleType.ONE_TIME,
    )

    with pytest.raises(ValueError, match="ONE_TIME tasks must have run_at"):
        await scheduler.schedule_task(task)


@pytest.mark.asyncio
async def test_invalid_recurring_task(scheduler):
    """Test that RECURRING task without interval raises error."""
    async def dummy_func():
        return "done"

    task = ScheduledTask(
        name="invalid_task",
        func=dummy_func,
        schedule_type=TaskScheduleType.RECURRING,
    )

    with pytest.raises(ValueError, match="RECURRING tasks must have positive interval"):
        await scheduler.schedule_task(task)


@pytest.mark.asyncio
async def test_concurrent_task_execution(scheduler):
    """Test that multiple tasks can run concurrently."""
    execution_order = []
    lock = asyncio.Lock()

    async def task1():
        async with lock:
            execution_order.append("t1_start")
        await asyncio.sleep(0.2)
        async with lock:
            execution_order.append("t1_end")

    async def task2():
        async with lock:
            execution_order.append("t2_start")
        await asyncio.sleep(0.2)
        async with lock:
            execution_order.append("t2_end")

    scheduled_task1 = ScheduledTask(
        name="concurrent_1",
        func=task1,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
    )

    scheduled_task2 = ScheduledTask(
        name="concurrent_2",
        func=task2,
        schedule_type=TaskScheduleType.ONE_TIME,
        run_at=datetime.now() + timedelta(milliseconds=100),
    )

    await scheduler.schedule_task(scheduled_task1)
    await scheduler.schedule_task(scheduled_task2)

    await scheduler.start()
    await asyncio.sleep(2)  # Wait for scheduler loop + task execution

    # Should have interleaved starts and ends if running concurrently
    assert "t1_start" in execution_order, f"execution_order was: {execution_order}"
    assert "t2_start" in execution_order
    t1_start_idx = execution_order.index("t1_start")
    t2_start_idx = execution_order.index("t2_start")
    # Both starts should happen before both ends
    assert abs(t1_start_idx - t2_start_idx) < 2

    await scheduler.stop()


@pytest.mark.asyncio
async def test_max_workers_limit(scheduler):
    """Test that max_workers limits concurrent execution."""
    running_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def slow_task():
        nonlocal running_count, max_concurrent
        async with lock:
            running_count += 1
            max_concurrent = max(max_concurrent, running_count)

        await asyncio.sleep(0.3)

        async with lock:
            running_count -= 1

    # Create tasks that would exceed max_workers
    for i in range(4):
        task = ScheduledTask(
            name=f"worker_task_{i}",
            func=slow_task,
            schedule_type=TaskScheduleType.ONE_TIME,
            run_at=datetime.now() + timedelta(milliseconds=100),
        )
        await scheduler.schedule_task(task)

    await scheduler.start()
    await asyncio.sleep(1)

    assert max_concurrent <= scheduler.max_workers

    await scheduler.stop()
