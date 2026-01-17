"""
Tests for the Live Sync Module.

Tests cover:
- Sync Protocol (messages, backoff)
- Pattern Subscriber (events, subscriptions)
- Sync Engine (push/pull)
- Conflict Resolver (version vectors, strategies)
- Sync Hooks (middleware chain)
- Sync Monitor (health tracking, alerts)
"""

import json
import pytest
import time

from src.evaluation.immune_system.live_sync import (
    # Protocol
    SyncEventType,
    SyncMessage,
    BackoffStrategy,
    PatternSubscriber,
    PatternEvent,
    ConnectionStatus,
    # Engine
    SyncEngine,
    PatternChange,
    ConflictResolver,
    ConflictStrategy,
    VersionVector,
    Pattern,
    SyncHooks,
    SyncContext,
    HookEventType,
    # Monitor
    SyncHealthMonitor,
    SyncStatus,
)


# ====================
# Sync Protocol Tests
# ====================

class TestSyncEventType:
    """Tests for SyncEventType enum."""

    def test_event_types_exist(self):
        """All expected event types should exist."""
        assert SyncEventType.CONNECT.value == "connect"
        assert SyncEventType.HEARTBEAT.value == "heartbeat"
        assert SyncEventType.PATTERN_CREATED.value == "pattern_created"
        assert SyncEventType.PATTERN_UPDATED.value == "pattern_updated"
        assert SyncEventType.PATTERN_DELETED.value == "pattern_deleted"


class TestSyncMessage:
    """Tests for SyncMessage dataclass."""

    def test_create_message(self):
        """Should create message with default id."""
        msg = SyncMessage(
            type=SyncEventType.HEARTBEAT,
            payload={"timestamp": 123456}
        )
        assert msg.type == SyncEventType.HEARTBEAT
        assert msg.payload == {"timestamp": 123456}
        assert msg.id is not None
        assert len(msg.id) == 36  # UUID length

    def test_to_json(self):
        """Should serialize to valid JSON."""
        msg = SyncMessage(
            type=SyncEventType.PATTERN_CREATED,
            payload={"pattern_id": "test123"},
            id="fixed-id"
        )
        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["id"] == "fixed-id"
        assert parsed["type"] == "pattern_created"
        assert parsed["payload"]["pattern_id"] == "test123"

    def test_from_json(self):
        """Should deserialize from JSON."""
        json_str = '{"id": "abc", "type": "heartbeat", "payload": {"ts": 1}}'
        msg = SyncMessage.from_json(json_str)

        assert msg.id == "abc"
        assert msg.type == SyncEventType.HEARTBEAT
        assert msg.payload["ts"] == 1

    def test_from_json_invalid(self):
        """Should raise on invalid JSON."""
        with pytest.raises(Exception):
            SyncMessage.from_json("not json")


class TestBackoffStrategy:
    """Tests for BackoffStrategy."""

    def test_initial_delay(self):
        """First delay should be close to base."""
        backoff = BackoffStrategy(base_delay=1.0, max_delay=60.0)
        delay = backoff.get_delay()
        assert 1.0 <= delay < 1.2  # Base + up to 10% jitter

    def test_exponential_increase(self):
        """Delays should increase exponentially."""
        backoff = BackoffStrategy(base_delay=1.0, max_delay=60.0)
        delays = [backoff.get_delay() for _ in range(5)]

        # Each delay should be roughly double the previous (ignoring jitter)
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]

    def test_max_delay_cap(self):
        """Delays should not exceed max."""
        backoff = BackoffStrategy(base_delay=1.0, max_delay=10.0)
        for _ in range(10):
            backoff.get_delay()

        delay = backoff.get_delay()
        assert delay <= 11.0  # max + jitter

    def test_reset(self):
        """Reset should restore initial state."""
        backoff = BackoffStrategy(base_delay=1.0)
        for _ in range(5):
            backoff.get_delay()

        backoff.reset()
        delay = backoff.get_delay()
        assert 1.0 <= delay < 1.2


# ====================
# Pattern Subscriber Tests
# ====================

class TestPatternEvent:
    """Tests for PatternEvent."""

    def test_from_json(self):
        """Should parse valid JSON."""
        data = json.dumps({
            "event_id": "evt-1",
            "project_id": "proj-a",
            "event_type": "PATTERN_UPDATE",
            "timestamp": 12345.0,
            "payload": {"action": "add"}
        })
        event = PatternEvent.from_json(data)

        assert event.event_id == "evt-1"
        assert event.project_id == "proj-a"
        assert event.payload["action"] == "add"

    def test_from_json_invalid(self):
        """Should raise on missing fields."""
        with pytest.raises(ValueError):
            PatternEvent.from_json('{"event_id": "1"}')

    def test_to_json(self):
        """Should serialize to JSON."""
        event = PatternEvent(
            event_id="e1",
            project_id="p1",
            event_type="UPDATE",
            timestamp=100.0,
            payload={"key": "val"}
        )
        data = json.loads(event.to_json())
        assert data["event_id"] == "e1"
        assert data["payload"]["key"] == "val"


class TestPatternSubscriber:
    """Tests for PatternSubscriber."""

    def test_initial_status(self):
        """Should start disconnected."""
        sub = PatternSubscriber(
            endpoint_url="wss://test.local/sync",
            auth_token="token"
        )
        assert sub.status == ConnectionStatus.DISCONNECTED

    def test_subscribe(self):
        """Should register subscriptions."""
        sub = PatternSubscriber(
            endpoint_url="wss://test.local/sync",
            auth_token="token"
        )
        received = []

        async def handler(project_id, payload):
            received.append((project_id, payload))

        sub.subscribe("proj-1", handler)
        assert "proj-1" in sub._subscriptions
        assert "proj-1" in sub._callbacks

    def test_unsubscribe(self):
        """Should remove subscriptions."""
        sub = PatternSubscriber(
            endpoint_url="wss://test.local/sync",
            auth_token="token"
        )

        async def handler(project_id, payload):
            pass

        sub.subscribe("proj-1", handler)
        sub.unsubscribe("proj-1")

        assert "proj-1" not in sub._subscriptions
        assert "proj-1" not in sub._callbacks


# ====================
# Sync Engine Tests
# ====================

class TestSyncEngine:
    """Tests for SyncEngine."""

    def test_register_peer(self):
        """Should register peers with correct flags."""
        engine = SyncEngine(project_id="local")
        engine.register_peer("remote-1", is_subscriber=True, is_subscription=True)

        assert "remote-1" in engine.subscribers
        assert "remote-1" in engine.subscriptions
        assert "remote-1" in engine._sync_states

    def test_get_sync_state(self):
        """Should return sync state for registered peer."""
        engine = SyncEngine(project_id="local")
        engine.register_peer("peer-a", is_subscriber=True)

        state = engine.get_sync_state("peer-a")
        assert state is not None
        assert state.peer_id == "peer-a"
        assert state.last_pushed_version == 0

    def test_get_sync_state_unregistered(self):
        """Should return None for unregistered peer."""
        engine = SyncEngine(project_id="local")
        assert engine.get_sync_state("unknown") is None


class TestPatternChange:
    """Tests for PatternChange dataclass."""

    def test_create_change(self):
        """Should create pattern change."""
        change = PatternChange(
            pattern_id="pat-1",
            version=5,
            timestamp=time.time(),
            data={"name": "test"},
            deleted=False
        )
        assert change.pattern_id == "pat-1"
        assert change.version == 5
        assert not change.deleted


# ====================
# Conflict Resolver Tests
# ====================

class TestVersionVector:
    """Tests for VersionVector."""

    def test_increment(self):
        """Should increment node counter."""
        vv = VersionVector()
        vv.increment("node-a")
        vv.increment("node-a")

        assert vv.clocks["node-a"] == 2

    def test_merge(self):
        """Should merge with max values."""
        vv1 = VersionVector({"a": 2, "b": 1})
        vv2 = VersionVector({"a": 1, "c": 3})

        merged = vv1.merge(vv2)

        assert merged.clocks["a"] == 2
        assert merged.clocks["b"] == 1
        assert merged.clocks["c"] == 3

    def test_compare_identical(self):
        """Should return 0 for identical vectors."""
        vv1 = VersionVector({"a": 1, "b": 2})
        vv2 = VersionVector({"a": 1, "b": 2})

        assert vv1.compare(vv2) == 0

    def test_compare_less_than(self):
        """Should return -1 when causally precedes."""
        vv1 = VersionVector({"a": 1})
        vv2 = VersionVector({"a": 2})

        assert vv1.compare(vv2) == -1

    def test_compare_greater_than(self):
        """Should return 1 when causally succeeds."""
        vv1 = VersionVector({"a": 2})
        vv2 = VersionVector({"a": 1})

        assert vv1.compare(vv2) == 1

    def test_compare_concurrent(self):
        """Should return None for concurrent modifications."""
        vv1 = VersionVector({"a": 2, "b": 1})
        vv2 = VersionVector({"a": 1, "b": 2})

        assert vv1.compare(vv2) is None


class TestConflictResolver:
    """Tests for ConflictResolver."""

    def test_accept_new_pattern(self):
        """Should accept new patterns."""
        resolver = ConflictResolver(node_id="node-1")
        remote = Pattern(
            pattern_id="pat-1",
            node_id="node-2",
            data={"key": "value"},
            timestamp=time.time()
        )

        result, changed = resolver.resolve(None, remote)

        assert changed
        assert result.pattern_id == "pat-1"

    def test_update_stale_local(self):
        """Should update when local is stale."""
        resolver = ConflictResolver(node_id="node-1")

        local_vv = VersionVector({"node-2": 1})
        remote_vv = VersionVector({"node-2": 2})

        local = Pattern(
            pattern_id="pat-1",
            node_id="node-1",
            data={"old": True},
            timestamp=100.0,
            version_vector=local_vv
        )
        remote = Pattern(
            pattern_id="pat-1",
            node_id="node-2",
            data={"new": True},
            timestamp=200.0,
            version_vector=remote_vv
        )

        result, changed = resolver.resolve(local, remote)

        assert changed
        assert result.data["new"]

    def test_ignore_stale_remote(self):
        """Should ignore when remote is stale."""
        resolver = ConflictResolver(node_id="node-1")

        local_vv = VersionVector({"node-1": 2})
        remote_vv = VersionVector({"node-1": 1})

        local = Pattern(
            pattern_id="pat-1",
            node_id="node-1",
            data={"current": True},
            timestamp=200.0,
            version_vector=local_vv
        )
        remote = Pattern(
            pattern_id="pat-1",
            node_id="node-2",
            data={"old": True},
            timestamp=100.0,
            version_vector=remote_vv
        )

        result, changed = resolver.resolve(local, remote)

        assert not changed
        assert result.data["current"]

    def test_lww_conflict_resolution(self):
        """Should resolve conflicts with LWW."""
        resolver = ConflictResolver(
            node_id="node-1",
            strategy=ConflictStrategy.LAST_WRITE_WINS
        )

        # Concurrent modifications
        local_vv = VersionVector({"node-1": 1})
        remote_vv = VersionVector({"node-2": 1})

        local = Pattern(
            pattern_id="pat-1",
            node_id="node-1",
            data={"source": "local"},
            timestamp=100.0,
            version_vector=local_vv
        )
        remote = Pattern(
            pattern_id="pat-1",
            node_id="node-2",
            data={"source": "remote"},
            timestamp=200.0,  # Remote is newer
            version_vector=remote_vv
        )

        result, changed = resolver.resolve(local, remote)

        assert changed
        assert result.data["source"] == "remote"
        assert len(resolver.audit_trail) == 1


# ====================
# Sync Hooks Tests
# ====================

class TestSyncHooks:
    """Tests for SyncHooks."""

    @pytest.mark.asyncio
    async def test_register_and_emit(self):
        """Should register and emit hooks."""
        hooks = SyncHooks()
        called = []

        async def my_hook(ctx: SyncContext):
            called.append(ctx.project_id)

        hooks.register(HookEventType.BEFORE_PUSH, my_hook)

        ctx = SyncContext(
            event_type=HookEventType.BEFORE_PUSH,
            project_id="proj-1",
            payload=[]
        )
        await hooks.emit(HookEventType.BEFORE_PUSH, ctx)

        assert called == ["proj-1"]

    @pytest.mark.asyncio
    async def test_decorator_registration(self):
        """Should register hooks via decorators."""
        hooks = SyncHooks()
        called = []

        @hooks.before_push()
        async def before_hook(ctx: SyncContext):
            called.append("before")

        @hooks.after_push()
        async def after_hook(ctx: SyncContext):
            called.append("after")

        ctx = SyncContext(
            event_type=HookEventType.BEFORE_PUSH,
            project_id="p1",
            payload=[]
        )

        await hooks.emit(HookEventType.BEFORE_PUSH, ctx)
        await hooks.emit(HookEventType.AFTER_PUSH, ctx)

        assert called == ["before", "after"]

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Should execute hooks in priority order."""
        hooks = SyncHooks()
        order = []

        @hooks.before_push(priority=50)
        async def first(ctx: SyncContext):
            order.append("first")

        @hooks.before_push(priority=100)
        async def second(ctx: SyncContext):
            order.append("second")

        @hooks.before_push(priority=10)
        async def zeroth(ctx: SyncContext):
            order.append("zeroth")

        ctx = SyncContext(
            event_type=HookEventType.BEFORE_PUSH,
            project_id="p1",
            payload=[]
        )
        await hooks.emit(HookEventType.BEFORE_PUSH, ctx)

        assert order == ["zeroth", "first", "second"]

    @pytest.mark.asyncio
    async def test_abort_stops_chain(self):
        """Should stop chain when aborted."""
        hooks = SyncHooks()
        called = []

        @hooks.before_push(priority=1)
        async def first(ctx: SyncContext):
            called.append("first")
            ctx.abort("Testing abort")

        @hooks.before_push(priority=2)
        async def second(ctx: SyncContext):
            called.append("second")

        ctx = SyncContext(
            event_type=HookEventType.BEFORE_PUSH,
            project_id="p1",
            payload=[]
        )
        await hooks.emit(HookEventType.BEFORE_PUSH, ctx)

        assert called == ["first"]
        assert ctx.is_aborted
        assert ctx.metadata["abort_reason"] == "Testing abort"

    @pytest.mark.asyncio
    async def test_error_triggers_on_error(self):
        """Should trigger ON_ERROR on exception."""
        hooks = SyncHooks()
        error_handled = []

        @hooks.before_push()
        async def failing_hook(ctx: SyncContext):
            raise ValueError("Test error")

        @hooks.on_error()
        async def error_handler(ctx: SyncContext):
            error_handled.append(ctx.errors[0])

        ctx = SyncContext(
            event_type=HookEventType.BEFORE_PUSH,
            project_id="p1",
            payload=[]
        )
        await hooks.emit(HookEventType.BEFORE_PUSH, ctx)

        assert len(error_handled) == 1
        assert isinstance(error_handled[0], ValueError)


# ====================
# Sync Monitor Tests
# ====================

class TestSyncHealthMonitor:
    """Tests for SyncHealthMonitor."""

    def test_record_success(self):
        """Should record successful sync."""
        monitor = SyncHealthMonitor()
        monitor.record_sync_success("proj-1", latency_ms=50.0)

        metrics = monitor.get_dashboard_metrics()
        project = next(
            p for p in metrics["projects"]
            if p["project_id"] == "proj-1"
        )

        assert project["status"] == "healthy"
        assert project["latency_ms"] == 50.0
        assert project["failures_consecutive"] == 0

    def test_record_failure(self):
        """Should record failed sync."""
        monitor = SyncHealthMonitor()
        monitor.record_sync_failure("proj-1", "Connection timeout")

        metrics = monitor.get_dashboard_metrics()
        project = next(
            p for p in metrics["projects"]
            if p["project_id"] == "proj-1"
        )

        assert project["status"] == "degraded"
        assert project["failures_consecutive"] == 1

    def test_critical_on_consecutive_failures(self):
        """Should go critical after max failures."""
        monitor = SyncHealthMonitor()

        for _ in range(3):
            monitor.record_sync_failure("proj-1", "Error")

        metrics = monitor.get_dashboard_metrics()
        project = next(
            p for p in metrics["projects"]
            if p["project_id"] == "proj-1"
        )

        assert project["status"] == "critical"
        assert project["failures_consecutive"] == 3

    def test_critical_on_high_latency(self):
        """Should go critical on very high latency."""
        monitor = SyncHealthMonitor(latency_critical_ms=1000.0)
        monitor.record_sync_success("proj-1", latency_ms=5500.0)

        metrics = monitor.get_dashboard_metrics()
        project = next(
            p for p in metrics["projects"]
            if p["project_id"] == "proj-1"
        )

        assert project["status"] == "critical"

    def test_degraded_on_moderate_latency(self):
        """Should go degraded on moderate latency."""
        monitor = SyncHealthMonitor()
        monitor.record_sync_success("proj-1", latency_ms=1500.0)

        metrics = monitor.get_dashboard_metrics()
        project = next(
            p for p in metrics["projects"]
            if p["project_id"] == "proj-1"
        )

        assert project["status"] == "degraded"

    def test_check_health_and_alert(self):
        """Should generate alerts for unhealthy projects."""
        monitor = SyncHealthMonitor()

        # Create critical project
        for _ in range(3):
            monitor.record_sync_failure("proj-fail", "Error")

        # Create healthy project
        monitor.record_sync_success("proj-ok", latency_ms=50.0)

        alerts = monitor.check_health_and_alert()

        critical_alerts = [a for a in alerts if a.severity == SyncStatus.CRITICAL]
        assert len(critical_alerts) == 1
        assert critical_alerts[0].project_id == "proj-fail"

    def test_dashboard_summary(self):
        """Should provide correct summary counts."""
        monitor = SyncHealthMonitor()

        monitor.record_sync_success("healthy-1", 50.0)
        monitor.record_sync_success("healthy-2", 50.0)
        monitor.record_sync_success("degraded-1", 1500.0)
        for _ in range(3):
            monitor.record_sync_failure("critical-1", "Error")

        metrics = monitor.get_dashboard_metrics()

        assert metrics["summary"]["total_projects"] == 4
        assert metrics["summary"]["healthy"] == 2
        assert metrics["summary"]["degraded"] == 1
        assert metrics["summary"]["critical"] == 1

    def test_get_project_status(self):
        """Should return status for specific project."""
        monitor = SyncHealthMonitor()
        monitor.record_sync_success("proj-1", 100.0)

        status = monitor.get_project_status("proj-1")

        assert status["project_id"] == "proj-1"
        assert status["health_status"] == "healthy"
        assert status["total_syncs"] == 1

    def test_get_project_status_unknown(self):
        """Should return None for unknown project."""
        monitor = SyncHealthMonitor()
        assert monitor.get_project_status("unknown") is None
