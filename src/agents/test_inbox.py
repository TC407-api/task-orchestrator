"""
Tests for Universal Inbox / Approval Queue system.

Tests cover:
- Event publishing and subscription
- Approval workflows
- SQLite persistence
- Decorator usage
- Expiration handling
"""
import asyncio
import pytest
from datetime import datetime, timedelta

from .inbox import (
    UniversalInbox,
    AgentEvent,
    EventType,
    ActionRiskLevel,
    ApprovalStatus,
    PendingAction,
    requires_approval,
)


class TestAgentEvent:
    """Test AgentEvent dataclass."""

    def test_event_creation(self):
        """Test creating an event."""
        event = AgentEvent(
            event_type=EventType.AGENT_START,
            agent_name="test_agent",
            data={"test": "data"},
        )

        assert event.event_type == EventType.AGENT_START
        assert event.agent_name == "test_agent"
        assert event.data == {"test": "data"}
        assert event.event_id is not None

    def test_event_to_dict(self):
        """Test event serialization."""
        event = AgentEvent(
            event_type=EventType.TEXT_OUTPUT,
            agent_name="agent1",
            data={"message": "hello"},
        )

        data = event.to_dict()
        assert data["event_type"] == "TEXT_OUTPUT"
        assert data["agent_name"] == "agent1"
        assert data["data"] == {"message": "hello"}


class TestPendingAction:
    """Test PendingAction dataclass."""

    def test_action_creation(self):
        """Test creating a pending action."""
        action = PendingAction(
            action_type="delete_file",
            description="Delete test.txt",
            agent_name="file_agent",
            risk_level=ActionRiskLevel.HIGH,
            payload={"path": "/tmp/test.txt"},
        )

        assert action.action_type == "delete_file"
        assert action.status == ApprovalStatus.PENDING
        assert not action.is_expired()

    def test_action_expiration(self):
        """Test action expiration."""
        action = PendingAction(
            action_type="delete_file",
            expires_at=datetime.now() - timedelta(seconds=1),
        )

        assert action.is_expired()

    def test_action_to_dict(self):
        """Test action serialization."""
        action = PendingAction(
            action_type="send_email",
            description="Send to user@example.com",
            risk_level=ActionRiskLevel.MEDIUM,
        )

        data = action.to_dict()
        assert data["action_type"] == "send_email"
        assert data["risk_level"] == "MEDIUM"
        assert data["status"] == "PENDING"


@pytest.mark.asyncio
class TestUniversalInbox:
    """Test UniversalInbox class."""

    async def test_inbox_creation(self):
        """Test creating an inbox."""
        inbox = UniversalInbox()
        assert inbox is not None
        assert inbox.db_path == ":memory:"

    async def test_inbox_with_file_db(self, tmp_path):
        """Test inbox with file-based database."""
        db_file = tmp_path / "test.db"
        inbox = UniversalInbox(str(db_file))

        assert inbox.db_path == str(db_file)
        assert db_file.exists()

    async def test_publish_event(self):
        """Test publishing an event."""
        inbox = UniversalInbox()
        event = AgentEvent(
            event_type=EventType.AGENT_START,
            agent_name="test_agent",
        )

        await inbox.publish(event)
        # Event should be stored without error

    async def test_subscribe_events(self):
        """Test subscribing to events."""
        inbox = UniversalInbox()

        received_events = []

        async def subscriber():
            async for event in inbox.subscribe():
                received_events.append(event)
                if len(received_events) >= 2:
                    break

        async def publisher():
            await asyncio.sleep(0.1)
            for i in range(2):
                event = AgentEvent(
                    event_type=EventType.TEXT_OUTPUT,
                    agent_name=f"agent{i}",
                )
                await inbox.publish(event)

        await asyncio.gather(
            subscriber(),
            publisher(),
            return_exceptions=True,
        )

        assert len(received_events) >= 1

    async def test_require_approval(self):
        """Test creating an approval request."""
        inbox = UniversalInbox()

        action = await inbox.require_approval(
            action_type="delete_file",
            description="Delete /tmp/test.txt",
            agent_name="file_agent",
            payload={"path": "/tmp/test.txt"},
            risk_level=ActionRiskLevel.CRITICAL,
        )

        assert action.status == ApprovalStatus.PENDING
        assert action.action_type == "delete_file"
        assert action.risk_level == ActionRiskLevel.CRITICAL

    async def test_get_pending_approvals(self):
        """Test retrieving pending approvals."""
        inbox = UniversalInbox()

        # Create multiple approval requests
        for i in range(3):
            await inbox.require_approval(
                action_type=f"action{i}",
                description=f"Test action {i}",
                agent_name="test_agent",
                payload={},
                risk_level=ActionRiskLevel.HIGH if i < 2 else ActionRiskLevel.LOW,
            )

        pending = inbox.get_pending_approvals()
        assert len(pending) == 3

        # Filter by risk level
        high_risk = inbox.get_pending_approvals(
            risk_level=ActionRiskLevel.HIGH
        )
        assert len(high_risk) == 2

    async def test_approve_action(self):
        """Test approving an action."""
        inbox = UniversalInbox()

        action = await inbox.require_approval(
            action_type="delete_file",
            description="Delete test.txt",
            agent_name="file_agent",
            payload={"path": "/tmp/test.txt"},
        )

        # Approve the action
        approved = await inbox.approve(
            action.action_id,
            approved_by="user123",
        )

        assert approved.status == ApprovalStatus.APPROVED
        assert approved.approved_by == "user123"
        assert approved.approved_at is not None

    async def test_approve_with_callback(self):
        """Test approval with execution callback."""
        inbox = UniversalInbox()

        action = await inbox.require_approval(
            action_type="send_email",
            description="Send test email",
            agent_name="email_agent",
            payload={"to": "test@example.com"},
        )

        async def send_email(pending_action):
            return f"Email sent to {pending_action.payload['to']}"

        approved = await inbox.approve(
            action.action_id,
            approved_by="system",
            execute_callback=send_email,
        )

        assert approved.status == ApprovalStatus.APPROVED
        assert approved.execution_result is not None
        assert "Email sent" in approved.execution_result

    async def test_reject_action(self):
        """Test rejecting an action."""
        inbox = UniversalInbox()

        action = await inbox.require_approval(
            action_type="delete_file",
            description="Delete test.txt",
            agent_name="file_agent",
            payload={"path": "/tmp/test.txt"},
        )

        # Reject the action
        rejected = await inbox.reject(
            action.action_id,
            reason="Suspicious operation",
            rejected_by="security_gate",
        )

        assert rejected.status == ApprovalStatus.REJECTED
        assert rejected.rejection_reason == "Suspicious operation"

    async def test_cannot_approve_expired_action(self):
        """Test that expired actions cannot be approved."""
        inbox = UniversalInbox()

        action = await inbox.require_approval(
            action_type="test",
            description="Test",
            agent_name="agent",
            payload={},
            timeout_seconds=0,  # Expires immediately
        )

        await asyncio.sleep(0.1)  # Wait for expiration

        with pytest.raises(ValueError, match="Approval expired"):
            await inbox.approve(action.action_id)

    async def test_cannot_approve_twice(self):
        """Test that actions cannot be approved twice."""
        inbox = UniversalInbox()

        action = await inbox.require_approval(
            action_type="test",
            description="Test",
            agent_name="agent",
            payload={},
        )

        # Approve once
        await inbox.approve(action.action_id)

        # Try to approve again
        with pytest.raises(ValueError, match="already"):
            await inbox.approve(action.action_id)

    async def test_get_action(self):
        """Test retrieving a specific action."""
        inbox = UniversalInbox()

        action = await inbox.require_approval(
            action_type="test",
            description="Test",
            agent_name="agent",
            payload={},
        )

        retrieved = inbox.get_action(action.action_id)
        assert retrieved is not None
        assert retrieved.action_id == action.action_id

    async def test_get_event_history(self):
        """Test retrieving event history."""
        inbox = UniversalInbox()

        # Publish events
        for i in range(3):
            event = AgentEvent(
                event_type=EventType.TEXT_OUTPUT if i < 2 else EventType.ERROR,
                agent_name="test_agent",
            )
            await inbox.publish(event)

        history = inbox.get_event_history()
        assert len(history) >= 3

        # Filter by event type
        text_events = inbox.get_event_history(event_type=EventType.TEXT_OUTPUT)
        assert len(text_events) >= 2

    async def test_clear_expired_actions(self):
        """Test clearing expired actions."""
        inbox = UniversalInbox()

        # Create some expired actions
        for _ in range(2):
            await inbox.require_approval(
                action_type="test",
                description="Test",
                agent_name="agent",
                payload={},
                timeout_seconds=0,
            )

        await asyncio.sleep(0.1)

        cleared = await inbox.clear_expired_actions()
        assert cleared >= 2

    async def test_export_data(self):
        """Test exporting all inbox data."""
        inbox = UniversalInbox()

        # Add some data
        event = AgentEvent(
            event_type=EventType.AGENT_START,
            agent_name="test_agent",
        )
        await inbox.publish(event)

        await inbox.require_approval(
            action_type="test",
            description="Test",
            agent_name="agent",
            payload={},
        )

        exported = inbox.export_data()
        assert "pending_actions" in exported
        assert "events" in exported
        assert "action_history" in exported
        assert "exported_at" in exported


@pytest.mark.asyncio
class TestRequiresApprovalDecorator:
    """Test @requires_approval decorator."""

    async def test_async_function_with_approval(self):
        """Test decorating an async function."""
        UniversalInbox()

        @requires_approval(
            action_type="send_message",
            risk_level=ActionRiskLevel.HIGH,
        )
        async def send_message(message: str, inbox: UniversalInbox):
            return f"Sent: {message}"

        # This should raise because no approval was given
        # In real use, approvals come from external approval loop

    async def test_requires_approval_without_inbox(self):
        """Test that decorator requires inbox parameter."""
        @requires_approval(action_type="test")
        async def test_func():
            return "test"

        with pytest.raises(ValueError, match="inbox"):
            await test_func()


@pytest.mark.asyncio
class TestWorkflows:
    """Test complete workflows."""

    async def test_high_risk_action_workflow(self):
        """Test complete workflow for high-risk action."""
        inbox = UniversalInbox()

        # 1. Action requests approval
        action = await inbox.require_approval(
            action_type="delete_database",
            description="Delete production database backup",
            agent_name="maintenance_agent",
            payload={"database": "prod_backup_db"},
            risk_level=ActionRiskLevel.CRITICAL,
            timeout_seconds=10,
        )

        # 2. Check pending approvals
        pending = inbox.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0].action_id == action.action_id

        # 3. Get full action details
        action_details = inbox.get_action(action.action_id)
        assert action_details is not None
        assert action_details.description == "Delete production database backup"

        # 4. Approve the action
        approved = await inbox.approve(
            action.action_id,
            approved_by="admin_user",
        )

        # 5. Verify approval
        assert approved.status == ApprovalStatus.APPROVED
        assert approved.approved_by == "admin_user"

        # 6. Check no more pending
        pending = inbox.get_pending_approvals()
        assert len(pending) == 0

    async def test_rejection_workflow(self):
        """Test rejection workflow."""
        inbox = UniversalInbox()

        action = await inbox.require_approval(
            action_type="modify_config",
            description="Modify system configuration",
            agent_name="config_agent",
            payload={"config": "system.conf"},
            risk_level=ActionRiskLevel.HIGH,
        )

        # Reject due to safety concerns
        rejected = await inbox.reject(
            action.action_id,
            reason="Insufficient change control approval",
            rejected_by="security_team",
        )

        assert rejected.status == ApprovalStatus.REJECTED
        assert rejected.rejection_reason == "Insufficient change control approval"

    async def test_multiple_concurrent_approvals(self):
        """Test handling multiple concurrent approval requests."""
        inbox = UniversalInbox()

        # Create multiple approval requests
        actions = []
        for i in range(5):
            action = await inbox.require_approval(
                action_type=f"action_{i}",
                description=f"Test action {i}",
                agent_name=f"agent_{i % 2}",
                payload={"index": i},
                risk_level=ActionRiskLevel.MEDIUM,
            )
            actions.append(action)

        # Get all pending
        pending = inbox.get_pending_approvals()
        assert len(pending) == 5

        # Approve some, reject others
        await inbox.approve(actions[0].action_id)
        await inbox.approve(actions[1].action_id)
        await inbox.reject(actions[2].action_id, reason="Failed validation")

        # Check status
        pending = inbox.get_pending_approvals()
        assert len(pending) == 2  # actions[3] and actions[4]

    async def test_event_audit_trail(self):
        """Test complete event audit trail."""
        inbox = UniversalInbox()

        # Create approval request
        action = await inbox.require_approval(
            action_type="export_data",
            description="Export user data",
            agent_name="analytics_agent",
            payload={"format": "json"},
            risk_level=ActionRiskLevel.HIGH,
        )

        # Publish some events
        events = [
            AgentEvent(
                event_type=EventType.AGENT_START,
                agent_name="analytics_agent",
                data={"task": "export"},
            ),
            AgentEvent(
                event_type=EventType.AWAITING_INPUT,
                agent_name="analytics_agent",
                related_approval_id=action.action_id,
            ),
        ]

        for event in events:
            await inbox.publish(event)

        # Approve
        await inbox.approve(action.action_id, approved_by="data_officer")

        # Get event history
        history = inbox.get_event_history(agent_name="analytics_agent")
        assert len(history) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
