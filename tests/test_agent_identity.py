"""Tests for Agent Identity Manager (NHI lifecycle management).

TDD RED Phase: These tests define the expected behavior for the AgentIdentityManager.
They should fail initially until the implementation is complete.
"""
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.governance.agent_identity import (
    AgentCredential,
    AgentIdentityManager,
    ArchetypeType,
)


class TestAgentCredential:
    """Tests for AgentCredential dataclass."""

    def test_credential_has_required_fields(self):
        """AgentCredential should have all required security fields."""
        now = datetime.now()
        expires = now + timedelta(days=90)

        credential = AgentCredential(
            agent_id="test-agent-123",
            public_key="-----BEGIN PUBLIC KEY-----\nMIIBIjAN...",
            archetype=ArchetypeType.BUILDER,
            capabilities=["code_write", "test_run"],
            created_at=now,
            expires_at=expires,
        )

        assert credential.agent_id == "test-agent-123"
        assert credential.public_key.startswith("-----BEGIN PUBLIC KEY-----")
        assert credential.archetype == ArchetypeType.BUILDER
        assert credential.capabilities == ["code_write", "test_run"]
        assert credential.created_at == now
        assert credential.expires_at == expires
        assert credential.revoked is False

    def test_credential_default_revoked_is_false(self):
        """AgentCredential revoked field should default to False."""
        credential = AgentCredential(
            agent_id="test-agent",
            public_key="key",
            archetype=ArchetypeType.RESEARCHER,
            capabilities=[],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90),
        )
        assert credential.revoked is False

    def test_credential_is_expired(self):
        """is_expired() should return True for expired credentials."""
        now = datetime.now()
        expired_credential = AgentCredential(
            agent_id="test-agent",
            public_key="key",
            archetype=ArchetypeType.ARCHITECT,
            capabilities=[],
            created_at=now - timedelta(days=100),
            expires_at=now - timedelta(days=10),  # Expired 10 days ago
        )
        assert expired_credential.is_expired() is True

    def test_credential_is_not_expired(self):
        """is_expired() should return False for valid credentials."""
        now = datetime.now()
        valid_credential = AgentCredential(
            agent_id="test-agent",
            public_key="key",
            archetype=ArchetypeType.QC,
            capabilities=[],
            created_at=now,
            expires_at=now + timedelta(days=90),
        )
        assert valid_credential.is_expired() is False


class TestAgentIdentityManagerCreation:
    """Tests for creating agent identities."""

    @pytest.fixture
    def manager(self) -> AgentIdentityManager:
        """Create an AgentIdentityManager with temp storage."""
        with TemporaryDirectory() as tmpdir:
            yield AgentIdentityManager(storage_path=Path(tmpdir))

    def test_create_agent_generates_keypair(self, manager: AgentIdentityManager):
        """create_agent() should generate a valid RSA keypair."""
        credential = manager.create_agent(
            archetype=ArchetypeType.BUILDER,
            capabilities=["code_write", "code_read"],
        )

        assert credential.agent_id is not None
        assert len(credential.agent_id) == 36  # UUID format
        assert credential.public_key.startswith("-----BEGIN PUBLIC KEY-----")
        assert credential.archetype == ArchetypeType.BUILDER
        assert credential.capabilities == ["code_write", "code_read"]

    def test_agent_credential_expires_after_90_days(self, manager: AgentIdentityManager):
        """Created credentials should expire after 90 days by default."""
        before = datetime.now()
        credential = manager.create_agent(
            archetype=ArchetypeType.RESEARCHER,
            capabilities=["web_search"],
        )
        after = datetime.now()

        # Expiry should be roughly 90 days from creation
        expected_expiry_min = before + timedelta(days=89, hours=23)
        expected_expiry_max = after + timedelta(days=90, hours=1)

        assert expected_expiry_min <= credential.expires_at <= expected_expiry_max

    def test_create_agent_persists_credential(self, manager: AgentIdentityManager):
        """Created credentials should be persisted to storage."""
        credential = manager.create_agent(
            archetype=ArchetypeType.ARCHITECT,
            capabilities=["design"],
        )

        # Should be retrievable
        retrieved = manager.get_agent(credential.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == credential.agent_id
        assert retrieved.archetype == credential.archetype

    def test_create_agent_stores_private_key_securely(self, manager: AgentIdentityManager):
        """Private key should be stored but not exposed in credential."""
        credential = manager.create_agent(
            archetype=ArchetypeType.QC,
            capabilities=["test_run"],
        )

        # Public key should be in credential
        assert "PUBLIC KEY" in credential.public_key
        # Private key should NOT be in the credential
        assert not hasattr(credential, 'private_key')
        # But manager should be able to sign with it
        signature = manager.sign_data(credential.agent_id, b"test data")
        assert signature is not None


class TestAgentIdentityManagerRotation:
    """Tests for credential rotation."""

    @pytest.fixture
    def manager_with_agent(self) -> tuple[AgentIdentityManager, AgentCredential]:
        """Create manager with an existing agent."""
        with TemporaryDirectory() as tmpdir:
            manager = AgentIdentityManager(storage_path=Path(tmpdir))
            credential = manager.create_agent(
                archetype=ArchetypeType.BUILDER,
                capabilities=["code_write"],
            )
            yield manager, credential

    def test_rotate_credential_creates_new_keypair(
        self, manager_with_agent: tuple[AgentIdentityManager, AgentCredential]
    ):
        """rotate_credential() should create a new keypair."""
        manager, old_credential = manager_with_agent

        new_credential = manager.rotate_credential(old_credential.agent_id)

        assert new_credential.agent_id == old_credential.agent_id  # Same ID
        assert new_credential.public_key != old_credential.public_key  # New key
        assert new_credential.archetype == old_credential.archetype
        assert new_credential.capabilities == old_credential.capabilities

    def test_rotate_credential_invalidates_old(
        self, manager_with_agent: tuple[AgentIdentityManager, AgentCredential]
    ):
        """rotate_credential() should mark old credential as revoked."""
        manager, old_credential = manager_with_agent
        old_public_key = old_credential.public_key

        new_credential = manager.rotate_credential(old_credential.agent_id)

        # Old key should no longer verify
        assert not manager.verify_agent(
            old_credential.agent_id,
            signature=b"old_sig",
            data=b"test",
            public_key=old_public_key,
        )

    def test_rotate_nonexistent_agent_raises(self, manager_with_agent):
        """rotate_credential() should raise for nonexistent agent."""
        manager, _ = manager_with_agent

        with pytest.raises(ValueError, match="Agent not found"):
            manager.rotate_credential("nonexistent-agent-id")


class TestAgentIdentityManagerRevocation:
    """Tests for credential revocation."""

    @pytest.fixture
    def manager_with_agent(self) -> tuple[AgentIdentityManager, AgentCredential]:
        """Create manager with an existing agent."""
        with TemporaryDirectory() as tmpdir:
            manager = AgentIdentityManager(storage_path=Path(tmpdir))
            credential = manager.create_agent(
                archetype=ArchetypeType.BUILDER,
                capabilities=["code_write"],
            )
            yield manager, credential

    def test_revoke_prevents_verification(
        self, manager_with_agent: tuple[AgentIdentityManager, AgentCredential]
    ):
        """revoke_credential() should prevent future verification."""
        manager, credential = manager_with_agent

        # Before revocation, verification should work
        signature = manager.sign_data(credential.agent_id, b"test data")
        assert manager.verify_agent(
            credential.agent_id,
            signature=signature,
            data=b"test data",
        )

        # Revoke
        manager.revoke_credential(credential.agent_id)

        # After revocation, verification should fail
        assert not manager.verify_agent(
            credential.agent_id,
            signature=signature,
            data=b"test data",
        )

    def test_revoke_marks_credential_as_revoked(
        self, manager_with_agent: tuple[AgentIdentityManager, AgentCredential]
    ):
        """revoke_credential() should set revoked=True."""
        manager, credential = manager_with_agent

        manager.revoke_credential(credential.agent_id)

        updated = manager.get_agent(credential.agent_id)
        assert updated.revoked is True

    def test_revoke_nonexistent_agent_raises(self, manager_with_agent):
        """revoke_credential() should raise for nonexistent agent."""
        manager, _ = manager_with_agent

        with pytest.raises(ValueError, match="Agent not found"):
            manager.revoke_credential("nonexistent-agent-id")


class TestAgentIdentityManagerVerification:
    """Tests for signature verification."""

    @pytest.fixture
    def manager_with_agent(self) -> tuple[AgentIdentityManager, AgentCredential]:
        """Create manager with an existing agent."""
        with TemporaryDirectory() as tmpdir:
            manager = AgentIdentityManager(storage_path=Path(tmpdir))
            credential = manager.create_agent(
                archetype=ArchetypeType.BUILDER,
                capabilities=["code_write"],
            )
            yield manager, credential

    def test_verify_agent_with_valid_signature(
        self, manager_with_agent: tuple[AgentIdentityManager, AgentCredential]
    ):
        """verify_agent() should return True for valid signatures."""
        manager, credential = manager_with_agent

        data = b"important action data"
        signature = manager.sign_data(credential.agent_id, data)

        assert manager.verify_agent(
            credential.agent_id,
            signature=signature,
            data=data,
        ) is True

    def test_verify_agent_rejects_invalid_signature(
        self, manager_with_agent: tuple[AgentIdentityManager, AgentCredential]
    ):
        """verify_agent() should return False for invalid signatures."""
        manager, credential = manager_with_agent

        assert manager.verify_agent(
            credential.agent_id,
            signature=b"fake_signature",
            data=b"some data",
        ) is False

    def test_verify_agent_rejects_tampered_data(
        self, manager_with_agent: tuple[AgentIdentityManager, AgentCredential]
    ):
        """verify_agent() should return False if data was tampered."""
        manager, credential = manager_with_agent

        data = b"original data"
        signature = manager.sign_data(credential.agent_id, data)

        # Tamper with data
        tampered_data = b"tampered data"

        assert manager.verify_agent(
            credential.agent_id,
            signature=signature,
            data=tampered_data,
        ) is False


class TestAgentIdentityManagerExpiry:
    """Tests for listing expiring credentials."""

    @pytest.fixture
    def manager(self) -> AgentIdentityManager:
        """Create an AgentIdentityManager with temp storage."""
        with TemporaryDirectory() as tmpdir:
            yield AgentIdentityManager(storage_path=Path(tmpdir))

    def test_list_expiring_agents(self, manager: AgentIdentityManager):
        """list_expiring() should return agents expiring within N days."""
        # Create agent with custom expiry (5 days from now)
        credential = manager.create_agent(
            archetype=ArchetypeType.BUILDER,
            capabilities=["code_write"],
            expires_in_days=5,
        )

        # Create agent with default expiry (90 days)
        manager.create_agent(
            archetype=ArchetypeType.RESEARCHER,
            capabilities=["search"],
        )

        # List agents expiring in 7 days
        expiring = manager.list_expiring(days=7)

        assert len(expiring) == 1
        assert expiring[0].agent_id == credential.agent_id

    def test_list_expiring_empty_when_none_expiring(self, manager: AgentIdentityManager):
        """list_expiring() should return empty list when no agents expiring."""
        # Create agent with default 90-day expiry
        manager.create_agent(
            archetype=ArchetypeType.BUILDER,
            capabilities=["code_write"],
        )

        # List agents expiring in 7 days - should be empty
        expiring = manager.list_expiring(days=7)

        assert len(expiring) == 0


class TestArchetypeType:
    """Tests for ArchetypeType enum."""

    def test_all_archetypes_defined(self):
        """All expected archetypes should be defined."""
        assert ArchetypeType.ARCHITECT is not None
        assert ArchetypeType.BUILDER is not None
        assert ArchetypeType.QC is not None
        assert ArchetypeType.RESEARCHER is not None

    def test_archetype_values(self):
        """Archetype values should match expected strings."""
        assert ArchetypeType.ARCHITECT.value == "architect"
        assert ArchetypeType.BUILDER.value == "builder"
        assert ArchetypeType.QC.value == "qc"
        assert ArchetypeType.RESEARCHER.value == "researcher"
