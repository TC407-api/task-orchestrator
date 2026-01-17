"""Tests for RS256 JWT enhancements.

TDD RED Phase: These tests define the expected behavior for RS256 JWT support.
They should fail initially until the implementation is complete.
"""
import pytest
import json
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jose import jwt, JWTError

from src.api.auth.jwt import (
    JWTKeyManager,
    create_agent_token,
    verify_agent_token,
    AgentTokenData,
)
from src.governance.agent_identity import AgentCredential, ArchetypeType


class TestJWTKeyManager:
    """Tests for JWTKeyManager key management."""

    @pytest.fixture
    def key_manager(self) -> JWTKeyManager:
        """Create a JWTKeyManager with temp storage."""
        with TemporaryDirectory() as tmpdir:
            yield JWTKeyManager(key_storage_path=tmpdir)

    def test_generate_keypair_creates_valid_keys(self, key_manager: JWTKeyManager):
        """generate_keypair() should create valid RSA keys."""
        private_key, public_key = key_manager.generate_keypair()

        assert private_key.startswith("-----BEGIN RSA PRIVATE KEY-----") or \
               private_key.startswith("-----BEGIN PRIVATE KEY-----")
        assert public_key.startswith("-----BEGIN PUBLIC KEY-----")

    def test_get_current_private_key(self, key_manager: JWTKeyManager):
        """get_current_private_key() should return the current signing key."""
        # Generate a keypair first
        key_manager.generate_keypair()

        private_key = key_manager.get_current_private_key()

        assert private_key is not None
        assert "PRIVATE KEY" in private_key

    def test_get_jwks_returns_valid_json(self, key_manager: JWTKeyManager):
        """get_jwks() should return a valid JWKS structure."""
        key_manager.generate_keypair()

        jwks = key_manager.get_jwks()

        assert "keys" in jwks
        assert isinstance(jwks["keys"], list)
        assert len(jwks["keys"]) >= 1

        # Each key should have required JWKS fields
        for key in jwks["keys"]:
            assert "kty" in key  # Key type (RSA)
            assert "use" in key  # Key usage (sig)
            assert "kid" in key  # Key ID
            assert "n" in key    # RSA modulus
            assert "e" in key    # RSA exponent


class TestKeyRotation:
    """Tests for key rotation functionality."""

    @pytest.fixture
    def key_manager(self) -> JWTKeyManager:
        """Create a JWTKeyManager with temp storage."""
        with TemporaryDirectory() as tmpdir:
            yield JWTKeyManager(key_storage_path=tmpdir)

    def test_key_rotation_keeps_old_keys(self, key_manager: JWTKeyManager):
        """rotate_keys() should keep old public keys for verification."""
        # Generate initial keypair
        key_manager.generate_keypair()
        initial_jwks = key_manager.get_jwks()
        initial_kid = initial_jwks["keys"][0]["kid"]

        # Rotate keys
        key_manager.rotate_keys(keep_old_for_days=7)

        # JWKS should now have both old and new keys
        rotated_jwks = key_manager.get_jwks()
        kids = [k["kid"] for k in rotated_jwks["keys"]]

        assert len(rotated_jwks["keys"]) >= 2
        assert initial_kid in kids  # Old key still present

    def test_key_rotation_creates_new_primary_key(self, key_manager: JWTKeyManager):
        """rotate_keys() should create a new primary signing key."""
        # Generate initial keypair
        key_manager.generate_keypair()
        initial_private = key_manager.get_current_private_key()

        # Rotate keys
        key_manager.rotate_keys(keep_old_for_days=7)

        new_private = key_manager.get_current_private_key()

        # Private key should be different
        assert new_private != initial_private

    def test_old_keys_eventually_removed(self, key_manager: JWTKeyManager):
        """Old keys should be removed after keep_old_for_days expires."""
        key_manager.generate_keypair()

        # Rotate with 0-day retention (immediate removal)
        key_manager.rotate_keys(keep_old_for_days=0)

        jwks = key_manager.get_jwks()

        # Should only have the new key
        assert len(jwks["keys"]) == 1


class TestCreateAgentToken:
    """Tests for creating agent tokens."""

    @pytest.fixture
    def mock_credential(self) -> AgentCredential:
        """Create a mock agent credential."""
        return AgentCredential(
            agent_id="agent-123-456",
            public_key="-----BEGIN PUBLIC KEY-----\nMIIBIjAN...",
            archetype=ArchetypeType.BUILDER,
            capabilities=["code_write", "code_read", "test_run"],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90),
        )

    @pytest.fixture
    def key_manager(self) -> JWTKeyManager:
        """Create a JWTKeyManager with temp storage."""
        with TemporaryDirectory() as tmpdir:
            mgr = JWTKeyManager(key_storage_path=tmpdir)
            mgr.generate_keypair()
            yield mgr

    def test_create_agent_token_uses_rs256(
        self, mock_credential: AgentCredential, key_manager: JWTKeyManager
    ):
        """create_agent_token() should use RS256 algorithm."""
        with patch("src.api.auth.jwt.get_key_manager", return_value=key_manager):
            token = create_agent_token(
                agent_credential=mock_credential,
                session_id="session-abc",
                env_fingerprint="fingerprint-xyz",
            )

        # Decode header to verify algorithm
        header = jwt.get_unverified_header(token)
        assert header["alg"] == "RS256"

    def test_create_agent_token_includes_claims(
        self, mock_credential: AgentCredential, key_manager: JWTKeyManager
    ):
        """create_agent_token() should include agent-specific claims."""
        with patch("src.api.auth.jwt.get_key_manager", return_value=key_manager):
            token = create_agent_token(
                agent_credential=mock_credential,
                session_id="session-abc",
                env_fingerprint="fingerprint-xyz",
            )

            # Verify token to get claims
            token_data = verify_agent_token(token)

        assert token_data.agent_id == "agent-123-456"
        assert token_data.session_id == "session-abc"
        assert token_data.env_fingerprint == "fingerprint-xyz"
        assert token_data.archetype == ArchetypeType.BUILDER
        assert token_data.capabilities == ["code_write", "code_read", "test_run"]


class TestVerifyAgentToken:
    """Tests for verifying agent tokens."""

    @pytest.fixture
    def key_manager(self) -> JWTKeyManager:
        """Create a JWTKeyManager with temp storage."""
        with TemporaryDirectory() as tmpdir:
            mgr = JWTKeyManager(key_storage_path=tmpdir)
            mgr.generate_keypair()
            yield mgr

    @pytest.fixture
    def valid_token(self, key_manager: JWTKeyManager) -> str:
        """Create a valid agent token."""
        credential = AgentCredential(
            agent_id="agent-test-123",
            public_key="-----BEGIN PUBLIC KEY-----\nMIIBIjAN...",
            archetype=ArchetypeType.RESEARCHER,
            capabilities=["web_search"],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90),
        )

        with patch("src.api.auth.jwt.get_key_manager", return_value=key_manager):
            return create_agent_token(
                agent_credential=credential,
                session_id="session-123",
                env_fingerprint="fingerprint-abc",
            )

    def test_verify_token_with_public_key(
        self, valid_token: str, key_manager: JWTKeyManager
    ):
        """verify_agent_token() should verify with public key from JWKS."""
        with patch("src.api.auth.jwt.get_key_manager", return_value=key_manager):
            token_data = verify_agent_token(valid_token)

        assert token_data is not None
        assert token_data.agent_id == "agent-test-123"

    def test_expired_token_rejected(self, key_manager: JWTKeyManager):
        """verify_agent_token() should reject expired tokens."""
        credential = AgentCredential(
            agent_id="agent-expired",
            public_key="-----BEGIN PUBLIC KEY-----\nMIIBIjAN...",
            archetype=ArchetypeType.BUILDER,
            capabilities=[],
            created_at=datetime.now() - timedelta(days=2),
            expires_at=datetime.now() + timedelta(days=88),
        )

        with patch("src.api.auth.jwt.get_key_manager", return_value=key_manager):
            # Create token with negative expiry
            token = create_agent_token(
                agent_credential=credential,
                session_id="session-expired",
                env_fingerprint="fp",
                expires_delta=timedelta(seconds=-1),  # Already expired
            )

            with pytest.raises(JWTError):
                verify_agent_token(token)

    def test_tampered_token_rejected(
        self, valid_token: str, key_manager: JWTKeyManager
    ):
        """verify_agent_token() should reject tampered tokens."""
        # Tamper with the token payload
        parts = valid_token.split(".")
        # Modify the payload slightly
        import base64
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
        payload["agent_id"] = "hacked-agent"
        tampered_payload = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"

        with patch("src.api.auth.jwt.get_key_manager", return_value=key_manager):
            with pytest.raises(JWTError):
                verify_agent_token(tampered_token)

    def test_wrong_key_token_rejected(self, key_manager: JWTKeyManager):
        """Token signed with different key should be rejected."""
        # Create a token with a different key manager
        with TemporaryDirectory() as tmpdir:
            other_manager = JWTKeyManager(key_storage_path=tmpdir)
            other_manager.generate_keypair()

            credential = AgentCredential(
                agent_id="agent-other",
                public_key="-----BEGIN PUBLIC KEY-----\nMIIBIjAN...",
                archetype=ArchetypeType.QC,
                capabilities=[],
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=90),
            )

            with patch("src.api.auth.jwt.get_key_manager", return_value=other_manager):
                wrong_key_token = create_agent_token(
                    agent_credential=credential,
                    session_id="session",
                    env_fingerprint="fp",
                )

        # Try to verify with original key manager (different keys)
        with patch("src.api.auth.jwt.get_key_manager", return_value=key_manager):
            with pytest.raises(JWTError):
                verify_agent_token(wrong_key_token)


class TestAgentTokenData:
    """Tests for AgentTokenData model."""

    def test_agent_token_data_has_required_fields(self):
        """AgentTokenData should have all required fields."""
        token_data = AgentTokenData(
            agent_id="agent-123",
            session_id="session-456",
            env_fingerprint="fingerprint-789",
            archetype=ArchetypeType.ARCHITECT,
            capabilities=["design", "review"],
            exp=datetime.now() + timedelta(hours=1),
            iat=datetime.now(),
        )

        assert token_data.agent_id == "agent-123"
        assert token_data.session_id == "session-456"
        assert token_data.env_fingerprint == "fingerprint-789"
        assert token_data.archetype == ArchetypeType.ARCHITECT
        assert token_data.capabilities == ["design", "review"]
        assert token_data.exp is not None
        assert token_data.iat is not None


class TestJWKSEndpoint:
    """Tests for JWKS endpoint functionality."""

    @pytest.fixture
    def key_manager(self) -> JWTKeyManager:
        """Create a JWTKeyManager with temp storage."""
        with TemporaryDirectory() as tmpdir:
            mgr = JWTKeyManager(key_storage_path=tmpdir)
            mgr.generate_keypair()
            yield mgr

    def test_jwks_endpoint_returns_valid_json(self, key_manager: JWTKeyManager):
        """JWKS endpoint should return valid JSON structure."""
        jwks = key_manager.get_jwks()

        # Should be serializable to JSON
        jwks_json = json.dumps(jwks)
        assert jwks_json is not None

        # Re-parse to verify structure
        parsed = json.loads(jwks_json)
        assert "keys" in parsed

    def test_jwks_keys_have_correct_algorithm(self, key_manager: JWTKeyManager):
        """JWKS keys should specify RS256 algorithm."""
        jwks = key_manager.get_jwks()

        for key in jwks["keys"]:
            # If alg is specified, it should be RS256
            if "alg" in key:
                assert key["alg"] == "RS256"
            # Key type should be RSA
            assert key["kty"] == "RSA"
