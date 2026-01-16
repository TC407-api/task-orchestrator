"""Agent Identity Manager for Non-Human Identity (NHI) lifecycle management.

Provides cryptographic identity for agents with:
- RSA keypair generation for signature verification
- 90-day credential rotation
- Revocation support
- Secure private key storage
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import structlog

logger = structlog.get_logger(__name__)


class ArchetypeType(str, Enum):
    """Agent archetype classifications."""
    ARCHITECT = "architect"
    BUILDER = "builder"
    QC = "qc"
    RESEARCHER = "researcher"


@dataclass
class AgentCredential:
    """
    Cryptographic credential for an agent identity.

    Attributes:
        agent_id: Unique identifier (UUID)
        public_key: PEM-encoded public key for verification
        archetype: Agent's role classification
        capabilities: List of allowed operations
        created_at: Credential creation timestamp
        expires_at: Credential expiration timestamp
        revoked: Whether credential has been revoked
    """
    agent_id: str
    public_key: str
    archetype: ArchetypeType
    capabilities: List[str]
    created_at: datetime
    expires_at: datetime
    revoked: bool = False

    def is_expired(self) -> bool:
        """Check if credential has expired."""
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "agent_id": self.agent_id,
            "public_key": self.public_key,
            "archetype": self.archetype.value,
            "capabilities": self.capabilities,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "revoked": self.revoked,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCredential":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            public_key=data["public_key"],
            archetype=ArchetypeType(data["archetype"]),
            capabilities=data["capabilities"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            revoked=data.get("revoked", False),
        )


class AgentIdentityManager:
    """
    Manages the lifecycle of agent identities with cryptographic binding.

    Features:
    - RSA keypair generation (2048-bit)
    - Credential rotation with configurable expiry
    - Revocation support
    - Signature verification
    """

    DEFAULT_EXPIRY_DAYS = 90
    KEY_SIZE = 2048

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the identity manager.

        Args:
            storage_path: Directory for credential storage.
                         Defaults to ~/.claude/governance/agents/
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".claude" / "governance" / "agents"

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._credentials_path = self.storage_path / "credentials"
        self._keys_path = self.storage_path / "keys"
        self._credentials_path.mkdir(exist_ok=True)
        self._keys_path.mkdir(exist_ok=True)

        # In-memory cache
        self._credentials: dict[str, AgentCredential] = {}
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Load existing credentials from storage."""
        for cred_file in self._credentials_path.glob("*.json"):
            try:
                with open(cred_file, "r") as f:
                    data = json.load(f)
                    credential = AgentCredential.from_dict(data)
                    self._credentials[credential.agent_id] = credential
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("failed_to_load_credential", file=str(cred_file), error=str(e))

    def _save_credential(self, credential: AgentCredential) -> None:
        """Save credential to storage."""
        cred_file = self._credentials_path / f"{credential.agent_id}.json"
        with open(cred_file, "w") as f:
            json.dump(credential.to_dict(), f, indent=2)
        self._credentials[credential.agent_id] = credential

    def _generate_keypair(self) -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Generate RSA keypair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.KEY_SIZE,
            backend=default_backend(),
        )
        return private_key, private_key.public_key()

    def _save_private_key(self, agent_id: str, private_key: rsa.RSAPrivateKey) -> None:
        """Save private key securely."""
        key_file = self._keys_path / f"{agent_id}.pem"
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        key_file.write_bytes(pem)
        # Restrict file permissions (Unix-like systems)
        try:
            key_file.chmod(0o600)
        except (OSError, AttributeError):
            pass  # Windows doesn't support Unix permissions

    def _load_private_key(self, agent_id: str) -> Optional[rsa.RSAPrivateKey]:
        """Load private key from storage."""
        key_file = self._keys_path / f"{agent_id}.pem"
        if not key_file.exists():
            return None
        pem = key_file.read_bytes()
        return serialization.load_pem_private_key(pem, password=None, backend=default_backend())

    def _public_key_to_pem(self, public_key: rsa.RSAPublicKey) -> str:
        """Convert public key to PEM string."""
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

    def create_agent(
        self,
        archetype: ArchetypeType,
        capabilities: List[str],
        expires_in_days: Optional[int] = None,
    ) -> AgentCredential:
        """
        Create a new agent with cryptographic identity.

        Args:
            archetype: Agent's role classification
            capabilities: List of allowed operations
            expires_in_days: Days until credential expires (default: 90)

        Returns:
            AgentCredential with generated keypair
        """
        agent_id = str(uuid.uuid4())
        private_key, public_key = self._generate_keypair()

        now = datetime.now()
        expiry_days = expires_in_days or self.DEFAULT_EXPIRY_DAYS
        expires_at = now + timedelta(days=expiry_days)

        credential = AgentCredential(
            agent_id=agent_id,
            public_key=self._public_key_to_pem(public_key),
            archetype=archetype,
            capabilities=capabilities,
            created_at=now,
            expires_at=expires_at,
        )

        self._save_private_key(agent_id, private_key)
        self._save_credential(credential)

        logger.info(
            "agent_created",
            agent_id=agent_id,
            archetype=archetype.value,
            expires_at=expires_at.isoformat(),
        )

        return credential

    def get_agent(self, agent_id: str) -> Optional[AgentCredential]:
        """
        Get an agent credential by ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            AgentCredential if found, None otherwise
        """
        return self._credentials.get(agent_id)

    def rotate_credential(self, agent_id: str) -> AgentCredential:
        """
        Rotate an agent's keypair, invalidating the old one.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            New AgentCredential with fresh keypair

        Raises:
            ValueError: If agent not found
        """
        old_credential = self._credentials.get(agent_id)
        if not old_credential:
            raise ValueError(f"Agent not found: {agent_id}")

        # Generate new keypair
        private_key, public_key = self._generate_keypair()

        # Create new credential preserving identity
        now = datetime.now()
        new_credential = AgentCredential(
            agent_id=agent_id,
            public_key=self._public_key_to_pem(public_key),
            archetype=old_credential.archetype,
            capabilities=old_credential.capabilities,
            created_at=now,
            expires_at=now + timedelta(days=self.DEFAULT_EXPIRY_DAYS),
        )

        # Save new keys (overwrites old)
        self._save_private_key(agent_id, private_key)
        self._save_credential(new_credential)

        logger.info("agent_credential_rotated", agent_id=agent_id)

        return new_credential

    def revoke_credential(self, agent_id: str) -> None:
        """
        Revoke an agent's credential.

        Args:
            agent_id: The agent's unique identifier

        Raises:
            ValueError: If agent not found
        """
        credential = self._credentials.get(agent_id)
        if not credential:
            raise ValueError(f"Agent not found: {agent_id}")

        # Mark as revoked
        revoked_credential = AgentCredential(
            agent_id=credential.agent_id,
            public_key=credential.public_key,
            archetype=credential.archetype,
            capabilities=credential.capabilities,
            created_at=credential.created_at,
            expires_at=credential.expires_at,
            revoked=True,
        )

        self._save_credential(revoked_credential)

        logger.info("agent_credential_revoked", agent_id=agent_id)

    def sign_data(self, agent_id: str, data: bytes) -> bytes:
        """
        Sign data with an agent's private key.

        Args:
            agent_id: The agent's unique identifier
            data: Data to sign

        Returns:
            Signature bytes

        Raises:
            ValueError: If agent not found or private key unavailable
        """
        private_key = self._load_private_key(agent_id)
        if not private_key:
            raise ValueError(f"Private key not found for agent: {agent_id}")

        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return signature

    def verify_agent(
        self,
        agent_id: str,
        signature: bytes,
        data: bytes,
        public_key: Optional[str] = None,
    ) -> bool:
        """
        Verify an agent's signature.

        Args:
            agent_id: The agent's unique identifier
            signature: Signature to verify
            data: Original data that was signed
            public_key: Optional specific public key to use

        Returns:
            True if signature is valid, False otherwise
        """
        credential = self._credentials.get(agent_id)
        if not credential:
            return False

        # Check if revoked or expired
        if credential.revoked or credential.is_expired():
            return False

        # Use provided public key or credential's key
        pem_key = public_key or credential.public_key

        try:
            pub_key = serialization.load_pem_public_key(
                pem_key.encode("utf-8"),
                backend=default_backend(),
            )
            pub_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    def list_expiring(self, days: int = 7) -> List[AgentCredential]:
        """
        List agents with credentials expiring within N days.

        Args:
            days: Number of days to look ahead

        Returns:
            List of AgentCredential objects expiring soon
        """
        cutoff = datetime.now() + timedelta(days=days)
        now = datetime.now()

        expiring = [
            cred for cred in self._credentials.values()
            if not cred.revoked and now < cred.expires_at <= cutoff
        ]

        return sorted(expiring, key=lambda c: c.expires_at)

    def cleanup_expired(self) -> int:
        """
        Remove expired and revoked credentials from storage.

        Returns:
            Number of credentials removed
        """
        removed = 0
        for agent_id, credential in list(self._credentials.items()):
            if credential.is_expired() or credential.revoked:
                # Remove files
                cred_file = self._credentials_path / f"{agent_id}.json"
                key_file = self._keys_path / f"{agent_id}.pem"

                if cred_file.exists():
                    cred_file.unlink()
                if key_file.exists():
                    key_file.unlink()

                del self._credentials[agent_id]
                removed += 1

        if removed:
            logger.info("credentials_cleaned_up", count=removed)

        return removed
