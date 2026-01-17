"""JWT token creation and validation utilities.

Uses python-jose for JWT operations and passlib for credential hashing.
Supports both HS256 (symmetric) and RS256 (asymmetric) algorithms.
"""
import base64
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.backends import default_backend
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import structlog

if TYPE_CHECKING:
    from src.governance.agent_identity import AgentCredential

logger = structlog.get_logger(__name__)


# Configuration - in production, load from environment/secrets manager
# Default values for development only
_jwt_secret = os.getenv("JWT_SECRET_KEY")
if not _jwt_secret:
    raise ValueError("JWT_SECRET_KEY environment variable must be set")
JWT_SECRET_KEY: str = _jwt_secret
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)
JWT_RS256_KEY_PATH = os.getenv(
    "JWT_RS256_KEY_PATH",
    str(Path.home() / ".claude" / "governance" / "jwt_keys")
)

# Credential hashing context using bcrypt
# The "auto" setting marks old hash schemes for automatic upgrade
crypt_context = CryptContext(schemes=["bcrypt"])


class TokenData(BaseModel):
    """Data extracted from a validated JWT token."""

    sub: str  # Subject (usually user ID or username)
    exp: Optional[datetime] = None
    scopes: list[str] = []


class TokenPayload(BaseModel):
    """Payload for creating a JWT token."""

    sub: str
    scopes: list[str] = []
    exp: Optional[datetime] = None


def create_access_token(
    subject: str,
    scopes: Optional[list[str]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: The subject of the token (usually user ID or username)
        scopes: Optional list of permission scopes
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.now(timezone.utc) + expires_delta

    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "scopes": scopes or [],
    }

    encoded_jwt = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token.

    Args:
        token: The JWT token string to verify

    Returns:
        TokenData with the decoded token information

    Raises:
        JWTError: If the token is invalid or expired
    """
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

    sub = payload.get("sub")
    if sub is None:
        raise JWTError("Token missing subject claim")
    sub_str: str = str(sub)

    exp = payload.get("exp")
    exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None

    scopes = payload.get("scopes", [])

    return TokenData(sub=sub_str, exp=exp_datetime, scopes=scopes)


def hash_credential(plain_text: str) -> str:
    """
    Hash a credential (e.g., API key or secret) using bcrypt.

    Args:
        plain_text: The plain text credential to hash

    Returns:
        Hashed credential string
    """
    return crypt_context.hash(plain_text)


def verify_credential(plain_text: str, hashed: str) -> bool:
    """
    Verify a plain text credential against its hash.

    Args:
        plain_text: The plain text credential to verify
        hashed: The hashed credential to compare against

    Returns:
        True if the credential matches, False otherwise
    """
    return crypt_context.verify(plain_text, hashed)


# =============================================================================
# RS256 JWT Support for Agent Tokens
# =============================================================================


class AgentTokenData(BaseModel):
    """Data extracted from a validated agent JWT token."""

    agent_id: str
    session_id: str
    env_fingerprint: str
    archetype: str  # Will be ArchetypeType value
    capabilities: List[str]
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None


class JWTKeyManager:
    """
    Manages RSA keypairs for RS256 JWT signing.

    Features:
    - Key generation and storage
    - Key rotation with grace period
    - JWKS endpoint support
    """

    KEY_SIZE = 2048

    def __init__(self, key_storage_path: Optional[str] = None):
        """
        Initialize the key manager.

        Args:
            key_storage_path: Directory for key storage
        """
        self.storage_path = Path(key_storage_path or JWT_RS256_KEY_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._current_key_id: Optional[str] = None
        self._keys: dict[str, dict] = {}  # kid -> {private, public, created, expires}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load existing keys from storage."""
        keys_file = self.storage_path / "keys.json"
        if keys_file.exists():
            try:
                with open(keys_file, "r") as f:
                    data = json.load(f)
                    self._keys = data.get("keys", {})
                    self._current_key_id = data.get("current_key_id")
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_keys(self) -> None:
        """Save keys to storage."""
        keys_file = self.storage_path / "keys.json"
        with open(keys_file, "w") as f:
            json.dump(
                {"keys": self._keys, "current_key_id": self._current_key_id},
                f,
                indent=2,
            )

    def generate_keypair(self) -> Tuple[str, str]:
        """
        Generate a new RSA keypair and set it as current.

        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        # Generate RSA keypair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.KEY_SIZE,
            backend=default_backend(),
        )
        public_key = private_key.public_key()

        # Convert to PEM
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        # Generate key ID
        kid = str(uuid.uuid4())[:8]

        # Store key
        now = datetime.now(timezone.utc)
        self._keys[kid] = {
            "private": private_pem,
            "public": public_pem,
            "created": now.isoformat(),
            "expires": None,  # Current key doesn't expire
        }

        # Set as current
        self._current_key_id = kid
        self._save_keys()

        logger.info("jwt_keypair_generated", kid=kid)

        return private_pem, public_pem

    def rotate_keys(self, keep_old_for_days: int = 7) -> None:
        """
        Rotate to a new keypair, keeping old for verification.

        Args:
            keep_old_for_days: Days to keep old public keys
        """
        if self._current_key_id:
            # Mark old key as expiring
            now = datetime.now(timezone.utc)
            if keep_old_for_days > 0:
                expires = now + timedelta(days=keep_old_for_days)
                self._keys[self._current_key_id]["expires"] = expires.isoformat()
            else:
                # Remove immediately
                del self._keys[self._current_key_id]

        # Generate new keypair
        self.generate_keypair()

        # Clean up expired keys
        self._cleanup_expired_keys()

    def _cleanup_expired_keys(self) -> None:
        """Remove keys that have expired."""
        now = datetime.now(timezone.utc)
        expired = [
            kid for kid, key_data in self._keys.items()
            if key_data.get("expires")
            and datetime.fromisoformat(key_data["expires"]) < now
        ]

        for kid in expired:
            del self._keys[kid]
            logger.info("jwt_key_expired_removed", kid=kid)

        if expired:
            self._save_keys()

    def get_current_private_key(self) -> Optional[str]:
        """Get the current private key for signing."""
        if not self._current_key_id:
            return None
        key_data = self._keys.get(self._current_key_id)
        return key_data["private"] if key_data else None

    def get_current_key_id(self) -> Optional[str]:
        """Get the current key ID."""
        return self._current_key_id

    def get_public_key(self, kid: str) -> Optional[str]:
        """Get a public key by key ID."""
        key_data = self._keys.get(kid)
        return key_data["public"] if key_data else None

    def get_jwks(self) -> dict:
        """
        Get JWKS (JSON Web Key Set) for public key distribution.

        Returns:
            JWKS structure with all valid public keys
        """
        keys = []

        for kid, key_data in self._keys.items():
            public_pem = key_data["public"]

            # Parse public key
            public_key = serialization.load_pem_public_key(
                public_pem.encode("utf-8"),
                backend=default_backend(),
            )

            # Only process RSA keys for JWKS (skip Ed25519, etc.)
            if not isinstance(public_key, RSAPublicKey):
                logger.warning("jwks_skip_non_rsa_key", kid=kid)
                continue

            # Get RSA numbers
            numbers = public_key.public_numbers()

            # Encode modulus and exponent as base64url
            n_bytes = numbers.n.to_bytes(
                (numbers.n.bit_length() + 7) // 8, byteorder="big"
            )
            e_bytes = numbers.e.to_bytes(
                (numbers.e.bit_length() + 7) // 8, byteorder="big"
            )

            jwk = {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": kid,
                "n": base64.urlsafe_b64encode(n_bytes).decode("utf-8").rstrip("="),
                "e": base64.urlsafe_b64encode(e_bytes).decode("utf-8").rstrip("="),
            }
            keys.append(jwk)

        return {"keys": keys}


# Global key manager instance (lazy initialization)
_key_manager: Optional[JWTKeyManager] = None


def get_key_manager() -> JWTKeyManager:
    """Get or create the global key manager."""
    global _key_manager
    if _key_manager is None:
        _key_manager = JWTKeyManager()
    return _key_manager


def create_agent_token(
    agent_credential: "AgentCredential",
    session_id: str,
    env_fingerprint: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create an RS256-signed JWT for an agent.

    Args:
        agent_credential: The agent's credential
        session_id: Associated session ID
        env_fingerprint: Environment fingerprint
        expires_delta: Optional custom expiration

    Returns:
        RS256-signed JWT token
    """
    key_manager = get_key_manager()
    private_key = key_manager.get_current_private_key()
    kid = key_manager.get_current_key_id()

    if not private_key or not kid:
        raise ValueError("No signing key available. Generate keypair first.")

    if expires_delta is None:
        expires_delta = timedelta(hours=1)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    payload = {
        "agent_id": agent_credential.agent_id,
        "session_id": session_id,
        "env_fingerprint": env_fingerprint,
        "archetype": agent_credential.archetype.value,
        "capabilities": agent_credential.capabilities,
        "exp": expire,
        "iat": now,
    }

    headers = {"kid": kid}

    token = jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
        headers=headers,
    )

    return token


def verify_agent_token(token: str) -> AgentTokenData:
    """
    Verify an RS256-signed agent JWT.

    Args:
        token: The JWT token to verify

    Returns:
        AgentTokenData with decoded claims

    Raises:
        JWTError: If token is invalid or expired
    """
    key_manager = get_key_manager()

    # Get key ID from header
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
    except JWTError:
        raise JWTError("Invalid token header")

    if not kid:
        raise JWTError("Token missing key ID (kid)")

    # Get public key
    public_key = key_manager.get_public_key(kid)
    if not public_key:
        raise JWTError(f"Unknown key ID: {kid}")

    # Verify and decode
    payload = jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
    )

    # Parse timestamps
    exp = payload.get("exp")
    iat = payload.get("iat")

    exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None
    iat_dt = datetime.fromtimestamp(iat, tz=timezone.utc) if iat else None

    return AgentTokenData(
        agent_id=payload["agent_id"],
        session_id=payload["session_id"],
        env_fingerprint=payload["env_fingerprint"],
        archetype=payload["archetype"],
        capabilities=payload.get("capabilities", []),
        exp=exp_dt,
        iat=iat_dt,
    )
