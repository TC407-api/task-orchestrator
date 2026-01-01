"""JWT token creation and validation utilities.

Uses python-jose for JWT operations and passlib for credential hashing.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel


# Configuration - in production, load from environment/secrets manager
# Default values for development only
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
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
    scopes: list[str] = None,
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

    sub: str = payload.get("sub")
    if sub is None:
        raise JWTError("Token missing subject claim")

    exp = payload.get("exp")
    exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None

    scopes = payload.get("scopes", [])

    return TokenData(sub=sub, exp=exp_datetime, scopes=scopes)


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
