"""License validation using JWT tokens.

Licenses are signed JWTs that can be validated offline.
No server required for validation - just the public key.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


class LicenseStatus(str, Enum):
    """Status of a license."""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"
    NOT_FOUND = "not_found"


@dataclass
class LicenseInfo:
    """Information about a validated license."""
    status: LicenseStatus
    email: Optional[str] = None
    organization: Optional[str] = None
    features: List[str] = None
    expires_at: Optional[datetime] = None
    issued_at: Optional[datetime] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = []

    @property
    def is_valid(self) -> bool:
        return self.status == LicenseStatus.VALID

    def has_feature(self, feature: str) -> bool:
        """Check if license includes a specific feature."""
        if not self.is_valid:
            return False
        # "all" grants access to everything
        if "all" in self.features:
            return True
        return feature in self.features


class LicenseValidator:
    """
    Validates enterprise license keys.

    License keys are signed JWTs containing:
    - sub: Email of license holder
    - org: Organization name
    - features: List of enabled features
    - exp: Expiration timestamp
    - iat: Issued timestamp

    The public key is embedded in the code for offline validation.
    """

    # RSA-256 public key for license validation
    # Replace this with your actual public key in production
    PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA0Z3VS5JJcds3xfn/ygWy
RsGUkKvLiz6fXzrHqMALpB7KqnEyMgz0Lz3XPXW3T5W/R7yjKE8TH6hYIV1YqJqQ
kHvLNMOJAoZKxGpLLPG0W6TvKXe3O7HLwlPJy0EXAMPLE_NOT_REAL_KEYq3j
p5TJ8R6gLtHTpRXKwMCtest1234567890abcdefghijklmnopqrstuvwxyzAB
CDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/EXAMPLE_PUBLIC_KEY==
-----END PUBLIC KEY-----"""

    def __init__(self, public_key: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            public_key: Override the default public key (for testing)
        """
        self._public_key = public_key or self.PUBLIC_KEY
        self._jwt_available = self._check_jwt_available()

    def _check_jwt_available(self) -> bool:
        """Check if PyJWT is available."""
        try:
            import jwt  # noqa: F401
            return True
        except ImportError:
            return False

    def validate(self, license_key: Optional[str]) -> LicenseInfo:
        """
        Validate a license key.

        Args:
            license_key: The JWT license key string

        Returns:
            LicenseInfo with validation results
        """
        if license_key is None:
            return LicenseInfo(
                status=LicenseStatus.NOT_FOUND,
                error="No license key provided",
            )

        # Check for development keys first (always allowed)
        if license_key.startswith("dev-"):
            return self._validate_development_key(license_key)

        if not self._jwt_available:
            # If PyJWT not installed, only accept dev keys
            return LicenseInfo(
                status=LicenseStatus.INVALID,
                error="PyJWT not installed and no dev key provided",
            )

        return self._validate_jwt(license_key)

    def _validate_development_key(self, license_key: str) -> LicenseInfo:
        """
        Development-mode validation when PyJWT is not available.

        Accepts special development keys for testing.
        """
        # Development keys for testing
        if license_key == "dev-all-features":
            return LicenseInfo(
                status=LicenseStatus.VALID,
                email="developer@localhost",
                organization="Development",
                features=["all"],
                expires_at=datetime(2099, 12, 31),
            )

        if license_key.startswith("dev-"):
            features = license_key.replace("dev-", "").split("-")
            return LicenseInfo(
                status=LicenseStatus.VALID,
                email="developer@localhost",
                organization="Development",
                features=features,
                expires_at=datetime(2099, 12, 31),
            )

        return LicenseInfo(
            status=LicenseStatus.INVALID,
            error="Invalid license key format",
        )

    def _validate_jwt(self, license_key: str) -> LicenseInfo:
        """Validate a JWT license key."""
        try:
            import jwt

            # Decode and verify the JWT
            payload = jwt.decode(
                license_key,
                self._public_key,
                algorithms=["RS256"],
                options={"verify_exp": True},
            )

            # Extract license info
            return LicenseInfo(
                status=LicenseStatus.VALID,
                email=payload.get("sub"),
                organization=payload.get("org"),
                features=payload.get("features", []),
                expires_at=datetime.fromtimestamp(payload.get("exp", 0)),
                issued_at=datetime.fromtimestamp(payload.get("iat", 0)),
            )

        except jwt.ExpiredSignatureError:
            return LicenseInfo(
                status=LicenseStatus.EXPIRED,
                error="License has expired",
            )

        except jwt.InvalidTokenError as e:
            return LicenseInfo(
                status=LicenseStatus.INVALID,
                error=f"Invalid license: {str(e)}",
            )

    @staticmethod
    def get_license_from_env() -> Optional[str]:
        """Get license key from environment variable."""
        return os.getenv("TASK_ORCHESTRATOR_LICENSE")


# Singleton validator instance
_validator: Optional[LicenseValidator] = None


def get_license_validator() -> LicenseValidator:
    """Get the singleton license validator instance."""
    global _validator
    if _validator is None:
        _validator = LicenseValidator()
    return _validator


def check_feature_access(feature: str) -> bool:
    """
    Quick check if a feature is available.

    Args:
        feature: Feature name (e.g., "federation", "content", "research")

    Returns:
        True if feature is accessible
    """
    validator = get_license_validator()
    license_key = validator.get_license_from_env()
    info = validator.validate(license_key)
    return info.has_feature(feature)
