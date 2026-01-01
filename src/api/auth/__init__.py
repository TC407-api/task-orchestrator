"""Authentication module for Task Orchestrator API."""
from .jwt import (
    create_access_token,
    verify_token,
    hash_credential,
    verify_credential,
    TokenData,
)
from .dependencies import (
    get_current_user,
    get_current_user_optional,
    require_scopes,
    bearer_scheme,
)

__all__ = [
    "create_access_token",
    "verify_token",
    "hash_credential",
    "verify_credential",
    "TokenData",
    "get_current_user",
    "get_current_user_optional",
    "require_scopes",
    "bearer_scheme",
]
