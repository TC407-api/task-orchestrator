"""Authentication module for Task Orchestrator API.

Supports both HS256 (symmetric) and RS256 (asymmetric) JWT algorithms.
"""
from .jwt import (
    # HS256 (symmetric) auth
    create_access_token,
    verify_token,
    hash_credential,
    verify_credential,
    TokenData,
    # RS256 (asymmetric) agent auth
    JWTKeyManager,
    AgentTokenData,
    create_agent_token,
    verify_agent_token,
    get_key_manager,
)
from .dependencies import (
    get_current_user,
    get_current_user_optional,
    require_scopes,
    bearer_scheme,
)

__all__ = [
    # HS256 (symmetric)
    "create_access_token",
    "verify_token",
    "hash_credential",
    "verify_credential",
    "TokenData",
    # RS256 (asymmetric) agent auth
    "JWTKeyManager",
    "AgentTokenData",
    "create_agent_token",
    "verify_agent_token",
    "get_key_manager",
    # Dependencies
    "get_current_user",
    "get_current_user_optional",
    "require_scopes",
    "bearer_scheme",
]
