"""FastAPI authentication dependencies."""
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError

from .jwt import verify_token, TokenData


# HTTP Bearer token scheme for JWT authentication
bearer_scheme = HTTPBearer(auto_error=True)
bearer_scheme_optional = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
) -> TokenData:
    """
    Dependency to get the current authenticated user from JWT token.

    Args:
        credentials: Bearer token credentials from Authorization header

    Returns:
        TokenData with user information

    Raises:
        HTTPException: 401 if token is invalid or missing
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token_data = verify_token(credentials.credentials)
        if token_data.sub is None:
            raise credentials_exception
        return token_data
    except JWTError:
        raise credentials_exception


async def get_current_user_optional(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme_optional)
    ]
) -> TokenData | None:
    """
    Optional authentication dependency - returns None if no token provided.

    Useful for endpoints that work differently for authenticated vs anonymous users.

    Args:
        credentials: Optional Bearer token from Authorization header

    Returns:
        TokenData if valid token provided, None otherwise
    """
    if credentials is None:
        return None

    try:
        return verify_token(credentials.credentials)
    except JWTError:
        return None


def require_scopes(*required_scopes: str):
    """
    Dependency factory to require specific scopes.

    Usage:
        @app.get("/admin", dependencies=[Depends(require_scopes("admin", "write"))])
        async def admin_endpoint():
            ...

    Args:
        required_scopes: Scopes that the token must have

    Returns:
        Dependency function that validates scopes
    """
    async def scope_checker(
        token_data: Annotated[TokenData, Depends(get_current_user)]
    ) -> TokenData:
        token_scopes = set(token_data.scopes)
        required = set(required_scopes)

        if not required.issubset(token_scopes):
            missing = required - token_scopes
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {', '.join(missing)}",
            )

        return token_data

    return scope_checker
