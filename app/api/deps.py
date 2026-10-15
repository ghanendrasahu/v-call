"""FastAPI dependencies: DB session, tenant context, auth."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import AuthTokenError, TokenPayload, decode_token
from app.db.enums import UserRole
from app.db.models import User
from app.db.session import get_session_factory

settings = get_settings()

# Token URL points at the (future) OAuth2 password flow endpoint.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/token", auto_error=False)


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Authenticated request identity, always tenant-scoped."""

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    role: UserRole
    email: str | None = None


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession (committed on success, rolled back on error)."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_current_context(
    token: str | None = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_db),
) -> TenantContext:
    """Resolve the bearer token into a validated tenant context.

    Raises 401 for missing/invalid tokens or inactive users.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise unauthorized

    try:
        payload: TokenPayload = decode_token(token)
    except AuthTokenError:
        raise unauthorized from None

    user = await session.get(User, uuid.UUID(payload.sub))
    if user is None or not user.is_active:
        raise unauthorized

    return TenantContext(
        user_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role,
        email=user.email,
    )


async def require_admin(
    context: TenantContext = Depends(get_current_context),
) -> TenantContext:
    """Dependency: only tenant admins pass."""
    if context.role is not UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return context
