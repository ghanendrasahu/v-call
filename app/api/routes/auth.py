"""OAuth2 authentication endpoints.

- ``POST /auth/register``: bootstrap a tenant + its first admin user.
- ``POST /auth/token``: OAuth2 password flow → JWT bearer.
- ``GET /auth/me``: introspect the current identity.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TenantContext, get_current_context, get_db
from app.core.rate_limit import limiter
from app.core.security import create_access_token, hash_password
from app.db.enums import UserRole
from app.db.models import User
from app.db.repositories import TenantRepository, UserRepository
from app.schemas.api import TokenOut, UserOut, UserRegisterIn

router = APIRouter(prefix="/auth", tags=["auth"])

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower()).strip("-") or "tenant"


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("5/minute")
async def register(
    request: Request,
    payload: UserRegisterIn,
    session: AsyncSession = Depends(get_db),
) -> UserOut:
    """Create a tenant and its first admin (bootstrap only)."""
    tenants = TenantRepository(session)
    users = UserRepository(session)

    slug = _slugify(payload.tenant_name)
    if await tenants.get_by_slug(slug) is not None:
        raise HTTPException(status_code=409, detail="Tenant already exists")
    if await users.get_by_email(payload.email) is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    tenant = await tenants.create(name=payload.tenant_name, slug=slug)
    user = await users.create(
        tenant_id=tenant.id,
        email=payload.email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=UserRole.ADMIN,
    )
    return UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        tenant_id=tenant.id,
    )


@router.post("/token", response_model=TokenOut)
@limiter.limit("10/minute")
async def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_db),
) -> TokenOut:
    """OAuth2 password grant → JWT bearer token."""
    users = UserRepository(session)
    user = await users.authenticate(email=form.username, password=form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        subject=str(user.id),
        tenant_id=str(user.tenant_id),
        role=user.role.value,
    )
    return TokenOut(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserOut)
async def me(
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> UserOut:
    """Return the authenticated user's profile."""
    user = await session.get(User, context.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        tenant_id=user.tenant_id,
    )
