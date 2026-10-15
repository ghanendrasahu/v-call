"""User repository: lookup + bootstrap for the OAuth2 flow."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.enums import UserRole
from app.db.models import User
from app.db.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Async data access for users."""

    model = User

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        email: str,
        password_hash: str | None,
        full_name: str | None = None,
        role: UserRole = UserRole.MEMBER,
    ) -> User:
        user = User(
            tenant_id=tenant_id,
            email=email.lower().strip(),
            hashed_password=password_hash,
            full_name=full_name,
            role=role,
        )
        return await self.add(user)

    async def get_by_email(self, email: str, *, tenant_id: uuid.UUID | None = None) -> User | None:
        """Case-insensitive email lookup, optionally tenant-scoped."""
        stmt = select(User).where(User.email == email.lower().strip())
        if tenant_id is not None:
            stmt = stmt.where(User.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def authenticate(
        self, *, email: str, password: str, tenant_id: uuid.UUID | None = None
    ) -> User | None:
        """Verify credentials; returns the active user or None."""
        from app.core.security import verify_password

        user = await self.get_by_email(email, tenant_id=tenant_id)
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user
