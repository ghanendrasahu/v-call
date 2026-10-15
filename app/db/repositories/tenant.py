"""Tenant repository — tenancy root (no tenant_id of its own)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Tenant


class TenantRepository:
    """Persistence for the tenancy root."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        name: str,
        slug: str,
        plan: str = "starter",
        settings: dict | None = None,
    ) -> Tenant:
        tenant = Tenant(name=name, slug=slug, plan=plan, settings=settings)
        self.session.add(tenant)
        await self.session.flush()
        await self.session.refresh(tenant)
        return tenant

    async def get_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        result = await self.session.execute(select(Tenant).where(Tenant.id == tenant_id))
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Tenant | None:
        result = await self.session.execute(select(Tenant).where(Tenant.slug == slug))
        return result.scalar_one_or_none()

    async def list_active(self) -> list[Tenant]:
        result = await self.session.execute(select(Tenant).where(Tenant.is_active.is_(True)))
        return list(result.scalars().all())

    async def deactivate(self, tenant_id: uuid.UUID) -> bool:
        tenant = await self.get_by_id(tenant_id)
        if tenant is None:
            return False
        tenant.is_active = False
        await self.session.flush()
        return True
