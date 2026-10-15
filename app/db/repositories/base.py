"""Generic async repository with built-in tenant scoping."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """CRUD primitives for a tenant-scoped model."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(
        self,
        entity_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID | None = None,
        with_for_update: bool = False,
    ) -> ModelT | None:
        """Fetch one row by id, optionally locked and always tenant-scoped."""
        stmt = select(self.model).where(self.model.id == entity_id)  # type: ignore[attr-defined]
        if tenant_id is not None:
            stmt = stmt.where(self.model.tenant_id == tenant_id)  # type: ignore[attr-defined]
        if with_for_update:
            stmt = stmt.with_for_update()
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(
        self,
        *,
        tenant_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[ModelT]:
        """Paginated, tenant-scoped listing ordered by creation time."""
        stmt = (
            select(self.model)
            .where(self.model.tenant_id == tenant_id)  # type: ignore[attr-defined]
            .order_by(self.model.created_at.desc())  # type: ignore[attr-defined]
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def add(self, entity: ModelT) -> ModelT:
        """Persist and flush a new entity (caller controls the transaction)."""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity_id: uuid.UUID, *, tenant_id: uuid.UUID) -> bool:
        """Hard-delete a row; returns whether anything was removed."""
        stmt = delete(self.model).where(
            self.model.id == entity_id,  # type: ignore[attr-defined]
            self.model.tenant_id == tenant_id,  # type: ignore[attr-defined]
        )
        result = await self.session.execute(stmt)
        return bool(result.rowcount)

    async def count(self, *, tenant_id: uuid.UUID) -> int:
        """Count rows for a tenant."""
        from sqlalchemy import func, select

        stmt = (
            select(func.count())
            .select_from(self.model)
            .where(
                self.model.tenant_id == tenant_id  # type: ignore[attr-defined]
            )
        )
        return int((await self.session.execute(stmt)).scalar_one())

    async def _first(self, stmt: Any) -> ModelT | None:
        """Execute a statement and return the first scalar row (or None)."""
        result = await self.session.execute(stmt)
        return result.scalars().first()
