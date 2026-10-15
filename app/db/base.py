"""Declarative base and shared column types.

SQLAlchemy 2.0 async engine + ``Mapped``/``mapped_column`` style throughout.
Column types are chosen to be portable: native ``UUID``/``JSONB`` on
PostgreSQL, but functional on SQLite so the async test-suite can run in-memory.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import JSON, DateTime, MetaData, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Stable constraint names so Alembic autogenerate diffs stay deterministic.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Cross-dialect UUID (native UUID on PostgreSQL, CHAR(32) on SQLite).
GUID = Uuid(as_uuid=True)

# JSONB on PostgreSQL, plain JSON elsewhere.
JSONType = JSON().with_variant(JSONB(), "postgresql")

# Alias for typing convenience in models.
JSONList = list


class Base(DeclarativeBase):
    """Async Declarative Base with a deterministic naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict: JSONType,
        list: JSONType,
    }


class TimestampMixin:
    """Adds ``created_at``/``updated_at`` maintained by the database clock."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
