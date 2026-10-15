"""Multi-tenant ORM models.

Every domain table carries ``tenant_id`` for row-level isolation. Relationships
cascade on tenant deletion so GDPR/DPDP "right to erasure" requests are a
single ``DELETE`` away.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import GUID, Base, JSONType, TimestampMixin
from app.db.enums import (
    ActionItemPriority,
    ActionItemStatus,
    AgentRunStatus,
    IntegrationProvider,
    MeetingSource,
    MeetingStatus,
    TargetSystem,
    UserRole,
)


def _enum(enum_cls: type, length: int = 48) -> SAEnum:
    """Portable VARCHAR-backed enum (no native PG enum types)."""
    return SAEnum(
        enum_cls,
        name=enum_cls.__name__.lower(),
        native_enum=False,
        length=length,
        create_constraint=False,
        values_callable=lambda members: [member.value for member in members],
    )


# ── Tenancy ─────────────────────────────────────────────────────────────
class Tenant(Base, TimestampMixin):
    """Root tenancy boundary. Everything else hangs off this."""

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(96), nullable=False, unique=True, index=True)
    plan: Mapped[str] = mapped_column(String(32), nullable=False, default="starter")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    settings: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    users: Mapped[list[User]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan", passive_deletes=True
    )
    meetings: Mapped[list[Meeting]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan", passive_deletes=True
    )
    credentials: Mapped[list[IntegrationCredential]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan", passive_deletes=True
    )


class User(Base, TimestampMixin):
    """A member of a tenant. Unique email *per tenant*."""

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(_enum(UserRole), nullable=False, default=UserRole.MEMBER)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    tenant: Mapped[Tenant] = relationship(back_populates="users")


# ── Meetings & transcripts ──────────────────────────────────────────────
class Meeting(Base, TimestampMixin):
    """A single meeting and its processing lifecycle."""

    __tablename__ = "meetings"
    __table_args__ = (Index("ix_meetings_tenant_status", "tenant_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False, default="Untitled meeting")
    source: Mapped[MeetingSource] = mapped_column(
        _enum(MeetingSource), nullable=False, default=MeetingSource.UPLOAD
    )
    provider_meeting_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    external_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    status: Mapped[MeetingStatus] = mapped_column(
        _enum(MeetingStatus), nullable=False, default=MeetingStatus.PENDING, index=True
    )
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_metadata: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tenant: Mapped[Tenant] = relationship(back_populates="meetings")
    segments: Mapped[list[TranscriptSegment]] = relationship(
        back_populates="meeting",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TranscriptSegment.start_ms",
    )
    action_items: Mapped[list[ActionItem]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", passive_deletes=True
    )
    follow_up_emails: Mapped[list[FollowUpEmail]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", passive_deletes=True
    )
    agent_runs: Mapped[list[AgentRun]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", passive_deletes=True
    )


class TranscriptSegment(Base, TimestampMixin):
    """A diarized utterance window (``Speaker 00`` @ t0→t1)."""

    __tablename__ = "transcript_segments"
    __table_args__ = (Index("ix_segments_meeting_start", "meeting_id", "start_ms"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, nullable=False, index=True)

    speaker_label: Mapped[str] = mapped_column(String(64), nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    meeting: Mapped[Meeting] = relationship(back_populates="segments")


# ── Intelligence outputs ────────────────────────────────────────────────
class ActionItem(Base, TimestampMixin):
    """A task extracted by the agents and optionally dispatched to a SaaS."""

    __tablename__ = "action_items"
    __table_args__ = (Index("ix_action_items_tenant_status", "tenant_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, nullable=False, index=True)

    assignee: Mapped[str | None] = mapped_column(String(255), nullable=True)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[ActionItemPriority] = mapped_column(
        _enum(ActionItemPriority), nullable=False, default=ActionItemPriority.MEDIUM
    )
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    target_system: Mapped[TargetSystem] = mapped_column(
        _enum(TargetSystem), nullable=False, default=TargetSystem.NONE
    )
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[ActionItemStatus] = mapped_column(
        _enum(ActionItemStatus), nullable=False, default=ActionItemStatus.OPEN, index=True
    )

    meeting: Mapped[Meeting] = relationship(back_populates="action_items")


class FollowUpEmail(Base, TimestampMixin):
    """A drafted follow-up email queued for delivery via the Gmail MCP server."""

    __tablename__ = "follow_up_emails"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, nullable=False, index=True)

    recipient_email: Mapped[str] = mapped_column(String(320), nullable=False)
    recipient_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(500), nullable=True)
    email_body: Mapped[str] = mapped_column(Text, nullable=False)
    sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mcp_message_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    meeting: Mapped[Meeting] = relationship(back_populates="follow_up_emails")


class AgentRun(Base, TimestampMixin):
    """Observability record for one full LangGraph run over a meeting."""

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, nullable=False, index=True)

    status: Mapped[AgentRunStatus] = mapped_column(
        _enum(AgentRunStatus), nullable=False, default=AgentRunStatus.RUNNING
    )
    graph_state: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    errors: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    meeting: Mapped[Meeting] = relationship(back_populates="agent_runs")


# ── Integrations ────────────────────────────────────────────────────────
class IntegrationCredential(Base, TimestampMixin):
    """Encrypted OAuth tokens for one provider, scoped to one tenant."""

    __tablename__ = "integration_credentials"
    __table_args__ = (
        UniqueConstraint("tenant_id", "provider", name="uq_credentials_tenant_provider"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[IntegrationProvider] = mapped_column(
        _enum(IntegrationProvider), nullable=False
    )

    # Encrypted at rest with Fernet — never read these directly in app code.
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Bearer")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scopes: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    provider_metadata: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tenant: Mapped[Tenant] = relationship(back_populates="credentials")
