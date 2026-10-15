"""API request/response schemas (Pydantic v2, strict)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.enums import (
    ActionItemPriority,
    ActionItemStatus,
    IntegrationProvider,
    MeetingSource,
    MeetingStatus,
    TargetSystem,
)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── Meetings ────────────────────────────────────────────────────────────
class MeetingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    source: MeetingSource = MeetingSource.UPLOAD
    provider_meeting_id: str | None = Field(default=None, max_length=255)
    external_url: str | None = Field(default=None, max_length=1024)


class MeetingSummary(ORMModel):
    id: uuid.UUID
    title: str
    source: MeetingSource
    status: MeetingStatus
    language: str | None
    duration_seconds: float | None
    occurred_at: datetime | None
    created_at: datetime
    completed_at: datetime | None


class TranscriptSegmentOut(ORMModel):
    speaker_label: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None


class ActionItemOut(ORMModel):
    id: uuid.UUID
    assignee: str | None
    task: str
    priority: ActionItemPriority
    due_date: object | None
    target_system: TargetSystem
    status: ActionItemStatus
    external_id: str | None


class SentimentOut(BaseModel):
    overall: str = "unknown"
    engagement_score: float = 0.0
    speaker_dynamics: dict = Field(default_factory=dict)
    summary: str = ""


class FollowUpEmailOut(ORMModel):
    recipient_email: str
    recipient_name: str | None
    subject: str | None
    email_body: str
    sent: bool
    sent_at: datetime | None


class MeetingDetail(MeetingSummary):
    segments: list[TranscriptSegmentOut] = Field(default_factory=list)
    action_items: list[ActionItemOut] = Field(default_factory=list)
    follow_up_emails: list[FollowUpEmailOut] = Field(default_factory=list)
    sentiment: SentimentOut | None = None


class MeetingAccepted(BaseModel):
    """202 response: processing has been queued."""

    meeting_id: uuid.UUID
    status: MeetingStatus
    detail: str


# ── Integrations ────────────────────────────────────────────────────────
class CredentialIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, str] = Field(
        min_length=1, description="Provider fields, e.g. {'api_token': ...}"
    )
    scopes: list[str] | None = None


class CredentialOut(BaseModel):
    id: uuid.UUID
    provider: IntegrationProvider
    token_type: str
    expires_at: datetime | None
    scopes: list[str] | None
    access_token_masked: str
    has_refresh_token: bool
    last_used_at: datetime | None


# ── Health ──────────────────────────────────────────────────────────────
class HealthOut(BaseModel):
    status: str
    service: str
    version: str
    checks: dict[str, str] = Field(default_factory=dict)
