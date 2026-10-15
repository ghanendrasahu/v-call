"""Domain enumerations shared by ORM models, Pydantic schemas, and agents.

All enums are ``str``-based so they serialise cleanly to JSON and render as
portable ``VARCHAR(n)`` columns (no native PG enums — easier migrations).
"""

from __future__ import annotations

import enum


class UserRole(enum.StrEnum):
    ADMIN = "admin"
    MEMBER = "member"
    SERVICE = "service"


class MeetingSource(enum.StrEnum):
    GOOGLE_MEET = "google_meet"
    ZOOM = "zoom"
    UPLOAD = "upload"
    OTHER = "other"


class MeetingStatus(enum.StrEnum):
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionItemPriority(enum.StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TargetSystem(enum.StrEnum):
    JIRA = "jira"
    NOTION = "notion"
    ZOHO_CRM = "zoho_crm"
    ZOHO_DESK = "zoho_desk"
    FRESHDESK = "freshdesk"
    NONE = "none"


class ActionItemStatus(enum.StrEnum):
    OPEN = "open"
    DISPATCHED = "dispatched"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class IntegrationProvider(enum.StrEnum):
    GOOGLE_CALENDAR = "google_calendar"
    GOOGLE_GMAIL = "google_gmail"
    JIRA = "jira"
    NOTION = "notion"
    ZOHO_CRM = "zoho_crm"
    ZOHO_DESK = "zoho_desk"
    FRESHDESK = "freshdesk"


class AgentRunStatus(enum.StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
