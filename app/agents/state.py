"""Typed graph state + structured-output schemas.

``MeetingState`` is the single source of truth flowing through the graph.
Agents return partial updates (LangGraph merges them), so each node only
touches its own keys.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field


def _merge_lists(existing: list | None, new: list | None) -> list:
    """Reducer: append + de-duplicate (agent retries must not double-append)."""
    merged = list(existing or [])
    for item in new or []:
        if item not in merged:
            merged.append(item)
    return merged


def _replace_list(existing: list | None, new: list | None) -> list:
    """Reducer: replace wholesale.

    Used for collections that a downstream node *rewrites* (the dispatcher
    adds status/external_id to every item) — appending would duplicate rows.
    """
    return list(new) if new is not None else list(existing or [])


def _merge_dicts(existing: dict | None, new: dict | None) -> dict:
    """Reducer: shallow merge (later writes win)."""
    merged = dict(existing or {})
    merged.update(new or {})
    return merged


class MeetingState(TypedDict):
    """State schema carried across the agent graph."""

    meeting_id: str
    tenant_id: str
    transcript_segments: list[dict]

    # Extraction outputs (dispatcher rewrites the whole list → replace)
    action_items: Annotated[list[dict], _replace_list]
    decisions_made: Annotated[list[str], _merge_lists]
    unresolved_debates: Annotated[list[str], _merge_lists]

    # Sentiment outputs
    sentiment_analysis: dict

    # Email drafting outputs
    follow_up_emails: Annotated[list[dict], _replace_list]
    # Optional attendees passed in by the caller: [{"name": ..., "email": ...}]
    participants: NotRequired[list[dict]]

    # Integration dispatch results: {"jira": "created: PROJ-1", "notion": "failed"}
    integration_status: Annotated[dict, _merge_dicts]

    # Error & retry bookkeeping
    errors: Annotated[list[str], _merge_lists]
    retry_count: NotRequired[int]
    #: True only while the *latest* extraction attempt failed. The errors
    #: ledger is append-only (and de-duplicating), so the router must consult
    #: this flag instead — otherwise one transient failure would burn every
    #: retry even after a successful attempt.
    extraction_failed: NotRequired[bool]


# ── Structured LLM output schemas (Instructor / with_structured_output) ─
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ActionItemSchema(_Strict):
    """One extracted action item."""

    assignee: str | None = Field(default=None, description="Person responsible, if named")
    task: str = Field(min_length=3, description="Imperative, self-contained task")
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    due_date: str | None = Field(
        default=None, description="ISO date (YYYY-MM-DD) if mentioned, else null"
    )
    target_system: Literal["jira", "notion", "zoho_crm", "zoho_desk", "freshdesk", "none"] = "none"


class ExtractionOutput(_Strict):
    """Result of the extraction agent."""

    action_items: list[ActionItemSchema] = Field(default_factory=list)
    decisions_made: list[str] = Field(default_factory=list)
    unresolved_debates: list[str] = Field(default_factory=list)


class SpeakerDynamics(_Strict):
    talk_time_ratio: float = Field(ge=0, le=1, description="Share of airtime, 0..1")
    interruptions: int = Field(default=0, ge=0)


class SentimentOutput(_Strict):
    """Result of the sentiment agent."""

    overall: Literal["positive", "neutral", "negative", "mixed"] = "neutral"
    engagement_score: float = Field(ge=0, le=1, description="0 = disengaged, 1 = highly engaged")
    speaker_dynamics: dict[str, SpeakerDynamics] = Field(default_factory=dict)
    summary: str = Field(default="", description="1-2 sentence team mood summary")


class EmailDraft(_Strict):
    """One drafted follow-up email."""

    recipient_email: str
    recipient_name: str | None = None
    subject: str = Field(min_length=3)
    email_body: str = Field(min_length=10)


class EmailsOutput(_Strict):
    """Result of the email drafting agent."""

    emails: list[EmailDraft] = Field(default_factory=list)
