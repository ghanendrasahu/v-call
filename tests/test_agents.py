"""Tests for the LangGraph agent orchestrator.

The graph is exercised end-to-end with injected fake structured-LLM objects
(exactly the ``with_structured_output`` surface the nodes rely on), so no
network or model weights are needed.
"""

from __future__ import annotations

import pytest

from app.agents.graph import (
    MAX_EXTRACTION_RETRIES,
    MeetingAnalysisGraph,
    route_after_extraction,
)
from app.agents.nodes.dispatcher import DispatcherRegistry, make_dispatcher_node
from app.agents.nodes.email import make_email_node
from app.agents.nodes.extraction import make_extraction_node
from app.agents.nodes.sentiment import make_sentiment_node
from app.agents.state import (
    ActionItemSchema,
    EmailDraft,
    EmailsOutput,
    ExtractionOutput,
    MeetingState,
    SentimentOutput,
    SpeakerDynamics,
    _merge_dicts,
    _merge_lists,
    _replace_list,
)
from tests.conftest import FailsOnce, FakeStructured

# ── Sample structured payloads ───────────────────────────────────────────
EXTRACTION = ExtractionOutput(
    action_items=[
        ActionItemSchema(
            assignee="Priya",
            task="Ship the RBAC spec to the platform team",
            priority="high",
            due_date="2026-10-20",
            target_system="jira",
        ),
        ActionItemSchema(task="Book the follow-up sync", priority="low"),
    ],
    decisions_made=["Adopt PostgreSQL 15 for the pilot"],
    unresolved_debates=["Whether the launch slips a week"],
)

SENTIMENT = SentimentOutput(
    overall="positive",
    engagement_score=0.82,
    speaker_dynamics={
        "Speaker 00": SpeakerDynamics(talk_time_ratio=0.6, interruptions=1),
        "Speaker 01": SpeakerDynamics(talk_time_ratio=0.4, interruptions=0),
    },
    summary="Energetic discussion; tension only around the date.",
)

EMAILS = EmailsOutput(
    emails=[
        EmailDraft(
            recipient_email="priya@acme.in",
            recipient_name="Priya",
            subject="Action items from today's sync",
            email_body="Thanks Priya — decisions, your actions, open debates.",
        )
    ]
)

SEGMENTS = [
    {"speaker_label": "Speaker 00", "start_ms": 0, "end_ms": 2000, "text": "Let's ship on time."},
    {"speaker_label": "Speaker 01", "start_ms": 2000, "end_ms": 4000, "text": "I agree."},
]
PARTICIPANTS = [{"name": "Priya", "email": "priya@acme.in"}]


class CapturingStructured(FakeStructured):
    """Fake structured model that also records the prompts it received."""

    def __init__(self, payload):
        super().__init__(payload)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        return await super().ainvoke(prompt)


class FailAlways:
    """Structured model that always raises."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt: str):
        self.calls += 1
        raise RuntimeError("llm unavailable")


class FakePort:
    """DispatchPort double."""

    def __init__(self, external_id: str = "JIRA-42", error: Exception | None = None):
        self.external_id = external_id
        self.error = error
        self.calls: list[dict] = []

    async def dispatch_action_item(self, item, *, tenant_id, meeting_id) -> str:
        self.calls.append({"item": item, "tenant_id": tenant_id, "meeting_id": meeting_id})
        if self.error is not None:
            raise self.error
        return self.external_id


def _base_state(**overrides) -> MeetingState:
    state: MeetingState = {
        "meeting_id": "m-1",
        "tenant_id": "t-1",
        "transcript_segments": SEGMENTS,
        "action_items": [],
        "decisions_made": [],
        "unresolved_debates": [],
        "sentiment_analysis": {},
        "follow_up_emails": [],
        "participants": PARTICIPANTS,
        "integration_status": {},
        "errors": [],
        "retry_count": 0,
    }
    state.update(overrides)
    return state


# ── State reducers ───────────────────────────────────────────────────────
def test_merge_lists_deduplicates():
    merged = _merge_lists(["a"], ["b", "a", "c"])
    assert merged == ["a", "b", "c"]
    assert _merge_lists(None, None) == []


def test_replace_list_is_wholesale():
    assert _replace_list(["old", "rows"], ["new"]) == ["new"]
    assert _replace_list(None, None) == []
    assert _replace_list(["kept"], None) == ["kept"]


def test_merge_dicts_later_wins():
    assert _merge_dicts({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}
    assert _merge_dicts(None, None) == {}


# ── Routing / retry policy ───────────────────────────────────────────────
def test_route_continues_when_extraction_succeeds():
    assert route_after_extraction(_base_state()) == "sentiment"
    # Explicit success flag (written by the node) also continues.
    assert route_after_extraction(_base_state(extraction_failed=False)) == "sentiment"


def test_route_retries_extraction_on_error():
    state = _base_state(errors=["extraction: boom"], extraction_failed=True, retry_count=0)
    assert route_after_extraction(state) == "extraction"


def test_route_gives_up_after_max_retries():
    state = _base_state(
        errors=["extraction: boom"],
        extraction_failed=True,
        retry_count=MAX_EXTRACTION_RETRIES,
    )
    assert route_after_extraction(state) == "sentiment"


def test_route_ignores_non_extraction_errors():
    state = _base_state(errors=["sentiment: boom"], retry_count=0)
    assert route_after_extraction(state) == "sentiment"


def test_route_does_not_retry_after_a_recovered_attempt():
    """A stale ledger error must not re-enter extraction once it succeeded."""
    state = _base_state(
        errors=["extraction: transient LLM failure"],  # left by an earlier attempt
        extraction_failed=False,
        retry_count=1,
    )
    assert route_after_extraction(state) == "sentiment"


# ── Individual nodes ─────────────────────────────────────────────────────
async def test_extraction_node_formats_transcript():
    model = CapturingStructured(EXTRACTION)
    node = make_extraction_node(model)

    result = await node(_base_state())

    assert "Speaker 00: Let's ship on time." in model.prompts[0]
    assert result["decisions_made"] == EXTRACTION.decisions_made
    assert result["action_items"][0]["target_system"] == "jira"
    assert model.calls == 1


async def test_extraction_node_isolates_failures():
    node = make_extraction_node(FailAlways())
    result = await node(_base_state())

    assert result["errors"] == ["extraction: llm unavailable"]
    assert result["extraction_failed"] is True  # router reads THIS, not the ledger


async def test_sentiment_node_normalises_dynamics():
    node = make_sentiment_node(FakeStructured(SENTIMENT))
    result = await node(_base_state())

    sentiment = result["sentiment_analysis"]
    assert sentiment["overall"] == "positive"
    assert sentiment["engagement_score"] == 0.82
    assert sentiment["speaker_dynamics"]["Speaker 00"] == {
        "talk_time_ratio": 0.6,
        "interruptions": 1,
    }


async def test_sentiment_failure_degrades_gracefully():
    node = make_sentiment_node(FailAlways())
    result = await node(_base_state())

    assert result["errors"] == ["sentiment: llm unavailable"]
    assert result["sentiment_analysis"]["overall"] == "unknown"


async def test_email_node_skips_when_no_recipients():
    model = CapturingStructured(EMAILS)
    node = make_email_node(model)

    out = await node(_base_state(participants=[{"name": "No Email"}]))

    assert out == {}
    assert model.calls == 0


async def test_email_node_drafts_per_attendee():
    node = make_email_node(FakeStructured(EMAILS))
    result = await node(_base_state())

    [email] = result["follow_up_emails"]
    assert email["recipient_email"] == "priya@acme.in"
    assert email["subject"].startswith("Action items")


# ── Dispatcher node ──────────────────────────────────────────────────────
async def test_dispatcher_routes_and_records_status():
    reg = DispatcherRegistry()
    port = FakePort(external_id="PROJ-7")
    reg.register("jira", port)
    node = make_dispatcher_node(reg)

    state = _base_state(
        action_items=[
            {"task": "Build it", "target_system": "jira", "priority": "high"},
            {"task": "Note it", "target_system": "none"},
            {"task": "Wikipedia it", "target_system": "notion"},
        ]
    )
    result = await node(state)

    jira_item, none_item, notion_item = result["action_items"]
    assert jira_item["status"] == "dispatched"
    assert jira_item["external_id"] == "PROJ-7"
    assert none_item["status"] == "open"
    assert notion_item["status"] == "skipped"  # not connected for this tenant
    assert result["integration_status"] == {
        "jira": "created: PROJ-7",
        "notion": "skipped (not connected)",
    }
    assert result["errors"] == []
    # The dispatcher received tenant/meeting scoping.
    assert port.calls[0]["tenant_id"] == "t-1"
    assert port.calls[0]["meeting_id"] == "m-1"


async def test_dispatcher_isolates_connector_failure():
    reg = DispatcherRegistry()
    reg.register("jira", FakePort(error=RuntimeError("jira 500")))
    node = make_dispatcher_node(reg)

    result = await node(_base_state(action_items=[{"task": "x", "target_system": "jira"}]))

    assert result["action_items"][0]["status"] == "failed"
    assert result["integration_status"]["jira"].startswith("failed:")
    assert result["errors"][0].startswith("dispatch[jira]:")


def test_registry_lifecycle():
    reg = DispatcherRegistry()
    port = FakePort()
    reg.register("jira", port)
    reg.register("notion", FakePort())

    assert reg.systems() == ["jira", "notion"]
    assert reg.get("jira") is port
    assert reg.get("zoho_crm") is None
    reg.clear()
    assert reg.systems() == []


# ── Full graph runs ──────────────────────────────────────────────────────
async def test_full_graph_happy_path():
    extraction = CapturingStructured(EXTRACTION)
    registry = DispatcherRegistry()
    port = FakePort(external_id="JIRA-1001")
    registry.register("jira", port)

    graph = MeetingAnalysisGraph(
        registry=registry,
        extraction_llm=extraction,
        sentiment_llm=FakeStructured(SENTIMENT),
        email_llm=FakeStructured(EMAILS),
    )
    final = await graph.run(
        meeting_id="m-1",
        tenant_id="t-1",
        transcript_segments=SEGMENTS,
        participants=PARTICIPANTS,
    )

    assert extraction.calls == 1
    assert final["retry_count"] == 0
    assert final["errors"] == []
    assert final["decisions_made"] == EXTRACTION.decisions_made
    assert final["unresolved_debates"] == EXTRACTION.unresolved_debates
    assert final["sentiment_analysis"]["overall"] == "positive"
    assert final["follow_up_emails"][0]["recipient_email"] == "priya@acme.in"
    # Dispatcher rewrote the action items in place (replace reducer).
    assert final["action_items"][0]["external_id"] == "JIRA-1001"
    assert final["action_items"][1]["status"] == "open"
    assert final["integration_status"]["jira"] == "created: JIRA-1001"
    assert port.calls[0]["item"]["task"] == final["action_items"][0]["task"]


async def test_graph_retries_transient_extraction_failure():
    registry = DispatcherRegistry()
    graph = MeetingAnalysisGraph(
        registry=registry,
        extraction_llm=FailsOnce(EXTRACTION),  # fails once, then succeeds
        sentiment_llm=FakeStructured(SENTIMENT),
        email_llm=FakeStructured(EMAILS),
    )
    final = await graph.run(
        meeting_id="m-1",
        tenant_id="t-1",
        transcript_segments=SEGMENTS,
        participants=PARTICIPANTS,
    )

    # One retry was consumed and the pipeline still produced real output.
    assert final["retry_count"] == 1
    assert final["action_items"] != []
    assert final["sentiment_analysis"]["overall"] == "positive"
    # The transient error remains on the errors ledger for observability.
    assert any(e.startswith("extraction:") for e in final["errors"])


async def test_graph_bounds_extraction_retries_and_continues_degraded():
    failing_extraction = FailAlways()
    registry = DispatcherRegistry()
    graph = MeetingAnalysisGraph(
        registry=registry,
        extraction_llm=failing_extraction,
        sentiment_llm=FakeStructured(SENTIMENT),
        email_llm=FakeStructured(EMAILS),
    )
    final = await graph.run(
        meeting_id="m-1",
        tenant_id="t-1",
        transcript_segments=SEGMENTS,
        participants=PARTICIPANTS,
    )

    # Initial attempt + MAX retries, then sentiment/email still ran.
    assert failing_extraction.calls == MAX_EXTRACTION_RETRIES + 1
    assert final["retry_count"] == MAX_EXTRACTION_RETRIES
    assert final["action_items"] == []
    assert final["sentiment_analysis"]["overall"] == "positive"
    # Identical failures collapse to one entry in the de-duplicating ledger.
    assert any(e.startswith("extraction:") for e in final["errors"])


@pytest.mark.parametrize("with_participants", [True, False])
async def test_graph_runs_without_participants(with_participants: bool):
    graph = MeetingAnalysisGraph(
        registry=DispatcherRegistry(),
        extraction_llm=FakeStructured(EXTRACTION),
        sentiment_llm=FakeStructured(SENTIMENT),
        email_llm=FakeStructured(EMAILS),
    )
    final = await graph.run(
        meeting_id="m-1",
        tenant_id="t-1",
        transcript_segments=SEGMENTS,
        participants=PARTICIPANTS if with_participants else None,
    )

    if with_participants:
        assert len(final["follow_up_emails"]) == 1
    else:
        assert final["follow_up_emails"] == []
    assert final["errors"] == []
