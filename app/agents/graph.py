"""LangGraph orchestration: extraction → sentiment → email → dispatch.

Stateful, typed, with a bounded retry loop around extraction (the only node
whose failure would poison downstream output) and per-node error isolation —
a failing sentiment/email node degrades gracefully instead of killing the run.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.agents.nodes.dispatcher import DispatcherRegistry, make_dispatcher_node
from app.agents.nodes.email import make_email_node
from app.agents.nodes.extraction import make_extraction_node
from app.agents.nodes.sentiment import make_sentiment_node
from app.agents.state import MeetingState

logger = logging.getLogger(__name__)

MAX_EXTRACTION_RETRIES = 2


def route_after_extraction(
    state: MeetingState,
) -> Literal["sentiment", "extraction"]:
    """Retry extraction on failure (bounded), otherwise continue.

    Consults ``extraction_failed`` (written by the extraction node for the
    *latest* attempt) rather than the append-only ``errors`` ledger — a stale
    error must not force retries after a successful attempt.
    """
    if not state.get("extraction_failed"):
        return "sentiment"
    retries = state.get("retry_count", 0)
    if retries < MAX_EXTRACTION_RETRIES:
        logger.warning(
            "Extraction retry %s/%s for meeting %s",
            retries + 1,
            MAX_EXTRACTION_RETRIES,
            state.get("meeting_id"),
        )
        return "extraction"
    # Exhausted — continue with degraded (empty) extraction so sentiment and
    # emails still run on the raw transcript.
    logger.error(
        "Extraction exhausted retries for meeting %s; continuing degraded",
        state.get("meeting_id"),
    )
    return "sentiment"


def _retry_counter(state: MeetingState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph(
    *,
    extraction_llm: Any | None = None,
    sentiment_llm: Any | None = None,
    email_llm: Any | None = None,
    registry: DispatcherRegistry | None = None,
) -> Any:
    """Assemble the meeting-analysis StateGraph (injectable LLMs for tests)."""

    extraction_node = make_extraction_node(extraction_llm)
    sentiment_node = make_sentiment_node(sentiment_llm)
    email_node = make_email_node(email_llm)
    dispatcher_node = make_dispatcher_node(registry)

    builder = StateGraph(MeetingState)
    builder.add_node("extraction", extraction_node)
    builder.add_node("retry_counter", _retry_counter)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("email", email_node)
    builder.add_node("dispatcher", dispatcher_node)

    builder.add_edge(START, "extraction")
    builder.add_conditional_edges(
        "extraction",
        route_after_extraction,
        {"sentiment": "sentiment", "extraction": "retry_counter"},
    )
    # Bump the counter, then re-enter extraction.
    builder.add_edge("retry_counter", "extraction")
    builder.add_edge("sentiment", "email")
    builder.add_edge("email", "dispatcher")
    builder.add_edge("dispatcher", END)

    return builder.compile()


class MeetingAnalysisGraph:
    """Thin service wrapper: run the graph and normalise the outcome."""

    def __init__(self, *, registry: DispatcherRegistry | None = None, **llms: Any) -> None:
        self._graph = build_graph(registry=registry, **llms)

    async def run(
        self,
        *,
        meeting_id: str,
        tenant_id: str,
        transcript_segments: list[dict],
        participants: list[dict] | None = None,
    ) -> MeetingState:
        """Execute the full pipeline and return the final state."""
        initial: MeetingState = {
            "meeting_id": meeting_id,
            "tenant_id": tenant_id,
            "transcript_segments": transcript_segments,
            "action_items": [],
            "decisions_made": [],
            "unresolved_debates": [],
            "sentiment_analysis": {},
            "follow_up_emails": [],
            "integration_status": {},
            "errors": [],
            "participants": participants or [],
            "retry_count": 0,
        }
        final: MeetingState = await self._graph.ainvoke(initial)
        return final


async def run_analysis(
    *,
    meeting_id: str,
    tenant_id: str,
    transcript_segments: list[dict],
    participants: list[dict] | None = None,
) -> MeetingState:
    """Module-level convenience wrapper (used by Celery workers)."""
    return await MeetingAnalysisGraph().run(
        meeting_id=meeting_id,
        tenant_id=tenant_id,
        transcript_segments=transcript_segments,
        participants=participants,
    )
