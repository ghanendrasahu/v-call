"""Sentiment agent: overall mood, engagement, and speaker dynamics."""

from __future__ import annotations

import logging
from typing import Any

from app.agents.llm import get_structured_model, invoke_structured
from app.agents.state import MeetingState, SentimentOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an organisational psychologist analysing meeting dynamics.
From the diarized transcript, assess:

1. overall — collective mood: positive | neutral | negative | mixed.
2. engagement_score — 0..1 based on participation breadth, energy, and
   follow-through. Silence and monologues score lower.
3. speaker_dynamics — per speaker: talk_time_ratio (their share of the
   conversation, sums to ~1 across speakers) and interruptions (count of
   clear conversational takeovers).
4. summary — 1-2 sentences on team mood and any tension worth flagging.

Respond with ONLY the structured object.
"""


def make_sentiment_node(llm: Any | None = None) -> Any:
    """Build the sentiment node."""

    model = llm if llm is not None else get_structured_model(SentimentOutput)

    async def sentiment_node(state: MeetingState) -> dict:
        segments = state.get("transcript_segments", [])
        transcript = "\n".join(
            f"{s.get('speaker_label', 'Speaker ??')}: {s.get('text', '')}" for s in segments
        )
        try:
            result: SentimentOutput = await invoke_structured(
                model, f"{SYSTEM_PROMPT}\n\nTRANSCRIPT:\n{transcript or '(empty)'}"
            )
        except Exception as exc:
            logger.exception("Sentiment analysis failed for %s", state.get("meeting_id"))
            return {
                "errors": [f"sentiment: {exc}"],
                "sentiment_analysis": {
                    "overall": "unknown",
                    "engagement_score": 0.0,
                    "speaker_dynamics": {},
                },
            }

        return {
            "sentiment_analysis": {
                "overall": result.overall,
                "engagement_score": result.engagement_score,
                "speaker_dynamics": {
                    name: dynamics.model_dump()
                    for name, dynamics in result.speaker_dynamics.items()
                },
                "summary": result.summary,
            }
        }

    return sentiment_node
