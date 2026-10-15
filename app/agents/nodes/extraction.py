"""Extraction agent: action items, decisions, unresolved debates."""

from __future__ import annotations

import logging
from typing import Any

from app.agents.llm import get_structured_model, invoke_structured
from app.agents.state import ExtractionOutput, MeetingState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert meeting analyst. Given a diarized transcript, extract:

1. action_items — concrete follow-up tasks. Each must be self-contained
   (someone who never attended should understand it), assigned to a named
   speaker when possible, with a priority (low/medium/high/critical) and an
   ISO due_date ONLY if one is explicitly stated or unambiguously implied
   ("by Friday" given today's date). Map to target_system when the speaker
   names a tool ("file a Jira ticket" → jira); otherwise 'none'.
2. decisions_made — agreements that closed a topic. Not restatements.
3. unresolved_debates — disagreements still open at the end of the meeting.

Rules:
- Never invent people, dates, or tasks.
- Prefer many small precise tasks over one vague task.
- Respond with ONLY the structured object.
"""


def _format_transcript(state: MeetingState) -> str:
    lines = []
    for seg in state.get("transcript_segments", []):
        speaker = seg.get("speaker_label", "Speaker ??")
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines) or "(empty transcript)"


def make_extraction_node(llm: Any | None = None) -> Any:
    """Build the extraction node, binding the structured LLM once."""

    model = llm if llm is not None else get_structured_model(ExtractionOutput)

    async def extraction_node(state: MeetingState) -> dict:
        transcript = _format_transcript(state)
        try:
            result: ExtractionOutput = await invoke_structured(
                model, f"{SYSTEM_PROMPT}\n\nTRANSCRIPT:\n{transcript}"
            )
        except Exception as exc:
            logger.exception("Extraction failed for meeting %s", state.get("meeting_id"))
            return {"errors": [f"extraction: {exc}"]}

        action_items = [
            {
                "assignee": item.assignee,
                "task": item.task,
                "priority": item.priority,
                "due_date": item.due_date,
                "target_system": item.target_system,
            }
            for item in result.action_items
        ]
        return {
            "action_items": action_items,
            "decisions_made": result.decisions_made,
            "unresolved_debates": result.unresolved_debates,
        }

    return extraction_node
