"""Email drafting agent: personalised follow-ups per attendee."""

from __future__ import annotations

import logging
from typing import Any

from app.agents.llm import get_structured_model, invoke_structured
from app.agents.state import EmailsOutput, MeetingState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You write concise, professional follow-up emails after a meeting.

Input you will receive: the transcript, plus the list of attendees
(name + email). Draft ONE email per attendee, addressed to them by name.

Each email must:
- Open with a one-line thank-you / context.
- Include a short "Decisions" bullet list (only their relevant decisions).
- Include their personal "Your actions" bullet list (tasks assigned to
  them, or none if they have none).
- Flag open debates they should know about.
- Stay under 180 words. No placeholders like [insert link].
- Use a clear subject line.

Respond with ONLY the structured object. Skip attendees with no email.
"""


def make_email_node(llm: Any | None = None) -> Any:
    """Build the follow-up email drafting node."""

    model = llm if llm is not None else get_structured_model(EmailsOutput)

    async def email_node(state: MeetingState) -> dict:
        participants = state.get("participants", []) or []
        attendees = [p for p in participants if p.get("email")]
        if not attendees:
            # No known recipients → nothing to draft (not an error).
            return {}

        transcript = "\n".join(
            f"{s.get('speaker_label', 'Speaker ??')}: {s.get('text', '')}"
            for s in state.get("transcript_segments", [])
        )
        context = (
            f"ATTENDEES: {attendees}\n\n"
            f"ACTION ITEMS: {state.get('action_items', [])}\n\n"
            f"DECISIONS: {state.get('decisions_made', [])}\n\n"
            f"OPEN DEBATES: {state.get('unresolved_debates', [])}\n\n"
            f"TRANSCRIPT:\n{transcript or '(empty)'}"
        )
        try:
            result: EmailsOutput = await invoke_structured(model, f"{SYSTEM_PROMPT}\n\n{context}")
        except Exception as exc:
            logger.exception("Email drafting failed for %s", state.get("meeting_id"))
            return {"errors": [f"email: {exc}"]}

        return {
            "follow_up_emails": [
                {
                    "recipient_email": email.recipient_email,
                    "recipient_name": email.recipient_name,
                    "subject": email.subject,
                    "email_body": email.email_body,
                }
                for email in result.emails
            ]
        }

    return email_node
