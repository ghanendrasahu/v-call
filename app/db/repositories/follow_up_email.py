"""Follow-up email repository: draft persistence + send tracking."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from app.db.models import FollowUpEmail
from app.db.repositories.base import BaseRepository


class FollowUpEmailRepository(BaseRepository[FollowUpEmail]):
    """Async data access for drafted follow-up emails."""

    model = FollowUpEmail

    async def replace_for_meeting(
        self,
        *,
        meeting_id: uuid.UUID,
        tenant_id: uuid.UUID,
        emails: list[dict],
    ) -> list[FollowUpEmail]:
        """Idempotently replace drafts for a meeting (agent re-runs safe).

        Already-sent emails are preserved so re-analysis never re-sends.
        """
        from sqlalchemy import delete

        await self.session.execute(
            delete(FollowUpEmail).where(
                FollowUpEmail.meeting_id == meeting_id,
                FollowUpEmail.tenant_id == tenant_id,
                FollowUpEmail.sent.is_(False),
            )
        )
        rows = [
            FollowUpEmail(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                recipient_email=item["recipient_email"],
                recipient_name=item.get("recipient_name"),
                subject=item.get("subject"),
                email_body=item["email_body"],
            )
            for item in emails
        ]
        self.session.add_all(rows)
        await self.session.flush()
        return rows

    async def list_unsent(
        self,
        meeting_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
    ) -> list[FollowUpEmail]:
        """Drafts still waiting to be sent."""
        stmt = (
            select(FollowUpEmail)
            .where(
                FollowUpEmail.meeting_id == meeting_id,
                FollowUpEmail.tenant_id == tenant_id,
                FollowUpEmail.sent.is_(False),
            )
            .order_by(FollowUpEmail.created_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def mark_sent(
        self,
        email_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
        mcp_message_id: str | None = None,
    ) -> FollowUpEmail | None:
        """Record successful delivery."""
        email = await self.get(email_id, tenant_id=tenant_id)
        if email is None:
            return None
        email.sent = True
        email.mcp_message_id = mcp_message_id
        email.sent_at = datetime.now(UTC)
        await self.session.flush()
        return email
