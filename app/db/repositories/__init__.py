"""Repository layer: tenant-scoped async data access.

Repositories own query construction and encryption concerns so services/routes
stay free of SQLAlchemy internals.
"""

from app.db.repositories.action_item import ActionItemRepository
from app.db.repositories.base import BaseRepository
from app.db.repositories.credential import IntegrationCredentialRepository
from app.db.repositories.follow_up_email import FollowUpEmailRepository
from app.db.repositories.meeting import MeetingRepository
from app.db.repositories.tenant import TenantRepository

__all__ = [
    "ActionItemRepository",
    "BaseRepository",
    "FollowUpEmailRepository",
    "IntegrationCredentialRepository",
    "MeetingRepository",
    "TenantRepository",
]
