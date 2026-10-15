"""Integration credential repository — encrypt-on-write, decrypt-on-read."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.crypto import (
    DecryptedCredential,
    DecryptionError,
    TokenCipher,
    get_token_cipher,
    mask_token,
)
from app.db.enums import IntegrationProvider
from app.db.models import IntegrationCredential
from app.db.repositories.base import BaseRepository


class IntegrationCredentialRepository(BaseRepository[IntegrationCredential]):
    """Tenant-scoped OAuth tokens, encrypted at rest with Fernet."""

    model = IntegrationCredential

    def __init__(self, session: AsyncSession, cipher: TokenCipher | None = None) -> None:
        super().__init__(session)
        self._cipher = cipher

    @property
    def cipher(self) -> TokenCipher:
        return self._cipher or get_token_cipher()

    async def upsert(
        self,
        *,
        tenant_id: uuid.UUID,
        provider: IntegrationProvider,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: datetime | None = None,
        token_type: str = "Bearer",
        scopes: list[str] | None = None,
        provider_metadata: dict | None = None,
    ) -> IntegrationCredential:
        """Create or update the encrypted credential for a provider."""
        existing = await self.get_for_provider(tenant_id, provider)
        encrypted_access = self.cipher.encrypt(access_token)
        encrypted_refresh = self.cipher.encrypt(refresh_token) if refresh_token else None

        if existing is not None:
            existing.access_token = encrypted_access
            existing.refresh_token = encrypted_refresh
            existing.expires_at = expires_at
            existing.token_type = token_type
            existing.scopes = scopes
            existing.provider_metadata = provider_metadata
            await self.session.flush()
            return existing

        credential = IntegrationCredential(
            tenant_id=tenant_id,
            provider=provider,
            access_token=encrypted_access,
            refresh_token=encrypted_refresh,
            expires_at=expires_at,
            token_type=token_type,
            scopes=scopes,
            provider_metadata=provider_metadata,
        )
        return await self.add(credential)

    async def get_for_provider(
        self,
        tenant_id: uuid.UUID,
        provider: IntegrationProvider,
    ) -> IntegrationCredential | None:
        """Fetch the raw (still-encrypted) credential row."""
        stmt = select(IntegrationCredential).where(
            IntegrationCredential.tenant_id == tenant_id,
            IntegrationCredential.provider == provider,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_decrypted(
        self,
        tenant_id: uuid.UUID,
        provider: IntegrationProvider,
    ) -> DecryptedCredential | None:
        """Fully decrypt a credential and bump ``last_used_at``."""
        credential = await self.get_for_provider(tenant_id, provider)
        if credential is None:
            return None
        try:
            access_token = self.cipher.decrypt(credential.access_token)
            refresh_token = (
                self.cipher.decrypt(credential.refresh_token) if credential.refresh_token else None
            )
        except DecryptionError:
            # Key rotation / corruption: surface a clear error to the caller.
            raise
        credential.last_used_at = datetime.now(UTC)
        await self.session.flush()
        return DecryptedCredential(
            provider=provider.value,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    async def revoke(
        self,
        tenant_id: uuid.UUID,
        provider: IntegrationProvider,
    ) -> bool:
        """Delete a stored credential (user disconnects the integration)."""
        credential = await self.get_for_provider(tenant_id, provider)
        if credential is None:
            return False
        await self.session.delete(credential)
        await self.session.flush()
        return True

    @staticmethod
    def masked_view(credential: IntegrationCredential) -> dict:
        """Safe-to-log representation: tokens are masked, never plaintext."""
        return {
            "id": str(credential.id),
            "provider": credential.provider.value,
            "token_type": credential.token_type,
            "expires_at": credential.expires_at.isoformat() if credential.expires_at else None,
            "scopes": credential.scopes,
            "access_token_masked": mask_token(credential.access_token),
            "has_refresh_token": credential.refresh_token is not None,
            "last_used_at": (
                credential.last_used_at.isoformat() if credential.last_used_at else None
            ),
        }
