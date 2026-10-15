"""Integration credential management (encrypted at rest, masked in output)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TenantContext, get_current_context, get_db
from app.db.enums import IntegrationProvider
from app.db.repositories import IntegrationCredentialRepository
from app.schemas.api import CredentialIn, CredentialOut

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.get("", response_model=list[CredentialOut])
async def list_integrations(
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> list[CredentialOut]:
    """List connected providers (tokens always masked)."""
    repo = IntegrationCredentialRepository(session)
    credentials = await repo.list(tenant_id=context.tenant_id, limit=100)
    return [CredentialOut(**repo.masked_view(c)) for c in credentials]


@router.put("/{provider}", response_model=CredentialOut)
async def upsert_credential(
    provider: IntegrationProvider,
    payload: CredentialIn,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> CredentialOut:
    """Create or rotate the stored credential for a provider."""
    repo = IntegrationCredentialRepository(session)
    credential = await repo.upsert(
        tenant_id=context.tenant_id,
        provider=provider,
        # For token-style providers the primary secret lives in 'token'.
        access_token=payload.values.get("token") or payload.values.get("access_token") or "",
        refresh_token=payload.values.get("refresh_token"),
        scopes=payload.scopes,
        provider_metadata=payload.values,
    )
    return CredentialOut(**repo.masked_view(credential))


@router.delete("/{provider}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_credential(
    provider: IntegrationProvider,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> None:
    """Disconnect a provider for this tenant."""
    repo = IntegrationCredentialRepository(session)
    if not await repo.revoke(context.tenant_id, provider):
        raise HTTPException(status_code=404, detail="Provider not connected")
