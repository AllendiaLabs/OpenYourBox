"""Entitlement probe route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from CloudService.api.auth import require_session
from CloudService.api.state import Session
from CloudService.storefront import entitlement as entitlement_provider

router = APIRouter()


@router.get("/v1/entitlement")
def get_entitlement(session: Session = Depends(require_session)) -> dict:
    """Return whether a new cloud job may be submitted."""
    ent = entitlement_provider.sync_from_storefront(session.customer_id)
    return {"sufficient": ent.sufficient, "balance_hint": ent.balance_hint}
