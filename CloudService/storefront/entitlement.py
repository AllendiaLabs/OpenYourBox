"""Mock storefront entitlement provider."""

from __future__ import annotations

from CloudService.api.state import STORE, Entitlement


def probe(customer_id: str) -> Entitlement:
    """Return authoritative entitlement for a platform customer.

    Production would sync with the WordPress commerce/membership ledger.
    """
    return STORE.get_entitlement(customer_id)


def sync_from_storefront(customer_id: str) -> Entitlement:
    """Stub storefront ledger sync (identity only in mock mode)."""
    return probe(customer_id)
