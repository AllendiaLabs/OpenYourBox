"""Device-code style platform account link bootstrap."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request

from CloudService.api.auth import error_response
from CloudService.api.state import STORE, LinkChallenge, Session

router = APIRouter()

LINK_TTL_SECONDS = 600


def _public_api() -> str:
    return os.environ.get("CLOUD_API_PUBLIC_URL", "http://127.0.0.1:8787").rstrip("/")


def _storefront() -> str:
    return os.environ.get("CLOUD_STOREFRONT_URL", _public_api()).rstrip("/")


@router.post("/v1/auth/link/start")
def link_start() -> dict:
    """Start a storefront device-code link challenge."""
    device_code = STORE.new_id("dev")
    user_code = STORE.new_id("UC")[-8:].upper()
    user_code = f"{user_code[:4]}-{user_code[4:]}"
    challenge = LinkChallenge(
        device_code=device_code,
        user_code=user_code,
        expires_at=time.time() + LINK_TTL_SECONDS,
    )
    with STORE.lock:
        STORE.links[device_code] = challenge
        STORE.links_by_user[user_code] = device_code
    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_url": f"{_storefront()}/link?code={user_code}",
        "expires_in": LINK_TTL_SECONDS,
        "interval": challenge.interval,
    }


@router.post("/v1/auth/link/token")
async def link_token(request: Request) -> dict | object:
    """Exchange a completed device-code flow for a linked session."""
    body = await request.json()
    device_code = str(body.get("device_code", "") or "")
    with STORE.lock:
        challenge = STORE.links.get(device_code)
        if challenge is None:
            return error_response(400, "link_expired", "Unknown or expired link code.")
        if time.time() > challenge.expires_at:
            return error_response(400, "link_expired", "Link code expired. Start again.")
        if not challenge.verified:
            return error_response(400, "link_pending", "Waiting for storefront verification.")
        access = STORE.new_id("tok")
        refresh = STORE.new_id("ref")
        session = Session(
            access_token=access,
            customer_id=challenge.customer_id or STORE.new_id("cust"),
            refresh_token=refresh,
            expires_at=time.time() + 86400 * 7,
        )
        STORE.sessions[access] = session
        STORE.refresh_tokens[refresh] = access
        STORE.links.pop(device_code, None)
        STORE.links_by_user.pop(challenge.user_code, None)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": 86400 * 7,
        "customer_id": session.customer_id,
    }


@router.post("/v1/auth/refresh")
async def refresh_session(request: Request) -> dict | object:
    """Issue a new access token from a refresh token."""
    body = await request.json()
    refresh = str(body.get("refresh_token", "") or "")
    with STORE.lock:
        access = STORE.refresh_tokens.get(refresh)
        if not access:
            return error_response(401, "unauthorized", "Refresh token is invalid.")
        old = STORE.sessions.get(access)
        if old is None or old.revoked:
            return error_response(401, "unauthorized", "Refresh token is invalid.")
        new_access = STORE.new_id("tok")
        new_refresh = STORE.new_id("ref")
        session = Session(
            access_token=new_access,
            customer_id=old.customer_id,
            refresh_token=new_refresh,
            expires_at=time.time() + 86400 * 7,
        )
        old.revoked = True
        STORE.sessions[new_access] = session
        STORE.refresh_tokens.pop(refresh, None)
        STORE.refresh_tokens[new_refresh] = new_access
    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "Bearer",
        "expires_in": 86400 * 7,
    }
