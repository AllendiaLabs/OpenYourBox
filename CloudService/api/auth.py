"""Bearer session helpers and authenticated FastAPI dependency."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from CloudService.api.state import STORE, Session

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

# Stable guest identity used when CLOUD_ALLOW_ANONYMOUS is enabled.
GUEST_ACCESS_TOKEN = "tok-anonymous-guest"
GUEST_CUSTOMER_ID = "cust-anonymous"


def anonymous_allowed() -> bool:
    """Return True when Cloud jobs may run without an Allendia link."""
    return os.environ.get("CLOUD_ALLOW_ANONYMOUS", "0") != "0"


def ensure_guest_session() -> Session:
    """Return the process-wide anonymous session, creating it if needed."""
    with STORE.lock:
        session = STORE.sessions.get(GUEST_ACCESS_TOKEN)
        if session is None or session.revoked:
            session = Session(
                access_token=GUEST_ACCESS_TOKEN,
                customer_id=GUEST_CUSTOMER_ID,
                refresh_token="",
                revoked=False,
                expires_at=0.0,
            )
            STORE.sessions[GUEST_ACCESS_TOKEN] = session
        return session


class CloudAPIError(Exception):
    """Control-plane error that serializes to the contract envelope."""

    def __init__(self, status_code: int, error_code: str, error_message: str) -> None:
        super().__init__(error_message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_message = error_message


def error_body(error_code: str, error_message: str) -> dict[str, str]:
    """Return the standard `{error_code, error_message}` payload."""
    return {"error_code": error_code, "error_message": error_message}


def error_response(status_code: int, error_code: str, error_message: str) -> JSONResponse:
    """Return an HTTP error using the contract envelope (no secrets)."""
    return JSONResponse(
        status_code=status_code,
        content=error_body(error_code, error_message),
    )


async def cloud_api_error_handler(_request: Request, exc: CloudAPIError) -> JSONResponse:
    """FastAPI exception handler for :class:`CloudAPIError`."""
    return error_response(exc.status_code, exc.error_code, exc.error_message)


def optional_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Extract a Bearer token, or empty string when none was sent."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        return ""
    return (credentials.credentials or "").strip()


def lookup_session(access_token: str) -> Session:
    """Resolve a live linked session for ``access_token``."""
    with STORE.lock:
        session = STORE.sessions.get(access_token)
        if (
            session is None
            or session.revoked
            or (session.expires_at and time.time() > session.expires_at)
        ):
            raise CloudAPIError(
                401, "unauthorized", "Linked session is invalid or expired."
            )
        return session


def require_session(token: str = Depends(optional_bearer_token)) -> Session:
    """FastAPI dependency: linked session, or anonymous guest when allowed."""
    if not token:
        if anonymous_allowed():
            return ensure_guest_session()
        raise CloudAPIError(401, "unauthorized", "Linked session required.")
    return lookup_session(token)


@router.post("/v1/auth/logout")
def logout(session: Session = Depends(require_session)) -> dict[str, bool]:
    """Invalidate the current linked session (best-effort)."""
    with STORE.lock:
        session.revoked = True
        STORE.refresh_tokens.pop(session.refresh_token, None)
    return {"ok": True}
