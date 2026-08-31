"""Staging storefront handlers for device-code account link completion.

This module verifies platform account linking only. It must never mark a
training job succeeded or fabricate train artifacts.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from CloudService.api.auth import error_body
from CloudService.api.state import STORE

router = APIRouter()


@router.get("/link", response_class=HTMLResponse)
def link_page(code: str = "") -> str:
    """Minimal verification page shown in the system browser."""
    escaped = code.replace("<", "")
    return f"""<!DOCTYPE html>
<html><head><title>OpenYourBox account link</title></head>
<body>
  <h1>Link OpenYourBox</h1>
  <p>User code: <strong>{escaped}</strong></p>
  <p>This staging storefront completes the device-code account link only.
     It does not start or fake training.</p>
  <form method="post" action="/mock/link/complete">
    <input type="hidden" name="user_code" value="{escaped}"/>
    <input type="hidden" name="customer_id" value="mock-customer"/>
    <button type="submit">Confirm link</button>
  </form>
</body></html>
"""


@router.post("/mock/link/complete")
async def complete_link(request: Request) -> JSONResponse:
    """Mark a device-code challenge as verified (tests and mock page)."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        user_code = str(body.get("user_code", "") or "")
        customer_id = str(body.get("customer_id", "") or "mock-customer")
    else:
        form = await request.form()
        user_code = str(form.get("user_code", "") or "")
        customer_id = str(form.get("customer_id", "") or "mock-customer")
    with STORE.lock:
        device_code = STORE.links_by_user.get(user_code)
        challenge = STORE.links.get(device_code or "")
        if challenge is None:
            return JSONResponse(
                status_code=404,
                content=error_body("not_found", "Unknown user code."),
            )
        challenge.verified = True
        challenge.customer_id = customer_id or STORE.new_id("cust")
    return JSONResponse({"ok": True, "customer_id": challenge.customer_id})
