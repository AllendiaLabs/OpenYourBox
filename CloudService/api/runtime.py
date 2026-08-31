"""Runtime helpers for bind address, advertised origin, and accelerator health.

Used by the control plane and the RunPod GPU entrypoint so signed download URLs
and plugin overrides match the public origin (localhost or RunPod HTTP proxy).
"""

from __future__ import annotations

import os
from typing import Any


def api_bind_host() -> str:
    """Return the address uvicorn should bind.

    Containers (including RunPod) default to all interfaces. Local staging keeps
    loopback unless ``CLOUD_API_HOST`` is set.
    """
    return os.environ.get("CLOUD_API_HOST", "127.0.0.1").strip() or "127.0.0.1"


def api_bind_port() -> int:
    """Return the TCP port uvicorn should bind (default 8787)."""
    raw = os.environ.get("CLOUD_API_PORT", "8787").strip() or "8787"
    try:
        return max(1, int(raw))
    except ValueError:
        return 8787


def resolve_public_api_url() -> str:
    """Return the advertised API origin (no trailing slash).

    Precedence: ``CLOUD_API_PUBLIC_URL``, then RunPod HTTP proxy from
    ``RUNPOD_POD_ID``, then a localhost URL derived from the bind port.
    """
    explicit = os.environ.get("CLOUD_API_PUBLIC_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    port = api_bind_port()
    if pod_id:
        return f"https://{pod_id}-{port}.proxy.runpod.net"
    host = api_bind_host()
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def apply_runtime_defaults() -> None:
    """Fill advertised URL env vars when the operator left them unset.

    Safe to call from process startup. Does not override an explicit
    ``CLOUD_API_PUBLIC_URL`` or ``CLOUD_STOREFRONT_URL``.
    """
    public = resolve_public_api_url()
    if not os.environ.get("CLOUD_API_PUBLIC_URL", "").strip():
        os.environ["CLOUD_API_PUBLIC_URL"] = public
    if not os.environ.get("CLOUD_STOREFRONT_URL", "").strip():
        os.environ["CLOUD_STOREFRONT_URL"] = public


def accelerator_health() -> dict[str, Any]:
    """Return CUDA availability without failing if PyTorch is missing."""
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return {"cuda": cuda, "device": "cuda" if cuda else "cpu"}
