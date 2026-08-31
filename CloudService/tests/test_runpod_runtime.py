"""RunPod public-URL resolution and health accelerator fields."""

from __future__ import annotations

import os

from CloudService.api.runtime import (
    apply_runtime_defaults,
    resolve_public_api_url,
)


def test_explicit_public_url_wins(monkeypatch) -> None:
    monkeypatch.setenv("CLOUD_API_PUBLIC_URL", "https://cloud.example.test/")
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123xyz")
    assert resolve_public_api_url() == "https://cloud.example.test"


def test_runpod_proxy_url_from_pod_id(monkeypatch) -> None:
    monkeypatch.delenv("CLOUD_API_PUBLIC_URL", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123xyz")
    monkeypatch.setenv("CLOUD_API_PORT", "8787")
    assert resolve_public_api_url() == "https://abc123xyz-8787.proxy.runpod.net"


def test_localhost_fallback_when_not_on_runpod(monkeypatch) -> None:
    monkeypatch.delenv("CLOUD_API_PUBLIC_URL", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.setenv("CLOUD_API_HOST", "0.0.0.0")
    monkeypatch.setenv("CLOUD_API_PORT", "8787")
    assert resolve_public_api_url() == "http://127.0.0.1:8787"


def test_apply_runtime_defaults_fills_storefront(monkeypatch) -> None:
    monkeypatch.delenv("CLOUD_API_PUBLIC_URL", raising=False)
    monkeypatch.delenv("CLOUD_STOREFRONT_URL", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "podid")
    monkeypatch.setenv("CLOUD_API_PORT", "8787")
    apply_runtime_defaults()
    assert os.environ["CLOUD_API_PUBLIC_URL"] == "https://podid-8787.proxy.runpod.net"
    assert os.environ["CLOUD_STOREFRONT_URL"] == "https://podid-8787.proxy.runpod.net"


def test_health_reports_accelerator(client) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["device"] in {"cpu", "cuda"}
    assert isinstance(body["cuda"], bool)
    assert body["public_url"] == "http://testserver"
