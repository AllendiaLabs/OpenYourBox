"""Entitlement gate and one-job-per-account contract tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from CloudService.api.state import STORE
from CloudService.tests.conftest import link_headers


def _mapping_manifest() -> dict:
    return {
        "schema_version": 1,
        "operation": "train_steerable",
        "client": {"plugin_version": "test", "client_instance_id": "inst-a"},
        "train_options": {"objective": "mapping", "total_steps": 8, "checkpoint_interval": 4},
        "capture_set": {"pairs": [{"pair_id": "p1", "x_name": "x.wav", "y_name": "y.wav", "kind": "pair"}]},
    }


def test_unauthenticated_jobs_are_unauthorized(client) -> None:
    response = client.post("/v1/jobs", json=_mapping_manifest())
    assert response.status_code == 401
    assert response.json()["error_code"] == "unauthorized"


def test_anonymous_submit_allowed_when_enabled(
    client, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CLOUD_ALLOW_ANONYMOUS", "1")
    monkeypatch.setenv("CLOUD_DATA_DIR", str(tmp_path))
    STORE.reset()
    from CloudService.api.app import create_app

    files = {
        "manifest": (
            None,
            __import__("json").dumps(_mapping_manifest()),
            "application/json",
        ),
        "file:x.wav": ("x.wav", b"RIFF....", "application/octet-stream"),
        "file:y.wav": ("y.wav", b"RIFF....", "application/octet-stream"),
    }
    with TestClient(create_app()) as anon_client:
        response = anon_client.post("/v1/jobs", files=files)
    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_insufficient_entitlement_refuses_submit(client) -> None:
    headers = link_headers(client, "cust-poor")
    STORE.set_entitlement("cust-poor", False)
    files = {
        "manifest": (None, __import__("json").dumps(_mapping_manifest()), "application/json"),
        "file:x.wav": ("x.wav", b"RIFF....", "application/octet-stream"),
        "file:y.wav": ("y.wav", b"RIFF....", "application/octet-stream"),
    }
    response = client.post("/v1/jobs", headers=headers, files=files)
    assert response.status_code == 403
    assert response.json()["error_code"] == "insufficient_entitlement"


def test_one_job_per_account(client) -> None:
    headers = link_headers(client, "cust-busy")
    STORE.set_entitlement("cust-busy", True)
    files = {
        "manifest": (None, __import__("json").dumps(_mapping_manifest()), "application/json"),
        "file:x.wav": ("x.wav", b"RIFF....", "application/octet-stream"),
        "file:y.wav": ("y.wav", b"RIFF....", "application/octet-stream"),
    }
    first = client.post("/v1/jobs", headers=headers, files=files)
    assert first.status_code == 202
    second = client.post("/v1/jobs", headers=headers, files=files)
    assert second.status_code == 409
    assert second.json()["error_code"] == "one_job_per_account"


def test_entitlement_probe_reports_insufficient(client) -> None:
    headers = link_headers(client, "cust-probe")
    STORE.set_entitlement("cust-probe", False, "0 credits")
    response = client.get("/v1/entitlement", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["sufficient"] is False
    assert "0 credits" in body["balance_hint"]
