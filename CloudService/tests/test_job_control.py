"""Stop-only job control tests (no pause/resume, no fake-success worker)."""

from __future__ import annotations

import json

from CloudService.api.state import STORE
from CloudService.tests.conftest import link_headers


def _submit(client, headers) -> str:
    manifest = {
        "schema_version": 1,
        "operation": "train_steerable",
        "train_options": {"objective": "mapping", "total_steps": 40},
        "capture_set": {"pairs": [{"pair_id": "p1", "x_name": "x.wav", "y_name": "y.wav"}]},
    }
    files = {
        "manifest": (None, json.dumps(manifest), "application/json"),
        "file:x.wav": ("x.wav", b"RIFF", "application/octet-stream"),
        "file:y.wav": ("y.wav", b"RIFF", "application/octet-stream"),
    }
    response = client.post("/v1/jobs", headers=headers, files=files)
    assert response.status_code == 202
    return response.json()["job_id"]


def test_stop_queued_job(client) -> None:
    headers = link_headers(client)
    job_id = _submit(client, headers)
    stopped = client.post(f"/v1/jobs/{job_id}/stop", headers=headers)
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    assert STORE.jobs[job_id].final_artifact is None
    illegal_stop = client.post(f"/v1/jobs/{job_id}/stop", headers=headers)
    assert illegal_stop.status_code == 409
    assert illegal_stop.json()["error_code"] == "conflict"


def test_pause_and_resume_routes_are_absent(client) -> None:
    headers = link_headers(client)
    job_id = _submit(client, headers)
    paused = client.post(f"/v1/jobs/{job_id}/pause", headers=headers)
    assert paused.status_code == 404
    resumed = client.post(f"/v1/jobs/{job_id}/resume", headers=headers)
    assert resumed.status_code == 404


def test_health_unauthenticated(client) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
