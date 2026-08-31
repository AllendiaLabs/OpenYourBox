"""Induced failures must never publish dummy ``succeeded`` artifacts."""

from __future__ import annotations

import time

from CloudService.api.state import STORE
from CloudService.tests.conftest import link_headers, mapping_manifest, wav_files


def test_invalid_wav_package_fails_without_dummy_success(live_client) -> None:
    headers = link_headers(live_client, "cust-fail")
    manifest = mapping_manifest(total_steps=2)
    files = wav_files(manifest)
    files["file:x.wav"] = ("x.wav", b"not-a-wav", "application/octet-stream")
    files["file:y.wav"] = ("y.wav", b"not-a-wav", "application/octet-stream")
    response = live_client.post("/v1/jobs", headers=headers, files=files)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    deadline = time.time() + 60.0
    last = None
    while time.time() < deadline:
        last = live_client.get(f"/v1/jobs/{job_id}", headers=headers).json()
        if last.get("status") in {"failed", "succeeded", "stopped"}:
            break
        time.sleep(0.2)
    assert last is not None
    assert last["status"] == "failed"
    assert last["has_final_artifact"] is False
    assert STORE.jobs[job_id].final_artifact is None
    download = live_client.get(f"/v1/jobs/{job_id}/artifact/download", headers=headers)
    assert download.status_code == 409
