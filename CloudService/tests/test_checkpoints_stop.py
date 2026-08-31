"""Real checkpoint publication and cooperative Stop mid-run."""

from __future__ import annotations

import time
from urllib.parse import urlparse

import pytest

from CloudService.api.state import STORE
from CloudService.tests.conftest import link_headers, mapping_manifest, wav_files

pytest.importorskip("torch")


def test_checkpoints_downloadable_then_stop_has_no_final_artifact(live_client) -> None:
    headers = link_headers(live_client, "cust-ckpt")
    manifest = mapping_manifest(total_steps=8, checkpoint_interval=1, export_checkpoints=True)
    response = live_client.post("/v1/jobs", headers=headers, files=wav_files(manifest))
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    deadline = time.time() + 120.0
    listed = None
    while time.time() < deadline:
        listed = live_client.get(f"/v1/jobs/{job_id}/checkpoints", headers=headers)
        assert listed.status_code == 200
        checkpoints = listed.json().get("checkpoints") or []
        status = live_client.get(f"/v1/jobs/{job_id}", headers=headers).json()["status"]
        if checkpoints:
            break
        if status in {"failed", "stopped", "succeeded"}:
            raise AssertionError({"status": status, "checkpoints": checkpoints})
        time.sleep(0.2)
    else:
        raise TimeoutError("No checkpoint published while running")

    checkpoint_id = listed.json()["checkpoints"][0]["checkpoint_id"]
    download = live_client.get(
        f"/v1/jobs/{job_id}/checkpoints/{checkpoint_id}/download", headers=headers
    )
    assert download.status_code == 200
    bytes_response = live_client.get(urlparse(download.json()["url"]).path)
    assert bytes_response.status_code == 200
    assert len(bytes_response.content) > 0

    stopped = live_client.post(f"/v1/jobs/{job_id}/stop", headers=headers)
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    assert STORE.jobs[job_id].final_artifact is None
    remaining = live_client.get(f"/v1/jobs/{job_id}/checkpoints", headers=headers)
    assert remaining.json()["checkpoints"]
    final = live_client.get(f"/v1/jobs/{job_id}/artifact/download", headers=headers)
    assert final.status_code == 409
