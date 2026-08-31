"""Crash/reconciler tests: stale heartbeat becomes failed / worker_lost."""

from __future__ import annotations

import time

from CloudService.api.jobs import reconcile_lost_workers
from CloudService.api.state import STORE
from CloudService.tests.conftest import link_headers, mapping_manifest, wav_files


def test_stale_heartbeat_fails_job_and_clears_active_slot(client) -> None:
    headers = link_headers(client, "cust-lost")
    response = client.post(
        "/v1/jobs", headers=headers, files=wav_files(mapping_manifest(total_steps=8))
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    job = STORE.jobs[job_id]
    job.status = "running"
    job.worker_heartbeat_at = time.time() - 999
    STORE.worker_threads.pop(job_id, None)

    failed = reconcile_lost_workers()
    assert job_id in failed
    assert job.status == "failed"
    assert job.error_code == "worker_lost"
    assert "lost" in job.error_message.lower()

    second = client.post(
        "/v1/jobs", headers=headers, files=wav_files(mapping_manifest(total_steps=8))
    )
    assert second.status_code == 202
    detail = client.get(f"/v1/jobs/{job_id}", headers=headers)
    assert detail.json()["error_code"] == "worker_lost"
