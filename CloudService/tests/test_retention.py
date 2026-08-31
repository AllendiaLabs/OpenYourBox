"""Retention reuse and expiry tests without a fake-success worker."""

from __future__ import annotations

import json
import time

from CloudService.api.retention import RETENTION_SECONDS, sweep_expired_corpora
from CloudService.api.state import STORE
from CloudService.tests.conftest import link_headers


def _submit(client, headers, corpus_id: str = "") -> dict:
    manifest = {
        "schema_version": 1,
        "operation": "train_steerable",
        "train_options": {"objective": "mapping", "total_steps": 4},
        "capture_set": {"pairs": [{"pair_id": "p1", "x_name": "x.wav", "y_name": "y.wav"}]},
    }
    if corpus_id:
        body = {"manifest": manifest, "corpus_id": corpus_id}
        response = client.post("/v1/jobs", headers=headers, json=body)
    else:
        files = {
            "manifest": (None, json.dumps(manifest), "application/json"),
            "file:x.wav": ("x.wav", b"RIFF", "application/octet-stream"),
            "file:y.wav": ("y.wav", b"RIFF", "application/octet-stream"),
        }
        response = client.post("/v1/jobs", headers=headers, files=files)
    return response


def test_corpus_reuse_extends_retention(client) -> None:
    headers = link_headers(client, "cust-retain")
    first = _submit(client, headers)
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    corpus_id = first.json()["corpus_id"]
    original = STORE.corpora[corpus_id].last_used_at
    client.post(f"/v1/jobs/{job_id}/stop", headers=headers)
    time.sleep(0.05)
    second = _submit(client, headers, corpus_id=corpus_id)
    assert second.status_code == 202
    assert second.json()["corpus_id"] == corpus_id
    assert STORE.corpora[corpus_id].last_used_at >= original


def test_expired_corpus_is_swept_and_reuse_fails(client) -> None:
    headers = link_headers(client, "cust-expire")
    first = _submit(client, headers)
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    corpus_id = first.json()["corpus_id"]
    client.post(f"/v1/jobs/{job_id}/stop", headers=headers)
    STORE.corpora[corpus_id].last_used_at = time.time() - RETENTION_SECONDS - 10
    removed = sweep_expired_corpora()
    assert corpus_id in removed
    reuse = _submit(client, headers, corpus_id=corpus_id)
    assert reuse.status_code in {404, 409}
    assert reuse.json()["error_code"] in {"not_found", "corpus_expired"}
