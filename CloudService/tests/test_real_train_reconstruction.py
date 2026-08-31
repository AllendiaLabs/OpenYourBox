"""Short real reconstruction train smoke: success yields a loadable encode/decode artifact."""

from __future__ import annotations

import io
import time
from urllib.parse import urlparse

import pytest

from CloudService.tests.conftest import link_headers, reconstruction_manifest, wav_files

torch = pytest.importorskip("torch")


def _wait_status(client, headers, job_id: str, wanted: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last: dict | None = None
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}", headers=headers)
        last = response.json()
        if last.get("status") == wanted:
            return last
        if last.get("status") in {"failed", "stopped"} and wanted == "succeeded":
            raise AssertionError(last)
        time.sleep(0.25)
    raise TimeoutError(last)


def test_real_reconstruction_success_downloads_non_empty_artifact(live_client) -> None:
    headers = link_headers(live_client, "cust-rave")
    manifest = reconstruction_manifest()
    response = live_client.post("/v1/jobs", headers=headers, files=wav_files(manifest))
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    body = _wait_status(live_client, headers, job_id, "succeeded")
    assert body["status"] == "succeeded"
    assert body["has_final_artifact"] is True
    download = live_client.get(f"/v1/jobs/{job_id}/artifact/download", headers=headers)
    assert download.status_code == 200
    artifact = live_client.get(urlparse(download.json()["url"]).path)
    assert artifact.status_code == 200
    assert len(artifact.content) > 0
    loaded = torch.jit.load(io.BytesIO(artifact.content))
    audio = torch.zeros(1, 1, 256)
    encoded = loaded.encode(audio)
    decoded = loaded.decode(encoded)
    forwarded = loaded.forward(audio)
    assert encoded.dim() == 3
    assert decoded.dim() == 3
    assert forwarded.dim() == 3
