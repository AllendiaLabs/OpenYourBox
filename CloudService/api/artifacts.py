"""Checkpoint list plus signed/local artifact download routes."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from CloudService.api.auth import CloudAPIError, error_response, require_session
from CloudService.api.runtime import resolve_public_api_url
from CloudService.api.state import STORE, Checkpoint, Job, Session, isoformat

router = APIRouter()

DOWNLOAD_TTL_SECONDS = 600


def _public_api() -> str:
    """Return the advertised origin used in signed download URLs."""
    return resolve_public_api_url()


def _owned_job(job_id: str, session: Session) -> Job:
    with STORE.lock:
        job = STORE.jobs.get(job_id)
        if job is None or job.customer_id != session.customer_id:
            raise CloudAPIError(404, "not_found", "Unknown job.")
        return job


def issue_signed_download(payload: bytes, filename: str) -> dict[str, str]:
    """Register a time-limited path token for ``payload`` (no query-string secrets)."""
    if not payload:
        raise CloudAPIError(409, "conflict", "Artifact bytes are not available.")
    secret = os.environ.get("CLOUD_DOWNLOAD_SECRET", "staging-dev-secret").encode("utf-8")
    nonce = uuid.uuid4().hex
    digest = hmac.new(secret, nonce.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    token = f"{nonce}.{digest}"
    expires = time.time() + DOWNLOAD_TTL_SECONDS
    with STORE.lock:
        STORE.signed_downloads[token] = (expires, payload, filename)
    return {
        "url": f"{_public_api()}/v1/downloads/{token}",
        "expires_at": isoformat(expires),
    }


def register_checkpoint(job: Job, payload: bytes, step: int, stage: str) -> Checkpoint | None:
    """Publish a real intermediate export. Empty payloads are ignored."""
    if not payload:
        return None
    checkpoint = Checkpoint(
        checkpoint_id=STORE.new_id("ckpt"),
        step=int(step),
        stage=str(stage or ""),
        created_at=isoformat(),
        payload=payload,
    )
    job.checkpoints.append(checkpoint)
    artifact_dir = STORE.artifact_dir(job.job_id)
    if artifact_dir is not None:
        ckpt_dir = artifact_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / f"{checkpoint.checkpoint_id}.pt").write_bytes(payload)
    STORE.persist_hint()
    return checkpoint


def register_final_artifact(job: Job, payload: bytes) -> bool:
    """Store a Gold-loadable final artifact. Returns False when bytes are empty."""
    if not payload:
        return False
    job.final_artifact = payload
    job.final_artifact_id = STORE.new_id("art")
    artifact_dir = STORE.artifact_dir(job.job_id)
    if artifact_dir is not None:
        (artifact_dir / "final.pt").write_bytes(payload)
    STORE.persist_hint()
    return True


def checkpoint_summary(job: Job) -> list[dict[str, object]]:
    """Return listable checkpoint metadata (no raw bytes)."""
    return [
        {
            "checkpoint_id": item.checkpoint_id,
            "step": item.step,
            "stage": item.stage,
            "created_at": item.created_at,
        }
        for item in job.checkpoints
    ]


@router.get("/v1/jobs/{job_id}/checkpoints")
def list_checkpoints(job_id: str, session: Session = Depends(require_session)) -> dict:
    """List published checkpoints for a job owned by the caller."""
    job = _owned_job(job_id, session)
    return {"checkpoints": checkpoint_summary(job)}


@router.get("/v1/jobs/{job_id}/checkpoints/{checkpoint_id}/download")
def download_checkpoint(
    job_id: str, checkpoint_id: str, session: Session = Depends(require_session)
) -> dict:
    """Return a signed URL for real checkpoint bytes."""
    job = _owned_job(job_id, session)
    with STORE.lock:
        match = next((c for c in job.checkpoints if c.checkpoint_id == checkpoint_id), None)
    if match is None or not match.payload:
        raise CloudAPIError(404, "not_found", "Unknown checkpoint.")
    return issue_signed_download(match.payload, f"{checkpoint_id}.pt")


@router.get("/v1/jobs/{job_id}/artifact/download")
def download_final_artifact(
    job_id: str, session: Session = Depends(require_session)
) -> dict:
    """Return a signed URL for the final artifact when the job succeeded."""
    job = _owned_job(job_id, session)
    if job.status != "succeeded" or not job.final_artifact:
        raise CloudAPIError(
            409, "conflict", "Final artifact is available only after success."
        )
    filename = f"{job.final_artifact_id or job.job_id}.pt"
    return issue_signed_download(job.final_artifact, filename)


@router.get("/v1/downloads/{token}")
def fetch_signed_download(token: str) -> Response:
    """Serve bytes for a time-limited signed path token (no Bearer in the query)."""
    with STORE.lock:
        record = STORE.signed_downloads.get(token)
    if record is None:
        return error_response(404, "not_found", "Download token is unknown.")
    expires, payload, filename = record
    if time.time() > expires:
        with STORE.lock:
            STORE.signed_downloads.pop(token, None)
        return error_response(404, "not_found", "Download token expired.")
    if not payload:
        return error_response(409, "conflict", "Artifact bytes are not available.")
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
