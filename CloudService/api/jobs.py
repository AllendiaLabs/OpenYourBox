"""Job submit/list/get and Stop-only control (no pause/resume)."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from CloudService.api.artifacts import checkpoint_summary
from CloudService.api.auth import CloudAPIError, require_session
from CloudService.api.retention import (
    extend_corpus_on_reuse,
    pin_job_corpus,
    require_live_corpus,
    sweep_expired_corpora,
)
from CloudService.api.state import (
    ACTIVE_STATUSES,
    STORE,
    TERMINAL_STATUSES,
    Corpus,
    Job,
    Session,
    heartbeat_timeout_seconds,
    isoformat,
)
from CloudService.storefront import entitlement as entitlement_provider

router = APIRouter()

_supervisor_stop = threading.Event()
_supervisor_thread: threading.Thread | None = None


def job_payload(job: Job, *, detail: bool = False) -> dict[str, Any]:
    """Serialize a job for list/get/control responses."""
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "objective": job.objective,
        "step": job.step,
        "total_steps": job.total_steps,
        "stage": job.stage,
        "loss": job.loss,
        "device": job.device,
        "corpus_id": job.corpus_id,
        "error_code": job.error_code or None,
        "error_message": job.error_message or None,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "has_final_artifact": bool(job.final_artifact),
    }
    if detail:
        payload["checkpoints"] = checkpoint_summary(job)
    return payload


def _client_instance_id(manifest: dict[str, Any]) -> str:
    client = manifest.get("client")
    if isinstance(client, dict):
        return str(client.get("client_instance_id", "") or "")
    return ""


def _objective(manifest: dict[str, Any]) -> str:
    options = manifest.get("train_options")
    if not isinstance(options, dict):
        options = {}
    value = str(options.get("objective", "mapping") or "mapping").strip().lower()
    if value not in {"mapping", "reconstruction"}:
        raise CloudAPIError(400, "validation_failed", "Unknown train objective.")
    return value


def _total_steps(manifest: dict[str, Any], objective: str) -> int:
    options = manifest.get("train_options")
    if not isinstance(options, dict):
        options = {}
    if objective == "reconstruction":
        recipe = options.get("reconstruction")
        if not isinstance(recipe, dict):
            recipe = {}
        stage1 = int(recipe.get("stage1_steps", options.get("total_steps", 1)) or 1)
        stage2 = int(recipe.get("stage2_steps", 0) or 0)
        return max(1, stage1 + stage2)
    return max(1, int(options.get("total_steps", 100) or 100))


def _validate_capture(manifest: dict[str, Any], objective: str) -> None:
    capture = manifest.get("capture_set")
    if not isinstance(capture, dict):
        capture = {}
    pairs = capture.get("pairs") or []
    clips = capture.get("clips") or []
    if objective == "mapping":
        if not pairs:
            raise CloudAPIError(
                400, "validation_failed", "Mapping jobs require at least one pair."
            )
        if clips:
            raise CloudAPIError(
                400, "validation_failed", "Mapping cannot train unpaired clips."
            )
    elif not pairs and not clips:
        raise CloudAPIError(
            400, "validation_failed", "Reconstruction jobs require clips or pairs."
        )


def _owned_job(job_id: str, session: Session) -> Job:
    with STORE.lock:
        job = STORE.jobs.get(job_id)
        if job is None or job.customer_id != session.customer_id:
            raise CloudAPIError(404, "not_found", "Unknown job.")
        return job


async def _read_manifest_and_files(request: Request) -> tuple[dict[str, Any], dict[str, bytes], str]:
    """Parse multipart or JSON submit bodies."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        manifest = body.get("manifest", body)
        if not isinstance(manifest, dict):
            raise CloudAPIError(400, "validation_failed", "Manifest must be a JSON object.")
        corpus_id = str(body.get("corpus_id", manifest.get("corpus_id", "")) or "")
        return manifest, {}, corpus_id

    form = await request.form()
    manifest_part = form.get("manifest")
    if manifest_part is None:
        raise CloudAPIError(400, "validation_failed", "Missing manifest part.")
    if hasattr(manifest_part, "read"):
        raw = await manifest_part.read()
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    else:
        text = str(manifest_part)
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CloudAPIError(400, "validation_failed", "Manifest is not valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise CloudAPIError(400, "validation_failed", "Manifest must be a JSON object.")
    corpus_id = str(form.get("corpus_id", manifest.get("corpus_id", "")) or "")
    files: dict[str, bytes] = {}
    for key in form.keys():
        if not str(key).startswith("file:"):
            continue
        part = form[key]
        name = str(key)[5:]
        if hasattr(part, "read"):
            data = await part.read()
            filename = getattr(part, "filename", "") or name
            files[str(filename or name)] = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        elif isinstance(part, (bytes, bytearray)):
            files[name] = bytes(part)
    return manifest, files, corpus_id


def _store_corpus(customer_id: str, files: dict[str, bytes]) -> Corpus:
    corpus_id = STORE.new_id("corpus")
    corpus = Corpus(
        corpus_id=corpus_id,
        customer_id=customer_id,
        byte_size=sum(len(blob) for blob in files.values()),
        last_used_at=time.time(),
        files=files,
    )
    with STORE.lock:
        STORE.corpora[corpus_id] = corpus
        if STORE.data_dir is not None:
            dest = STORE.data_dir / "corpora" / corpus_id
            dest.mkdir(parents=True, exist_ok=True)
            for name, blob in files.items():
                (dest / name).write_bytes(blob)
    return corpus


def mark_job_failed(job: Job, error_code: str, error_message: str) -> None:
    """Terminal failure that never publishes a dummy success artifact."""
    job.status = "failed"
    job.error_code = error_code
    job.error_message = error_message
    job.updated_at = isoformat()
    job.pin_corpus = False
    STORE.persist_hint()


def mark_job_stopped(job: Job) -> None:
    """Terminal stop: no final success artifact."""
    job.status = "stopped"
    job.stop_requested = True
    job.error_code = ""
    job.error_message = "Stopped by user"
    job.updated_at = isoformat()
    job.pin_corpus = False
    STORE.persist_hint()


def write_stop_command(job: Job) -> None:
    """Ask the live runner to honor cooperative stop."""
    if not job.command_file:
        return
    from pathlib import Path

    path = Path(job.command_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"command": "stop"}), encoding="utf-8")


def auto_worker_enabled() -> bool:
    """Return True when queued jobs should be claimed by the real runner."""
    return os.environ.get("CLOUD_AUTO_WORKER", "1") != "0"


def claim_queued_jobs() -> None:
    """Claim at most one queued job and start the real train runner."""
    if not auto_worker_enabled():
        return
    job_id = ""
    with STORE.lock:
        if any(job.status == "running" for job in STORE.jobs.values()):
            return
        queued = [
            job
            for job in STORE.jobs.values()
            if job.status == "queued" and not job.stop_requested
        ]
        if not queued:
            return
        job = queued[0]
        job.status = "running"
        job.worker_heartbeat_at = time.time()
        job.updated_at = isoformat()
        job_id = job.job_id
    from CloudService.worker.train_runner import start_job_thread

    start_job_thread(job_id)


def reconcile_lost_workers() -> list[str]:
    """Mark running jobs with a dead worker or stale heartbeat as ``failed``.

    Already published checkpoints are kept. The one-job slot is cleared because
    ``failed`` is terminal. Returns failed job ids.
    """
    timeout = heartbeat_timeout_seconds()
    now = time.time()
    failed: list[str] = []
    with STORE.lock:
        for job in STORE.jobs.values():
            if job.status != "running":
                continue
            thread = STORE.worker_threads.get(job.job_id)
            alive = thread is not None and thread.is_alive()
            heartbeat = job.worker_heartbeat_at or 0.0
            stale = heartbeat > 0 and (now - heartbeat) > timeout
            if alive and not stale:
                continue
            job.status = "failed"
            job.error_code = "worker_lost"
            job.error_message = (
                "Cloud worker was lost. Published checkpoints remain available."
            )
            job.updated_at = isoformat()
            job.pin_corpus = False
            STORE.worker_threads.pop(job.job_id, None)
            failed.append(job.job_id)
        if failed:
            STORE.persist_hint()
    return failed


def supervisor_loop() -> None:
    """Background claim, reconcile, and retention sweep."""
    while not _supervisor_stop.wait(0.25):
        try:
            claim_queued_jobs()
            reconcile_lost_workers()
            sweep_expired_corpora()
        except Exception:
            continue


def start_supervisor() -> None:
    """Start the process supervisor if it is not already running."""
    global _supervisor_thread
    if _supervisor_thread is not None and _supervisor_thread.is_alive():
        return
    _supervisor_stop.clear()
    _supervisor_thread = threading.Thread(target=supervisor_loop, name="cloud-supervisor", daemon=True)
    _supervisor_thread.start()


def stop_supervisor() -> None:
    """Signal the supervisor thread to exit."""
    _supervisor_stop.set()


@router.get("/v1/jobs")
def list_jobs(session: Session = Depends(require_session)) -> dict:
    """List jobs for the authenticated platform customer (active first)."""
    jobs = STORE.jobs_for(session.customer_id)
    return {"jobs": [job_payload(job) for job in jobs]}


@router.post("/v1/jobs", status_code=202)
async def submit_job(request: Request, session: Session = Depends(require_session)) -> dict:
    """Accept a real-training job after auth, entitlement, and one-job gates."""
    entitlement = entitlement_provider.sync_from_storefront(session.customer_id)
    if not entitlement.sufficient:
        raise CloudAPIError(
            403,
            "insufficient_entitlement",
            "Allendia credits unavailable for a new cloud job.",
        )
    if STORE.active_job_for(session.customer_id) is not None:
        raise CloudAPIError(
            409,
            "one_job_per_account",
            "This Allendia account already has an active cloud job.",
        )

    manifest, files, corpus_id = await _read_manifest_and_files(request)
    if str(manifest.get("operation", "train_steerable") or "") != "train_steerable":
        raise CloudAPIError(400, "validation_failed", "Unsupported operation.")
    objective = _objective(manifest)
    _validate_capture(manifest, objective)

    if corpus_id:
        try:
            corpus = require_live_corpus(corpus_id, session.customer_id)
        except KeyError as exc:
            raise CloudAPIError(404, "not_found", "Unknown corpus.") from exc
        except TimeoutError as exc:
            raise CloudAPIError(
                409, "corpus_expired", "Referenced corpus is past retention."
            ) from exc
        extend_corpus_on_reuse(corpus)
    else:
        if not files:
            raise CloudAPIError(400, "validation_failed", "Job package contains no files.")
        corpus = _store_corpus(session.customer_id, files)

    now = isoformat()
    job = Job(
        job_id=STORE.new_id("job"),
        customer_id=session.customer_id,
        status="queued",
        objective=objective,
        manifest=manifest,
        corpus_id=corpus.corpus_id,
        created_at=now,
        updated_at=now,
        total_steps=_total_steps(manifest, objective),
        submitter_client_instance_id=_client_instance_id(manifest),
    )
    with STORE.lock:
        if STORE.active_job_for(session.customer_id) is not None:
            raise CloudAPIError(
                409,
                "one_job_per_account",
                "This Allendia account already has an active cloud job.",
            )
        STORE.jobs[job.job_id] = job
        pin_job_corpus(job)
        STORE.persist_hint()

    if auto_worker_enabled():
        threading.Thread(target=claim_queued_jobs, daemon=True).start()
    return job_payload(job)


@router.get("/v1/jobs/{job_id}")
def get_job(job_id: str, session: Session = Depends(require_session)) -> dict:
    """Return job detail including progress and checkpoint summaries."""
    return job_payload(_owned_job(job_id, session), detail=True)


@router.post("/v1/jobs/{job_id}/stop")
def stop_job(job_id: str, session: Session = Depends(require_session)) -> dict:
    """Cooperative stop from queued or running. Terminal ``stopped``, no success artifact."""
    job = _owned_job(job_id, session)
    with STORE.lock:
        if job.status in TERMINAL_STATUSES:
            raise CloudAPIError(409, "conflict", "Job is already terminal.")
        if job.status not in ACTIVE_STATUSES:
            raise CloudAPIError(409, "conflict", "Job cannot be stopped.")
        mark_job_stopped(job)
        write_stop_command(job)
    return job_payload(job)
