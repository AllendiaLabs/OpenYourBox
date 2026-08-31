"""Real cloud train runner: materialize package, invoke shared recipes, publish artifacts."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from CloudService.api.artifacts import register_checkpoint, register_final_artifact
from CloudService.api.state import STORE, Job, isoformat

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _select_device(options: dict[str, Any]) -> str:
    """Prefer the caller device, otherwise auto (CUDA/MPS/CPU inside the recipe)."""
    requested = str(options.get("device", "auto") or "auto").strip().lower()
    return requested or "auto"


def materialize_train_request(job: Job, work_dir: Path) -> dict[str, Any]:
    """Write corpus files into ``work_dir`` and rewrite capture names to absolute paths."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with STORE.lock:
        corpus = STORE.corpora.get(job.corpus_id)
        files = dict(corpus.files) if corpus is not None else {}
        manifest = copy.deepcopy(job.manifest)
    for name, blob in files.items():
        (work_dir / name).write_bytes(blob)

    capture = manifest.setdefault("capture_set", {})
    if not isinstance(capture, dict):
        capture = {}
        manifest["capture_set"] = capture
    for pair in capture.get("pairs", []) or []:
        if not isinstance(pair, dict):
            continue
        for name_key, path_key in (("x_name", "x_path"), ("y_name", "y_path")):
            name = str(pair.get(name_key, "") or "")
            if name:
                pair[path_key] = str(work_dir / name)
    for clip in capture.get("clips", []) or []:
        if not isinstance(clip, dict):
            continue
        name = str(clip.get("name", "") or clip.get("clip_id", "") or "")
        if name and not clip.get("path"):
            clip["path"] = str(work_dir / name)

    manifest["request_id"] = job.job_id
    manifest["operation"] = "train_steerable"
    options = manifest.setdefault("train_options", {})
    if not isinstance(options, dict):
        options = {}
        manifest["train_options"] = options
    options["device"] = _select_device(options)
    mlflow = options.setdefault("mlflow", {})
    if isinstance(mlflow, dict):
        mlflow["enabled"] = False
    return manifest


def _apply_progress(job_id: str, payload: dict[str, Any]) -> None:
    """Mirror recipe events into the job store and publish real checkpoints."""
    with STORE.lock:
        job = STORE.jobs.get(job_id)
        if job is None:
            return
        job.worker_heartbeat_at = time.time()
        if "step" in payload:
            try:
                job.step = int(payload.get("step") or job.step)
            except (TypeError, ValueError):
                pass
        if "total_steps" in payload:
            try:
                job.total_steps = int(payload.get("total_steps") or job.total_steps)
            except (TypeError, ValueError):
                pass
        if payload.get("stage"):
            job.stage = str(payload.get("stage") or "")
        if "loss" in payload:
            try:
                job.loss = float(payload.get("loss") or 0.0)
            except (TypeError, ValueError):
                pass
        if payload.get("device"):
            job.device = str(payload.get("device") or job.device)
        job.updated_at = isoformat()
        artifact = str(payload.get("artifact_path", "") or "")
        status = str(payload.get("status", "") or "")
        if artifact and status == "running":
            path = Path(artifact)
            if path.is_file() and path.stat().st_size > 0:
                register_checkpoint(job, path.read_bytes(), job.step, job.stage)


def _fail(job: Job, message: str, error_code: str = "train_failed") -> None:
    from CloudService.api.jobs import mark_job_failed

    with STORE.lock:
        if job.status in {"stopped", "failed"}:
            return
        job.final_artifact = None
        job.final_artifact_id = ""
        mark_job_failed(job, error_code, message)


def run_job(job_id: str) -> None:
    """Materialize one claimed job and invoke shared mapping/reconstruction recipes."""
    from CloudService.api.jobs import mark_job_stopped

    with STORE.lock:
        job = STORE.jobs.get(job_id)
        if job is None:
            return
        if job.stop_requested or job.status == "stopped":
            mark_job_stopped(job)
            return
        job.worker_pid = os.getpid()
        job.worker_heartbeat_at = time.time()

    if STORE.data_dir is not None:
        work_dir = STORE.data_dir / "jobs" / job_id / "work"
    else:
        work_dir = Path(tempfile.mkdtemp(prefix=f"oyb-{job_id}-"))
    artifact_dir = work_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command_file = work_dir / "command.json"
    command_file.write_text("{}", encoding="utf-8")
    with STORE.lock:
        job.work_dir = str(work_dir)
        job.command_file = str(command_file)
        if job.stop_requested:
            mark_job_stopped(job)
            return

    try:
        request = materialize_train_request(job, work_dir)
    except Exception as exc:  # noqa: BLE001 — surface package errors honestly
        _fail(job, f"Could not materialize the training package: {exc}")
        return

    try:
        from Backend import train_worker
    except Exception as exc:  # noqa: BLE001
        _fail(job, f"Train worker could not be imported: {exc}")
        return

    original_emit = train_worker._emit

    def _hooked(payload: dict[str, Any]) -> None:
        if isinstance(payload, dict):
            _apply_progress(job_id, payload)
        try:
            original_emit(payload)
        except Exception:
            pass

    train_worker._emit = _hooked  # type: ignore[method-assign]
    try:
        with STORE.lock:
            current = STORE.jobs.get(job_id)
            if current is not None and current.stop_requested:
                mark_job_stopped(current)
                return
        result = train_worker.train_request(request, artifact_dir, command_file)
    except Exception as exc:  # noqa: BLE001
        _fail(job, str(exc) or "Training failed.")
        return
    finally:
        train_worker._emit = original_emit  # type: ignore[method-assign]

    status = str(result.get("status", "") or "")
    with STORE.lock:
        current = STORE.jobs.get(job_id)
        if current is None:
            return
        if current.stop_requested or status == "stopped" or current.status == "stopped":
            current.final_artifact = None
            current.final_artifact_id = ""
            mark_job_stopped(current)
            return
        if status != "success":
            _fail(current, str(result.get("message", "") or "Training failed."))
            return
        artifact_path = Path(str(result.get("artifact_path", "") or ""))
        payload = artifact_path.read_bytes() if artifact_path.is_file() else b""
        if not register_final_artifact(current, payload):
            _fail(current, "Training finished without a real artifact.")
            return
        current.status = "succeeded"
        current.error_code = ""
        current.error_message = ""
        if result.get("device"):
            current.device = str(result.get("device") or current.device)
        current.updated_at = isoformat()
        current.pin_corpus = False
        STORE.persist_hint()


def start_job_thread(job_id: str) -> None:
    """Run ``run_job`` on a daemon thread registered for liveness checks."""
    thread = threading.Thread(
        target=run_job, name=f"train-{job_id}", args=(job_id,), daemon=True
    )
    with STORE.lock:
        STORE.worker_threads[job_id] = thread
    thread.start()
