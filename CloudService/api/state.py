"""Shared in-memory (optionally file-backed) store for the cloud training API."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "stopped"})
RETENTION_SECONDS = 30 * 24 * 60 * 60


def _now() -> float:
    return time.time()


def isoformat(ts: float | None = None) -> str:
    """Return an ISO-8601 UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else _now()))


def heartbeat_timeout_seconds() -> float:
    """Return the worker-liveness timeout used by the crash reconciler."""
    raw = os.environ.get("CLOUD_HEARTBEAT_TIMEOUT_SECONDS", "120")
    try:
        return max(0.05, float(raw))
    except ValueError:
        return 120.0


@dataclass
class Session:
    """Linked platform-customer session."""

    access_token: str
    customer_id: str
    refresh_token: str = ""
    revoked: bool = False
    expires_at: float = 0.0


@dataclass
class LinkChallenge:
    """Device-code account link in progress."""

    device_code: str
    user_code: str
    expires_at: float
    interval: int = 5
    verified: bool = False
    customer_id: str = ""


@dataclass
class Entitlement:
    """Authoritative (staging) entitlement for one customer."""

    sufficient: bool = True
    balance_hint: str = "mock-credit"


@dataclass
class Corpus:
    """Retained uploaded corpus."""

    corpus_id: str
    customer_id: str
    byte_size: int
    last_used_at: float
    files: dict[str, bytes] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """Published intermediate artifact (real recipe export bytes)."""

    checkpoint_id: str
    step: int
    stage: str
    created_at: str
    payload: bytes = b""


@dataclass
class Job:
    """Cloud training job record.

    ``succeeded`` is legal only when ``final_artifact`` holds non-empty bytes.
    Product status never includes ``paused``.
    """

    job_id: str
    customer_id: str
    status: str
    objective: str
    manifest: dict[str, Any]
    corpus_id: str
    created_at: str
    updated_at: str
    step: int = 0
    total_steps: int = 100
    stage: str = ""
    loss: float = 0.0
    device: str = ""
    error_code: str = ""
    error_message: str = ""
    submitter_client_instance_id: str = ""
    checkpoints: list[Checkpoint] = field(default_factory=list)
    final_artifact: bytes | None = None
    final_artifact_id: str = ""
    pin_corpus: bool = True
    worker_heartbeat_at: float = 0.0
    worker_pid: int | None = None
    command_file: str = ""
    work_dir: str = ""
    stop_requested: bool = False


class Store:
    """Process-wide job, session, corpus, and entitlement store."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.sessions: dict[str, Session] = {}
        self.refresh_tokens: dict[str, str] = {}
        self.links: dict[str, LinkChallenge] = {}
        self.links_by_user: dict[str, str] = {}
        self.jobs: dict[str, Job] = {}
        self.corpora: dict[str, Corpus] = {}
        self.entitlements: dict[str, Entitlement] = {}
        self.signed_downloads: dict[str, tuple[float, bytes, str]] = {}
        self.worker_threads: dict[str, threading.Thread] = {}
        self.default_sufficient = os.environ.get("CLOUD_MOCK_ENTITLEMENT", "1") != "0"
        data_dir = os.environ.get("CLOUD_DATA_DIR", "")
        self.data_dir = Path(data_dir) if data_dir else None

    def reset(self) -> None:
        """Clear all records (tests)."""
        with self.lock:
            self.sessions.clear()
            self.refresh_tokens.clear()
            self.links.clear()
            self.links_by_user.clear()
            self.jobs.clear()
            self.corpora.clear()
            self.entitlements.clear()
            self.signed_downloads.clear()
            self.worker_threads.clear()
            self.default_sufficient = os.environ.get("CLOUD_MOCK_ENTITLEMENT", "1") != "0"
            data_dir = os.environ.get("CLOUD_DATA_DIR", "")
            self.data_dir = Path(data_dir) if data_dir else None

    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def get_entitlement(self, customer_id: str) -> Entitlement:
        with self.lock:
            return self.entitlements.setdefault(
                customer_id,
                Entitlement(sufficient=self.default_sufficient),
            )

    def set_entitlement(
        self, customer_id: str, sufficient: bool, balance_hint: str = "mock-credit"
    ) -> Entitlement:
        with self.lock:
            ent = Entitlement(sufficient=sufficient, balance_hint=balance_hint)
            self.entitlements[customer_id] = ent
            return ent

    def active_job_for(self, customer_id: str) -> Job | None:
        with self.lock:
            for job in self.jobs.values():
                if job.customer_id == customer_id and job.status in ACTIVE_STATUSES:
                    return job
        return None

    def jobs_for(self, customer_id: str) -> list[Job]:
        with self.lock:
            owned = [j for j in self.jobs.values() if j.customer_id == customer_id]
        owned.sort(
            key=lambda j: (0 if j.status in ACTIVE_STATUSES else 1, j.updated_at),
            reverse=True,
        )
        return owned

    def persist_hint(self) -> None:
        """Best-effort file snapshot when CLOUD_DATA_DIR is set."""
        if self.data_dir is None:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "jobs": {
                k: {
                    "status": v.status,
                    "customer_id": v.customer_id,
                    "corpus_id": v.corpus_id,
                }
                for k, v in self.jobs.items()
            },
        }
        (self.data_dir / "jobs-index.json").write_text(
            json.dumps(snapshot), encoding="utf-8"
        )

    def artifact_dir(self, job_id: str) -> Path | None:
        """Return the on-disk artifact directory for a job, if file-backed."""
        if self.data_dir is None:
            return None
        path = self.data_dir / "jobs" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path


STORE = Store()
