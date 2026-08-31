# Research: Real Cloud Training

**Feature**: `018-real-cloud-training`  
**Date**: 2026-08-31

## Decision 1 — Invoke the same recipes as local `train_worker`

**Decision**: `CloudService/worker/train_runner.py` materializes the uploaded job package (manifest + corpus files) into a working directory, builds a `train_steerable` request compatible with local IPC/`train_worker.train_request`, and runs mapping or reconstruction through the **existing** `Backend/train_worker.py` entrypoints (import shared module or subprocess with controlled cwd/artifact dir). Progress events from the recipe are mirrored into the job store for `GET /v1/jobs/{id}`. Checkpoints and final TorchScript artifacts are copied into the cloud artifact store with signed/downloadable ids.

**Rationale**: Spec requires outcome-class parity with Local (FR-001/002/012). Duplicating trainers drifts Gold export and objective behavior.

**Alternatives considered**:
- Cloud-only reimplementation — rejected (parity/maintenance).
- Always shell out to a full ChildProcess clone of the VST local path without shared code — workable fallback if import coupling is painful; prefer shared module first.

## Decision 2 — No mock/fake advancement worker

**Decision**: Remove `CLOUD_MOCK_WORKER` and any in-process fake step/loss/success path. Product and staging always execute real training. Automated tests use: (a) auth/entitlement/one-job fixtures that never claim `succeeded` with a dummy `.pt`, (b) short real trains with tiny corpora and small `total_steps` on CPU, (c) injected failure/Stop cases.

**Rationale**: Clarify session — official Cloud is always real; mock success code must be deleted (FR-007, SC-008).

**Alternatives considered**:
- Keep mock mode behind a flag for CI — rejected by product (risk of shipping fake success; staging/prod confusion).

## Decision 3 — Run + Stop only (local and cloud)

**Decision**: Remove Pause/Resume from Train panel, `TrainCoordinator`, local command file verbs used by UI, and cloud API (`/pause`, `/resume`). Keep `stop` for local worker and `POST /v1/jobs/{id}/stop`. Job status enum drops `paused` from the **active** product surface (legacy `paused` may map to `running` or be rejected if seen). Active job = `queued` | `running`.

**Rationale**: Clarify — matching Local and Cloud control surfaces; Pause deferred.

**Alternatives considered**:
- Soft pause on cloud only — rejected (asymmetry).
- UI-only pause while remote keeps training — rejected (dishonest).

## Decision 4 — Crash / reboot → fail (no auto-resume)

**Decision**: Worker process reports heartbeats (or parent supervisor watches child). If the worker dies or the host restarts, a reconciler marks any job still `queued`/`running` without a live worker as `failed` with a clear recoverable message (e.g. `worker_lost`). Already published checkpoints remain listable/downloadable. Same `job_id` is not auto-resumed; user may Stop (no-op if already failed) or submit a new job.

**Rationale**: Clarify B; production-safe simplicity for first real slice (FR-011a).

**Alternatives considered**:
- Auto-resume from last checkpoint — deferred (complexity, entitlement edge cases).
- Leave status `running` until operator intervention — rejected (zombie jobs block one-job-per-account).

## Decision 5 — Staging Done, production-safe defaults

**Decision**: Done = plugin pointed at staging (`apiBaseUrlOverride` / storefront override in `cloud.xml`) completes real Cloud → Gold. Defaults, TLS assumptions, Bearer-only auth, signed downloads, entitlement gates, retention, and one-job rules remain as in `017` so a later public host is not blocked. Public DNS go-live is out of Done.

**Rationale**: Clarify A + C guardrails (FR-008/008a, SC-007).

**Alternatives considered**:
- Require production host in Done — rejected (blocks trying real training now).
- Staging-only insecure shortcuts (skip entitlement) — rejected (compromises production).

## Decision 6 — CPU fallback is real training

**Decision**: Prefer CUDA/MPS when available; fall back to CPU. Device choice is reported in progress metadata when feasible (same spirit as local). Never treat missing GPU as permission to fabricate success.

**Rationale**: Spec FR-013; enables staging on laptops.

**Alternatives considered**:
- Refuse jobs without GPU — rejected for staging Done bar.

## Decision 7 — Inherit 017 package and client contracts

**Decision**: Job package remains `contracts` from `017` (`cloud-job-package.md`). Plugin packaging/`CloudTrainClient` submit/poll/download paths stay; this feature completes the server worker and strips pause/resume. Soft 2 GiB warn, corpus retention, submitter-only auto-load unchanged.

**Rationale**: Spec assumes `017` baseline; avoid redesigning Train UX beyond Run/Stop.

**Alternatives considered**:
- New package schema — unnecessary churn.

## Decision 8 — Stop semantics for real worker

**Decision**: `POST .../stop` sets a stop flag the runner honors (same as local `command_file` stop): finish current step if needed, exit with terminal `stopped`, **no** final success artifact auto-publish. Intermediate checkpoints already published remain available.

**Rationale**: FR-006; aligns with existing `train_worker` stop handling.

**Alternatives considered**:
- Kill -9 immediately — risk corruption; prefer cooperative stop first, escalate only if hung beyond a timeout then `failed`/`stopped` with message.
