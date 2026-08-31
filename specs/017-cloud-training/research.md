# Research: Cloud Training

**Feature**: `017-cloud-training`  
**Date**: 2026-08-31

## Decision 1 — Control plane: HTTPS REST job API (not WebSocket-only)

**Decision**: Plugin ↔ cloud uses a versioned HTTPS JSON API for submit, status poll, control (pause/resume/stop), checkpoint listing, and artifact download URLs. Status is polled on a background/message-thread timer (~1–2 s) to meet SC-003 (≥1 update / 5 s). Optional future push (SSE/WebSocket) is out of scope for this slice.

**Rationale**: Matches existing TrainCoordinator progress model (poll/stream → UI snapshot). Polling is simpler through DAW process networking, proxies, and firewalls than long-lived sockets in a VST.

**Alternatives considered**:
- WebSocket-only — richer live updates; harder to debug and more fragile in hosts.
- gRPC — stronger typing; worse fit for casual beta ops and JUCE client ergonomics.

## Decision 2 — Reuse local train recipes on the GPU worker

**Decision**: Cloud GPU workers execute the **same** training recipes as `Backend/train_worker.py` (mapping + reconstruction), driven by a job package whose graph/options mirror the local `train_steerable` start payload. Packaging may ship corpus files separately from the JSON request; the worker materializes local paths then runs the existing recipe entrypoints (shared module or vendored worker).

**Rationale**: Spec requires objective parity with local Train. Duplicating recipes would drift and break Gold export assumptions.

**Alternatives considered**:
- Separate cloud-only trainer — rejected (parity + maintenance).
- Remote SSH into user machine — rejected (not cloud GPU; constitution Phase 4 is remote paid/optional compute).

## Decision 3 — Plugin architecture: CloudTrainClient + destination fork in TrainCoordinator

**Decision**:
- Add `CloudTrainClient` (C++): HTTPS calls off the audio thread; owns poll timer / async I/O.
- Extend `TrainCoordinator` (or a thin facade used by `TrainPanel`) with `destination: local | cloud`. Local path unchanged (ChildProcess → `train_worker.py`). Cloud path: package → upload → poll → download; map remote states onto existing `TrainStatus` (+ `queued` surfaced in status message / progress).
- Persist API token via JUCE `PropertiesFile` / application settings (masked in UI); never log the raw token.
- Persist **submitter identity** locally: `(job_id, is_submitter=true)` so only the submitting instance auto-loads on success (FR-010 / FR-010a). Non-submitters discover jobs via `GET /jobs` by token and may download/load manually.

**Rationale**: Keeps one Train panel UX; isolates network from audio; preserves local Train with no token (FR-017).

**Alternatives considered**:
- Spawn a second “cloud_worker.py” ChildProcess for HTTP — possible later; C++ client is enough for beta and avoids another IPC layer for control.
- Auto-load on every token-bearing instance — rejected by clarify (submitter-only).

## Decision 4 — Auth: Bearer API token (beta)

**Decision**: `Authorization: Bearer <token>` on all cloud endpoints. Tokens issued out-of-band. Plugin settings: save / clear / replace; masked display. Invalid/revoked → clear auth error, no job start.

**Rationale**: Matches specify/clarify; WordPress/OAuth deferred.

**Alternatives considered**: Device code / OAuth — deferred with WordPress accounts.

## Decision 5 — Concurrency & ownership

**Decision**: Service enforces **one active job per token** (`queued`|`running`|`paused` count as active). Plugin also keeps one active train job per instance (local or cloud). Token-wide list/monitor/control/download. Soft refuse with explicit error code `one_job_per_token`.

**Rationale**: Clarify answers; limits beta GPU spend.

## Decision 6 — Corpus retention & reuse

**Decision**: Store uploaded corpus under a `corpus_id` tied to token/job. Retention = **30 days from `last_used_at`**. Qualifying reuse (at minimum: submit a new job that references `corpus_id` instead of re-uploading) resets `last_used_at`. Idle expiry deletes corpus blobs. Checkpoints/final artifacts follow the same retention clock unless still needed for an active job (active job pins artifacts until terminal + grace aligned with corpus policy).

**Rationale**: Clarify retention + extend-on-use; enables re-run without full re-upload when still retained.

## Decision 7 — Soft upload size warning

**Decision**: Soft threshold default **2 GiB** total selected corpus bytes (sum of selected library files to upload). Warn in Train panel; allow continue. Exact constant is a named setting (`cloudSoftUploadWarnBytes`) adjustable in ops without changing product rules. No hard product cap this slice.

**Rationale**: Clarify chose soft warning; 2 GiB is a practical home-broadband / beta-storage signal without blocking large RAVE corpora.

**Alternatives considered**: 500 MiB — too aggressive for reconstruction corpora; 10 GiB — weak warning for beta disks.

## Decision 8 — Cloud service layout (proprietary)

**Decision**: Introduce a proprietary **Cloud Training Service** repo-area (or sibling tree) with:
- **API**: job CRUD, auth, corpus upload, artifact signed download URLs, retention sweeper.
- **Worker**: GPU host pulling jobs, running train recipes, writing checkpoints to object storage.
- Constitution: Cloud Training Backend is **Proprietary**; VST remains the only end-user UI (no standalone train app for users).

For local/dev: API + fake/CPU worker mode for contract tests without a GPU.

**Rationale**: Phase 4 constitution; enables end-to-end beta before WordPress credits.

**Alternatives considered**: Fully managed third-party train SaaS — rejected (need custom graph package + Gold `.pt` export parity).

## Decision 9 — Packaging format

**Decision**: Job package =
1. `manifest.json` — graph fragment, armed ids, train_options (same fields as local start), objective, submitter client metadata (`client_instance_id`, plugin version).
2. Corpus archive or multipart file upload (wavs as selected); or `corpus_id` reference when retained.
3. Server assigns `job_id`, stores corpus, enqueues worker.

Progress events mirror local fields where possible: `status`, `step`, `loss`, `stage` (reconstruction), `checkpoint` entries, `error_code` / `error_message`.

**Rationale**: Minimize TrainPanel adapter surface.

## Decision 10 — Endpoint configuration

**Decision**: Default cloud base URL baked for production beta; optional override in settings (advanced) for staging/dev. Not required for MVP UX but needed for testing.

**Rationale**: Ops flexibility without a second product surface.

## Decision 11 — Security / privacy notes (non-WordPress)

**Decision**: TLS required; token never in query strings; corpus not public; downloads via time-limited signed URLs. Copyright acknowledgment remains local gate only (not uploaded as legal proof). No credit ledger.

**Rationale**: Spec assumptions + constitution copyright shield (local log).

## Unresolved → resolved for plan

| Former unknown | Resolution |
|----------------|------------|
| Hosting vendor | Self-owned API + GPU worker (vendor-agnostic object storage); not locked to one cloud brand in this plan |
| Soft warning threshold | **2 GiB** default |
| Transport | HTTPS JSON + poll |
| Auto-load scope | Submitter instance only (local `is_submitter` flag) |
