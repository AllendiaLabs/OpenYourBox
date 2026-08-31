# Research: Cloud Training

**Feature**: `017-cloud-training`  
**Date**: 2026-08-31

## Decision 1 — Control plane: HTTPS REST job API (not WebSocket-only)

**Decision**: Plugin ↔ cloud uses a versioned HTTPS JSON API for account/session validation, entitlement probe, submit, status poll, control (pause/resume/stop), checkpoint listing, and artifact download URLs. Status is polled on a background/message-thread timer (~1–2 s) to meet SC-003 (≥1 update / 5 s). Optional future push (SSE/WebSocket) is out of scope for this slice.

**Rationale**: Matches existing TrainCoordinator progress model (poll/stream → UI snapshot). Polling is simpler through DAW process networking, proxies, and firewalls than long-lived sockets in a VST.

**Alternatives considered**:
- WebSocket-only — richer live updates; harder to debug and more fragile in hosts.
- gRPC — stronger typing; worse fit for casual ops and JUCE client ergonomics.

## Decision 2 — Reuse local train recipes on the GPU worker

**Decision**: Cloud GPU workers execute the **same** training recipes as `Backend/train_worker.py` (mapping + reconstruction), driven by a job package whose graph/options mirror the local `train_steerable` start payload. Packaging may ship corpus files separately from the JSON request; the worker materializes local paths then runs the existing recipe entrypoints (shared module or vendored worker).

**Rationale**: Spec requires objective parity with local Train. Duplicating recipes would drift and break Gold export assumptions.

**Alternatives considered**:
- Separate cloud-only trainer — rejected (parity + maintenance).
- Remote SSH into user machine — rejected (not cloud GPU; constitution Phase 4 is remote optional compute).

## Decision 3 — Plugin architecture: CloudTrainClient + destination fork in TrainCoordinator

**Decision**:
- Add `CloudTrainClient` (C++): HTTPS calls off the audio thread; owns poll timer / async I/O.
- Extend `TrainCoordinator` (or a thin facade used by `TrainPanel`) with `destination: local | cloud`. Local path unchanged (ChildProcess → `train_worker.py`). Cloud path: package → upload → poll → download; map remote states onto existing `TrainStatus` (+ `queued` surfaced in status message / progress).
- Persist **linked account session** via JUCE `PropertiesFile` / application settings (masked in UI); never log raw secrets.
- Persist **submitter identity** locally: `(job_id, is_submitter=true)` so only the submitting instance auto-loads on success (FR-010 / FR-010a). Non-submitters discover jobs via `GET /v1/jobs` by account and may download/load manually.
- Persist product **storefront URL** (default + optional override) for “open storefront” affordances (FR-002a).

**Rationale**: Keeps one Train panel UX; isolates network from audio; preserves local Train with no account (FR-017).

**Alternatives considered**:
- Spawn a second “cloud_worker.py” ChildProcess for HTTP — possible later; C++ client is enough and avoids another IPC layer for control.
- Auto-load on every account-bearing instance — rejected by clarify (submitter-only).

## Decision 4 — Auth: platform customer account link (WordPress storefront)

**Decision**: Official cloud uses a **platform customer account** on the WordPress storefront—not a standalone developer API token product path.

Link flow (plugin):
1. User starts **Link account** in settings.
2. Plugin opens the storefront authorization page (system browser) using a **device-code or equivalent short-lived link code** shown in the VST, *or* a documented browser OAuth/PKCE flow suitable for desktop plugins.
3. On success, cloud API issues a **linked session** (access credential + refresh if applicable) bound to the storefront customer id.
4. Plugin saves credentials masked; status: Not linked | Linked | Auth error.
5. **Disconnect** clears local credentials and disables Cloud Run until re-linked.

All cloud endpoints require `Authorization: Bearer <linked_session_token>`. Invalid/expired/revoked → `unauthorized`; plugin prompts re-link.

**Rationale**: Matches constitution Phase 4 and updated spec (FR-001, FR-005a). Device-code-style link fits DAW hosts where custom URL schemes are unreliable.

**Alternatives considered**:
- Out-of-band beta API tokens without customer account — superseded by platform-account requirement.
- Username/password typed into the VST — rejected (credential phishing risk; storefront owns identity).
- In-plugin WordPress cookies — rejected (fragile; not VST-appropriate).

## Decision 5 — Entitlement gate at submit

**Decision**: Before accepting `POST /v1/jobs`, the cloud API MUST verify **sufficient active credit or purchase entitlement** for the requesting customer (server-side sync with the storefront ledger). Plugin MUST:
- Probe entitlement via `GET /v1/entitlement` (or equivalent embedded in session status) for UI hints.
- Refuse Cloud Run locally when unlinked or when last known entitlement is insufficient, with distinct messages (auth vs entitlement) (FR-005a, SC-008, SC-012).
- Offer **Open storefront** so users can obtain entitlement outside the VST (FR-002a). No in-plugin checkout.

On accept, service MAY reserve/consume entitlement per ops pricing rules (exact units are ops-owned; contract only requires a clear `insufficient_entitlement` refusal). Already-accepted jobs continue under accept-time rules unless ops documents otherwise (spec edge case).

**Rationale**: Spec US2/US5 + constitution: purchases/balance on storefront; VST consumes entitlement at submit.

**Alternatives considered**:
- Free unlimited cloud with only account login — rejected by product (paid hosted service).
- Decrement entitlement only in the plugin — rejected (must be authoritative server-side).

## Decision 6 — Concurrency & ownership

**Decision**: Service enforces **one active job per platform customer account** (`queued`|`running`|`paused` count as active). Plugin also keeps one active train job per instance (local or cloud). Account-wide list/monitor/control/download. Soft refuse with explicit error code `one_job_per_account`.

**Rationale**: Spec FR-013a; limits concurrent GPU spend per customer.

## Decision 7 — Corpus retention & reuse

**Decision**: Store uploaded corpus under a `corpus_id` tied to customer/job. Retention = **30 days from `last_used_at`**. Qualifying reuse (at minimum: submit a new job that references `corpus_id` instead of re-uploading) resets `last_used_at`. Idle expiry deletes corpus blobs. Checkpoints/final artifacts follow the same retention clock unless still needed for an active job (active job pins artifacts until terminal + grace aligned with corpus policy).

**Rationale**: Clarify retention + extend-on-use; enables re-run without full re-upload when still retained.

## Decision 8 — Soft upload size warning

**Decision**: Soft threshold default **2 GiB** total selected corpus bytes (sum of selected library files to upload). Warn in Train panel; allow continue. Exact constant is a named setting (`cloudSoftUploadWarnBytes`) adjustable in ops without changing product rules. No hard product cap this slice.

**Rationale**: Soft warning signal without blocking large RAVE corpora.

**Alternatives considered**: 500 MiB — too aggressive; 10 GiB — weak warning for beta disks.

## Decision 9 — Cloud service layout (proprietary)

**Decision**: Introduce a proprietary **Cloud Training Service** repo-area with:
- **API**: customer session auth, entitlement probe/consume, job CRUD, corpus upload, artifact signed download URLs, retention sweeper.
- **Worker**: GPU host pulling jobs, running train recipes, writing checkpoints to object storage.
- **Storefront integration**: server-side WordPress customer identity + entitlement sync (not shipped as user-facing PHP inside the VST).
- Constitution: Cloud Training Backend is **Proprietary**; VST remains the only end-user training UI.

For local/dev: API + fake/CPU worker + **mock entitlement** mode for contract tests without a live storefront.

**Rationale**: Phase 4 constitution; official hosted path with account entitlement.

**Alternatives considered**: Fully managed third-party train SaaS — rejected (need custom graph package + Gold `.pt` export parity + storefront entitlement).

## Decision 10 — Packaging format

**Decision**: Job package =
1. `manifest.json` — graph fragment, armed ids, train_options (same fields as local start), objective, submitter client metadata (`client_instance_id`, plugin version).
2. Corpus archive or multipart file upload (wavs as selected); or `corpus_id` reference when retained.
3. Server assigns `job_id`, stores corpus, enqueues worker (after auth + entitlement + one-job checks).

Progress events mirror local fields where possible: `status`, `step`, `loss`, `stage` (reconstruction), `checkpoint` entries, `error_code` / `error_message`.

**Rationale**: Minimize TrainPanel adapter surface.

## Decision 11 — Endpoint configuration

**Decision**: Default cloud API base URL and default storefront URL baked for production; optional overrides in settings (advanced) for staging/dev.

**Rationale**: Ops flexibility without a second product surface.

## Decision 12 — Security / privacy notes

**Decision**: TLS required; session credentials never in query strings; corpus not public; downloads via time-limited signed URLs. Copyright acknowledgment remains local gate only (not uploaded as legal proof). Entitlement and identity authoritative on server/storefront; plugin cache is advisory for UX only.

**Rationale**: Spec assumptions + constitution copyright shield (local log).

## Unresolved → resolved for plan

| Former unknown | Resolution |
|----------------|------------|
| Hosting vendor | Self-owned API + GPU worker (vendor-agnostic object storage) |
| Soft warning threshold | **2 GiB** default |
| Transport | HTTPS JSON + poll |
| Auto-load scope | Submitter instance only (local `is_submitter` flag) |
| Auth model | Platform customer linked session (storefront); not standalone beta tokens |
| Billing UI | Storefront only; plugin open-URL affordance |
| Concurrency key | Platform customer account |
