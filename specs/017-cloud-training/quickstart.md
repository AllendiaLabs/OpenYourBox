# Quickstart: Cloud Training

**Feature**: `017-cloud-training`  
**Purpose**: Validate end-to-end cloud train behavior against the spec (manual + automated where noted).

## Prerequisites

- Built OpenYourBox VST/AU with Train / Library working **locally** (004/005 baseline).
- Cloud API reachable (staging or local `CloudService` with mock/CPU worker).
- Valid beta API token issued out-of-band.
- Contracts: `contracts/cloud-train-api.md`, `cloud-train-plugin-ux.md`, `cloud-job-package.md`.

## Setup

1. Launch DAW → load OpenYourBox.  
2. Open plugin settings → enter API token → save (masked).  
3. Prepare a minimal eligible Train setup (armed graph, selected library entries, copyright acknowledged).  
4. Confirm **Local** Run still works once without a cloud job (SC-010).

## Scenarios

### 1 — Configure token (US1)

1. Clear token → Cloud Run refused with configure prompt.  
2. Save valid token → status shows configured.  
3. Use invalid token → cloud action shows auth error; no job created.

**Expect**: FR-001, FR-002, FR-015.

### 2 — Submit cloud job (US2)

1. Destination **Cloud**, objective mapping or reconstruction as eligible.  
2. Run → upload progress visible → job `queued`/`running`.  
3. Audio keeps playing on pre-job model.

**Expect**: SC-001 (setup ready), SC-002, FR-005–FR-006.

### 3 — Soft size warning (FR-016a / SC-013)

1. Select corpus totaling &gt; 2 GiB (or temporarily lower threshold in debug).  
2. Run Cloud → warning shown → proceed still allowed.

### 4 — One job per token (FR-013a / SC-012)

1. Start cloud job with token T.  
2. Second instance (or machine) with T → Cloud Run → clear `one_job_per_token` (or equivalent) message.

### 5 — Monitor / pause / stop (US3)

1. Observe status/loss updates while online (≥1 / 5 s average when running).  
2. Pause → Resume.  
3. Stop → terminal `stopped`; **no** Gold auto-load.

**Expect**: SC-003, SC-004.

### 6 — Rediscovery & token-wide control (US3 / SC-005a)

1. Submit on machine A.  
2. Open plugin on machine B with same token → list/attach job → pause or status visible within 1 minute.  
3. Restart plugin on A → reattach without resubmit (SC-005).

### 7 — Checkpoints & submitter auto-load (US4)

1. During run, download a checkpoint → optional load; job still running (SC-009).  
2. On success: **A (submitter)** auto-downloads and Gold auto-loads (SC-006).  
3. On **B (non-submitter)**: success visible; no auto graph swap; manual download/load works (SC-006a).

### 8 — Retention reuse (SC-011)

1. After success, note `corpus_id`.  
2. Within 30 days, submit new job referencing retained corpus (no full re-upload) → accepted; retention extended.  
3. Simulate expiry (test hook or clock) → reuse fails with `corpus_expired`; fresh upload required.

### 9 — No credits UI (US5)

Explore settings + Train → no WordPress / add-credits path (FR-014, SC-007).

### 10 — Offline monitoring

Disconnect network during run → UI shows offline/reconnect; reconnect resumes status without implying cancel.

## Automated smoke (as implemented)

```text
# API / retention / concurrency (from repo root, once CloudService tests exist)
pytest CloudService/tests -q
# or
pytest Tests/test_cloud_api.py -q

# Plugin-side client shaping / soft-warn (CTest target name TBD in tasks)
ctest -R CloudTrain -j
```

## Pass criteria

All scenarios above match acceptance in `spec.md`; local Train unaffected; constitution audio-thread rules held (no glitches attributable to cloud I/O).
