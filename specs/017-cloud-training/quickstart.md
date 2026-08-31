# Quickstart: Cloud Training

**Feature**: `017-cloud-training`  
**Purpose**: Validate end-to-end cloud train behavior against the spec (manual + automated where noted).

## Prerequisites

- Built OpenYourBox VST/AU with Train / Library working **locally** (004/005 baseline).
- Hosted Allendia cloud training API reachable at the product default
  (`https://cloud.openyourbox.allendia.com`, or a staging override in `cloud.xml`).
- Allendia storefront (`https://store.allendia.com`) for account sign-in, verification, and credits.
- Contracts: `contracts/cloud-train-api.md`, `cloud-train-plugin-ux.md`, `cloud-job-package.md`.

## Setup

1. Launch DAW → load OpenYourBox.  
2. Open Train → **Sign in with Allendia** → complete verification in the browser → confirm Signed in.  
3. Ensure credits are available (**Manage account** / storefront purchase).  
4. Prepare a minimal eligible Train setup (armed graph, selected library entries, copyright acknowledged).  
5. Confirm **Local** Run still works once **without** signing in (SC-010) — Sign out first if needed.

## Scenarios

### 1 — Link platform account (US1)

1. Disconnect → Cloud Run refused with link-account prompt.  
2. Link valid account → status shows Linked.  
3. Use revoked/expired session → cloud action shows auth error; no job created.  
4. Open storefront affordance launches storefront URL outside the VST.

**Expect**: FR-001, FR-002, FR-002a, FR-015.

### 2 — Entitlement gate (US2 / US5 / SC-012)

1. Linked account with `sufficient: false` → Cloud Run refused with entitlement message; Open storefront available.  
2. After entitlement becomes sufficient → Cloud Run accepted.

**Expect**: FR-005a, SC-008, SC-012.

### 3 — Submit cloud job (US2)

1. Destination **Cloud**, objective mapping or reconstruction as eligible.  
2. Run → upload progress visible → job `queued`/`running`.  
3. Audio keeps playing on pre-job model.

**Expect**: SC-001 (setup ready; purchase time excluded), SC-002, FR-005–FR-006.

### 4 — Soft size warning (FR-016a / SC-013)

1. Select corpus totaling &gt; 2 GiB (or temporarily lower threshold in debug).  
2. Run Cloud → warning shown → proceed still allowed.

### 5 — One job per account (FR-013a)

1. Start cloud job under customer C.  
2. Second instance (or machine) linked as C → Cloud Run → clear `one_job_per_account` message.

### 6 — Monitor / pause / stop (US3)

1. Observe status/loss updates while online (≥1 / 5 s average when running).  
2. Pause → Resume.  
3. Stop → terminal `stopped`; **no** Gold auto-load.

**Expect**: SC-003, SC-004.

### 7 — Rediscovery & account-wide control (US3 / SC-005a)

1. Submit on machine A.  
2. Open plugin on machine B with same linked account → list/attach job → pause or status visible within 1 minute.  
3. Restart plugin on A → reattach without resubmit (SC-005).

### 8 — Checkpoints & submitter auto-load (US4)

1. During run, download a checkpoint → optional load; job still running (SC-009).  
2. On success: **A (submitter)** auto-downloads and Gold auto-loads (SC-006).  
3. On **B (non-submitter)**: success visible; no auto graph swap; manual download/load works (SC-006a).

### 9 — Retention reuse (SC-011)

1. After success, note `corpus_id`.  
2. Within 30 days, submit new job referencing retained corpus (no full re-upload) → accepted; retention extended.  
3. Simulate expiry (test hook or clock) → reuse fails with `corpus_expired`; fresh upload required.

### 10 — Local without account (SC-010)

Disconnect account → destination Local → complete a local Train success path with no cloud dependency.

### 11 — Offline monitoring

Disconnect network during run → UI shows offline/reconnect; reconnect resumes status without implying cancel.

## Automated smoke (as implemented)

```text
# Plugin-side client shaping / soft-warn / gate messages
ctest -R CloudTrain -j
```

## Pass criteria

All scenarios above match acceptance in `spec.md`; local Train unaffected without account; constitution audio-thread rules held (no glitches attributable to cloud I/O).
