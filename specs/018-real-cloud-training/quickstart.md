# Quickstart: Real Cloud Training

**Feature**: `018-real-cloud-training`  
**Purpose**: Validate real Cloud training end-to-end on staging (Done bar). Public production host not required.

## Prerequisites

- Built OpenYourBox with Train / Library / Cloud client from `017` baseline.
- Python 3.10+ with `CloudService/requirements.txt` and local train deps (same PyTorch stack as `Backend/train_worker.py`).
- Contracts: `contracts/cloud-train-api.md`, `cloud-worker.md`, `cloud-train-plugin-ux.md`; package shape from `017` `cloud-job-package.md`.

## Setup (staging)

1. Install and run the API + real worker (no mock advancement):

```bash
pip install -r CloudService/requirements.txt
# plus existing Backend/train dependency install used for local Train
export CLOUD_API_PUBLIC_URL=http://127.0.0.1:8787
export CLOUD_DATA_DIR=/tmp/oyb-cloud-data
PYTHONPATH=. python -m uvicorn CloudService.api.app:app --host 127.0.0.1 --port 8787
```

2. Point the plugin at staging via user-data `cloud.xml`:

- `apiBaseUrlOverride` = `http://127.0.0.1:8787`
- `storefrontUrlOverride` = same origin if using staging link page

3. Launch DAW → OpenYourBox → complete Allendia link + ensure entitlement sufficient (staging link flow).
4. Confirm Train shows **Run** and **Stop** only (no Pause/Resume) for Local and Cloud.

## Scenarios

### 1 — Real mapping success (US1 / SC-001 / SC-003)

1. Destination **Cloud**, objective mapping, minimal eligible pairs, small `total_steps`.
2. Run → upload → `queued`/`running` with changing step/loss.
3. On success: submitting instance Gold auto-loads; hear trained chain (not silence/dummy).

**Expect**: Real artifact; no placeholder success.

### 2 — Real reconstruction success (US1 / SC-002)

Same as (1) with reconstruction objective and eligible library selection.

### 3 — Checkpoint mid-run (US2 / SC-005)

1. Enable checkpoints / short interval; Run Cloud.
2. When a checkpoint appears, download (optional load); job still running.
3. Let job finish or Stop; Stop ⇒ no success auto-load.

### 4 — Stop (FR-006 / SC-006)

1. Start Cloud job; Stop while running.
2. Status `stopped`; no Gold auto-load; prior checkpoints may still download.

### 5 — Worker lost (FR-011a / SC-006)

1. Start Cloud job; kill the train worker process (or stop host mid-run) without clean Stop.
2. Job becomes `failed` with recoverable message (e.g. worker lost); not stuck `running`.
3. Published checkpoints remain downloadable if any.

### 6 — No fake-success path (FR-007 / SC-008)

1. Grep/config review: no `CLOUD_MOCK_WORKER` fake advancement enabling `succeeded` with empty artifact.
2. Automated tests pass without a product mock-success worker.

### 7 — Local ungated (SC-010 / FR-010)

1. Sign out / unlink; Destination **Local**; complete a short local train success.
2. Confirm Run/Stop only locally as well.

### 8 — One job per account (US4)

1. Active Cloud job under account C.
2. Second submit as C → `one_job_per_account` refusal.

### 9 — Production-safe staging review (SC-007 / FR-008a)

Checklist: TLS assumptions documented for prod; Bearer not in query strings; entitlement still gated; retention still 30-day sliding; no staging skip that would be unsafe on a public host.

## Automated smoke

```text
ctest -R CloudTrain -j
PYTHONPATH=. pytest CloudService/tests -q
```

Prefer short real-train tests over fake advancement.

## Pass criteria

Scenarios 1–9 match `spec.md` acceptance; Local unaffected without account; audio thread rules held; mock fake-success worker absent.
