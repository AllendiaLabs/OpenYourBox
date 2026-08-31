# Contract: Cloud Training HTTP API (Real Training Deltas)

**Feature**: `018-real-cloud-training`  
**Supersedes**: Control and worker sections of `specs/017-cloud-training/contracts/cloud-train-api.md` where they conflict  
**Baseline**: Auth, entitlement, submit, list/get job, checkpoints, artifact download, retention from `017` remain unless noted

## General

Same as `017`: Bearer linked session, JSON/multipart, TLS in production, standard `error_code` envelope.

### Error codes (additions / emphasis)

| `error_code` | When |
|--------------|------|
| `worker_lost` | Job failed because worker/host died mid-run (may appear in job `error_code`) |
| `conflict` | Illegal control (e.g. stop when already terminal); **not** used for pause/resume (those endpoints are removed) |

`paused` is not a valid product job status in responses for new jobs.

## Control endpoints

### Removed

- `POST /v1/jobs/{job_id}/pause`
- `POST /v1/jobs/{job_id}/resume`

Clients MUST NOT call these. Servers MUST NOT implement product pause/resume (return `404` or omit routes).

### `POST /v1/jobs/{job_id}/stop`

- Allowed from `queued` or `running`.
- Result: terminal `stopped`; **no** final success artifact.
- Already published checkpoints remain listable/downloadable.
- Illegal if already terminal → `409` + `conflict`.

## Job status payload

`GET /v1/jobs/{job_id}` and list items:

```json
{
  "job_id": "…",
  "status": "running",
  "objective": "mapping",
  "step": 120,
  "total_steps": 500,
  "stage": "",
  "loss": 0.42,
  "device": "cpu",
  "corpus_id": "…",
  "error_code": null,
  "error_message": null,
  "updated_at": "ISO-8601"
}
```

`status` ∈ `queued` | `running` | `succeeded` | `failed` | `stopped`.

**Honesty rule**: `status == "succeeded"` ONLY if a real final artifact is available via artifact download.

## Worker / service obligations (API-facing)

- Accepting a job eventually runs **real** training (see `cloud-worker.md`).
- No mock advancement that fabricates step/loss/`succeeded` without training.
- Reconcile lost workers → `failed` + `worker_lost` (or equivalent message); clear one-job slot.
- Progress fields reflect actual runner events (≥ poll granularity for SC-004).

## Active job definition

Active = `queued` | `running` (not `paused`). One active job per platform customer account unchanged.

## Implementation anchors

- `CloudService/api/jobs.py`, `artifacts.py`, `app.py`, `state.py`
- `OpenYourBox/Source/train/CloudTrainClient.*` (stop only)
- Tests must not depend on pause/resume or fake-success workers
