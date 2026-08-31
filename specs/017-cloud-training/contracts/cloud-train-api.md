# Contract: Cloud Training HTTP API

**Feature**: `017-cloud-training`  
**Applies to**: Proprietary `CloudService/` API consumed by `CloudTrainClient`

## General

- Base URL: product default or settings override.
- Auth: `Authorization: Bearer <api_token>` on all endpoints below.
- Content: JSON unless multipart upload.
- Errors: `{ "error_code": "string", "error_message": "string" }` with suitable HTTP status.
- TLS required in production.

### Error codes (minimum)

| `error_code` | When |
|--------------|------|
| `unauthorized` | Missing/invalid/revoked token |
| `one_job_per_token` | Active job already exists for token |
| `validation_failed` | Bad manifest / ineligible package |
| `not_found` | Unknown job/corpus/checkpoint |
| `corpus_expired` | `corpus_id` no longer retained |
| `capacity` | Service overloaded |
| `conflict` | Illegal state transition (e.g. pause when not running) |

## Endpoints

### `GET /v1/health`

Unauthenticated liveness (optional for plugin).

### `GET /v1/jobs`

List jobs for the token (active first, then recent terminal).

Response:

```json
{
  "jobs": [
    {
      "job_id": "…",
      "status": "running",
      "objective": "reconstruction",
      "step": 1200,
      "total_steps": 2000000,
      "stage": "representation",
      "loss": 0.42,
      "corpus_id": "…",
      "created_at": "ISO-8601",
      "updated_at": "ISO-8601"
    }
  ]
}
```

### `POST /v1/jobs`

Create job. Either multipart (manifest JSON + files) **or** JSON body with `corpus_id` for reuse.

**Concurrency**: If token has an active job → `409` + `one_job_per_token`.

On accept: store corpus (if uploaded), set `last_used_at` now (create or reuse), enqueue → `202`/`200` with `{ "job_id", "status": "queued", "corpus_id" }`.

### `GET /v1/jobs/{job_id}`

Job detail including latest progress fields and checkpoint summary list.

### `POST /v1/jobs/{job_id}/pause` | `/resume` | `/stop`

Control verbs. `stop` → terminal `stopped` (not success). Illegal transition → `conflict`.

### `GET /v1/jobs/{job_id}/checkpoints`

List checkpoints with ids, step, stage, created_at.

### `GET /v1/jobs/{job_id}/checkpoints/{checkpoint_id}/download`

Returns redirect or JSON `{ "url": "<signed>", "expires_at": "…" }` for artifact bytes.

### `GET /v1/jobs/{job_id}/artifact/download`

Final artifact when `status == succeeded`; otherwise `conflict`/`not_found`.

### Retention

- Corpus: 30 days from `last_used_at`; reuse via `corpus_id` on `POST /v1/jobs` resets clock.
- Sweeper deletes expired corpora not pinned by an active job.
- Expired reference → `corpus_expired`.

## Worker obligations (server-side)

- Materialize package; run mapping/reconstruction recipe compatible with local `train_worker`.
- Emit progress updates readable via `GET /v1/jobs/{job_id}` (step, loss, stage).
- Publish checkpoints periodically; write final artifact on success only.
- Never treat `stopped` as success.

## Implementation anchors

- `CloudService/api/`
- `CloudService/worker/`
- `OpenYourBox/Source/train/CloudTrainClient.*`
- `Tests/test_cloud_api.py`
