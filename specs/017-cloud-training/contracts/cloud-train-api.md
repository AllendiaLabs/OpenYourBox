# Contract: Cloud Training HTTP API

**Feature**: `017-cloud-training`  
**Applies to**: Hosted Allendia cloud training API consumed by `CloudTrainClient`

## General

- Base URL: product default or settings override.
- Auth: `Authorization: Bearer <linked_session_token>` on all authenticated endpoints below (except health and link bootstrap as noted).
- Content: JSON unless multipart upload.
- Errors: `{ "error_code": "string", "error_message": "string" }` with suitable HTTP status.
- TLS required in production.

### Error codes (minimum)

| `error_code` | When |
|--------------|------|
| `unauthorized` | Missing/invalid/expired/revoked linked session |
| `insufficient_entitlement` | Linked account lacks active credit/purchase entitlement for a new job |
| `one_job_per_account` | Active job already exists for this platform customer |
| `validation_failed` | Bad manifest / ineligible package |
| `not_found` | Unknown job/corpus/checkpoint |
| `corpus_expired` | `corpus_id` no longer retained |
| `capacity` | Service overloaded |
| `conflict` | Illegal state transition (e.g. pause when not running) |
| `link_pending` / `link_expired` | Account link bootstrap not completed or code expired |

## Endpoints

### `GET /v1/health`

Unauthenticated liveness (optional for plugin).

### Account link bootstrap

#### `POST /v1/auth/link/start`

Unauthenticated or lightly rate-limited. Starts storefront account link.

Response (example shape):

```json
{
  "device_code": "…",
  "user_code": "ABCD-EFGH",
  "verification_url": "https://storefront.example/link",
  "expires_in": 600,
  "interval": 5
}
```

Plugin shows `user_code`, opens `verification_url`, polls token endpoint until linked or expired.

#### `POST /v1/auth/link/token`

Exchange completed device/link flow for `{ "access_token", "refresh_token?", "token_type": "Bearer", "expires_in?" }`.

#### `POST /v1/auth/refresh` (optional)

Refresh linked session when refresh tokens are issued.

#### `POST /v1/auth/logout`

Invalidate current linked session (best-effort); plugin always clears local credentials on Disconnect.

### `GET /v1/entitlement`

Authenticated. Returns whether a new cloud job may be submitted.

```json
{
  "sufficient": true,
  "balance_hint": "optional display string"
}
```

Insufficient → `200` with `sufficient: false` **or** `402`/`403` + `insufficient_entitlement` (pick one consistently in implementation; plugin handles both).

### `GET /v1/jobs`

List jobs for the authenticated platform customer (active first, then recent terminal).

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

**Gates (in order)**:
1. Valid linked session → else `401` + `unauthorized`
2. Sufficient entitlement → else `402`/`403` + `insufficient_entitlement`
3. No active job for account → else `409` + `one_job_per_account`
4. Manifest/corpus validation → else `validation_failed`

On accept: optionally reserve/consume entitlement; store corpus (if uploaded); set `last_used_at` now (create or reuse); enqueue → `202`/`200` with `{ "job_id", "status": "queued", "corpus_id" }`.

### `GET /v1/jobs/{job_id}`

Job detail including latest progress fields and checkpoint summary list. Must belong to authenticated customer.

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

## Storefront obligations (server-side)

- Resolve linked session → platform customer id.
- Provide authoritative entitlement for submit gate (sync or query WordPress commerce/membership as ops defines).
- Account creation and purchase UX remain on the storefront; API does not implement checkout pages for the VST.

## Implementation anchors

- Hosted Allendia cloud API + GPU workers (outside this repo)
- WordPress storefront at `https://store.allendia.com` (account, verification page, credits)
- `OpenYourBox/Source/train/CloudTrainClient.*`
- `Tests/CloudTrainClientTests.cpp`
