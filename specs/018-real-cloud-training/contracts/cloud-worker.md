# Contract: Cloud Train Worker (Real Recipes)

**Feature**: `018-real-cloud-training`  
**Applies to**: `CloudService/worker/train_runner.py` (+ process supervision)  
**Package input**: `specs/017-cloud-training/contracts/cloud-job-package.md`

## Purpose

Define how an accepted cloud job becomes real progress, checkpoints, and a Gold-eligible final artifact—same objective families as local Train.

## Lifecycle

1. **Claim** queued job (one runner at a time per deployment for this slice is fine).
2. **Materialize** corpus + manifest into an isolated work directory.
3. **Validate** objective and package; on hard failure → job `failed` with readable reason (no dummy success).
4. **Select device** (accelerator if available, else CPU); record on job progress.
5. **Invoke** mapping or reconstruction via shared `Backend/train_worker` recipes (`train_steerable` / `train_request` semantics).
6. **Heartbeat** while running; publish step/loss/stage to job store as recipe emits events.
7. **Checkpoints**: when `export_checkpoints` / interval request it, publish downloadable checkpoint artifacts (real exports).
8. **Stop**: honor cooperative stop → terminal `stopped`; do not publish final success artifact.
9. **Success**: export final TorchScript (or equivalent local-success artifact), register as final artifact, set `succeeded`.
10. **Failure**: OOM, recipe exception, validation → `failed` with message; no success artifact.
11. **Crash**: parent detects dead child / stale heartbeat → `failed` + `worker_lost`; keep prior checkpoints.

## Forbidden

- Fabricating monotonic fake loss curves or `succeeded` without recipe completion.
- Auto-resuming a `worker_lost` job under the same `job_id`.
- Treating `stopped` as `succeeded`.

## Progress event mapping (minimum)

| Recipe signal | Job fields |
|---------------|------------|
| step / total_steps | `step`, `total_steps` |
| loss | `loss` |
| reconstruction stage | `stage` |
| device | `device` |
| checkpoint written | new checkpoint record + bytes |
| final export path | final artifact record |
| stop honored | `status=stopped` |
| exception | `status=failed` + message |

## Staging configuration

| Variable | Purpose | Notes |
|----------|---------|-------|
| `CLOUD_API_HOST` / `PORT` | Bind | Default localhost staging |
| `CLOUD_API_PUBLIC_URL` | Advertised origin | Plugin override target |
| `CLOUD_DATA_DIR` | Job/corpus/artifact store | Persist across restarts for retention tests |
| `CLOUD_MOCK_WORKER` | — | **Removed / ignored**; must not enable fake success |

Entitlement may still use staging ledger stubs for link/credit gates (`017`), but **never** fake train success.

## Tests

- Short real mapping + reconstruction smoke (tiny wavs, small `total_steps`) → real artifact bytes non-empty and structurally loadable.
- Stop mid-run → `stopped`, no final success artifact.
- Kill runner / stale heartbeat → `failed` / `worker_lost`.
- No code path marks `succeeded` with empty placeholder payload.
