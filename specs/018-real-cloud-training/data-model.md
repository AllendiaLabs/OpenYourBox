# Data Model: Real Cloud Training

**Feature**: `018-real-cloud-training`  
**Date**: 2026-08-31  
**Baseline**: Inherits entities from `specs/017-cloud-training/data-model.md` unless superseded below.

## Entities (deltas)

### CloudTrainingJob (supersedes status / transitions)

| Field | Description |
|-------|-------------|
| `jobId` | Server-assigned unique id |
| `customerSubject` | Opaque platform customer identity |
| `status` | `queued` \| `running` \| `succeeded` \| `failed` \| `stopped` (**no `paused`**) |
| `objective` | `mapping` \| `reconstruction` |
| `step` / `totalSteps` | Real progress from train runner |
| `stage` | Optional reconstruction stage |
| `loss` | Latest scalar loss from real training |
| `errorCode` / `errorMessage` | Set on failure/stop reject paths |
| `corpusId` | Retained corpus reference |
| `workerHeartbeatAt` | Last liveness signal from the train runner (server-side) |
| `finalArtifactId` | Set only on `succeeded` with a real trainable artifact |
| `createdAt` / `updatedAt` | Timestamps |

**Active job**: status ∈ {`queued`, `running`}. At most one active job per platform customer account.

**State transitions**:

```text
(none) --submit--> queued --> running --> succeeded
                      |           |
                      |           +--> failed   (train error, OOM, worker_lost, …)
                      |           |
                      |           +--> stopped  (user Stop)
                      v
                   failed (validation after accept, enqueue failure)
```

Illegal: any transition into `paused`; `succeeded` without a real final artifact.

**Validation**:
- `succeeded` ⇒ non-empty `finalArtifactId` pointing to loadable Gold-eligible bytes.
- `worker_lost` / crash reconciler ⇒ `failed`, not `running`.
- Stop from `queued` or `running` ⇒ `stopped` (not success).

---

### WorkerRun (server-side, new)

| Field | Description |
|-------|-------------|
| `jobId` | Owning cloud job |
| `pid` / `processHandle` | Optional OS process identity for the runner child |
| `workDir` | Materialized package directory |
| `device` | `cuda` \| `mps` \| `cpu` (actual) |
| `startedAt` | When training process began |
| `lastHeartbeatAt` | Updated while steps advance |
| `stopRequested` | Cooperative stop flag |

**Relationships**: 0..1 live `WorkerRun` per active job. Death of process without terminal job status triggers reconciler → job `failed`.

---

### RealCheckpoint / FinalArtifact (clarified)

Same as `017` CloudCheckpoint / FinalArtifact, with constraint:

- Payload MUST be a real recipe export (TorchScript / checkpoint bytes the VST can prepare), never empty placeholder success files.
- On `stopped` or `failed`, no new final success artifact is published; prior checkpoints may remain.

---

### TrainControlSurface (local UI, new explicit entity)

| Value | Allowed |
|-------|---------|
| `run` | Start Local or Cloud train when idle and gates pass |
| `stop` | Request stop while queued/running |
| `pause` / `resume` | **Not offered** in this feature |

---

## Unchanged from 017 (reference)

- PlatformCustomerLink, CloudEntitlement, TrainDestination, JobPackage/Manifest, RetainedCloudCorpus (30-day sliding), SubmitterAutoLoadFlag, soft upload warn threshold.

## Removed / forbidden

- MockAdvancementWorker / `CLOUD_MOCK_WORKER` product mode.
- Job status `paused` as a supported product state.
