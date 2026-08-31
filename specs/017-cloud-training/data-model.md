# Data Model: Cloud Training

**Feature**: `017-cloud-training`  
**Date**: 2026-08-31

## Entities

### CloudApiToken (local)

| Field | Description |
|-------|-------------|
| `token` | Secret Bearer credential (stored locally; never shown in full after save) |
| `configured` | Whether a non-empty token is saved |
| `baseUrlOverride` | Optional staging/dev API base URL; empty → product default |

**Validation**: Cloud Run requires `configured`; empty token clears cloud capability.

**Relationships**: Authorizes all cloud API calls; scopes job ownership.

---

### TrainDestination (local UI state)

| Value | Meaning |
|-------|---------|
| `local` | Existing ChildProcess `train_worker` path |
| `cloud` | Remote job via `CloudTrainClient` |

**Persistence**: Last-used destination may be remembered per plugin instance (optional UX); default `local` so no-token users are unaffected.

---

### CloudTrainingJob (remote + local mirror)

| Field | Description |
|-------|-------------|
| `jobId` | Server-assigned unique id |
| `tokenSubject` | Opaque token identity (server-side; not the raw token in logs) |
| `status` | `queued` \| `running` \| `paused` \| `succeeded` \| `failed` \| `stopped` |
| `objective` | `mapping` \| `reconstruction` |
| `step` / `totalSteps` | Progress counters when running |
| `stage` | Optional; `representation` \| `quality` for reconstruction |
| `loss` | Latest scalar loss when reported |
| `errorCode` / `errorMessage` | Set on failure or reject |
| `corpusId` | Retained corpus reference |
| `createdAt` / `updatedAt` | Timestamps |
| `submitterClientInstanceId` | Id of submitting plugin instance (informational on server) |

**Active job**: status ∈ {`queued`, `running`, `paused`}. At most one active job per token.

**State transitions**:

```text
(none) --submit--> queued --> running <-> paused
                      |           |
                      v           v
                   failed     succeeded | failed | stopped
```

Stop/fail/success are terminal. Pause only from `running`. Resume only from `paused`.

---

### JobPackage (submit payload)

See `contracts/cloud-job-package.md`.

| Field | Description |
|-------|-------------|
| `manifest` | Graph fragment, armed ids, train_options, objective, client metadata |
| `corpusFiles` | Selected library audio to upload **or** |
| `corpusId` | Reference to retained corpus (reuse; resets retention) |

**Validation**: Same local gates before build (copyright, objective eligibility, mixed SR, etc.). Soft warn if total upload bytes &gt; `cloudSoftUploadWarnBytes` (default 2 GiB).

---

### RetainedCloudCorpus (remote)

| Field | Description |
|-------|-------------|
| `corpusId` | Unique id |
| `tokenSubject` | Owning token |
| `byteSize` | Stored size |
| `lastUsedAt` | Updated on create and on qualifying reuse |
| `expiresAt` | `lastUsedAt + 30 days` (computed or stored) |
| `objectKeys` | Storage locations for audio blobs |

**Rules**: Sweeper deletes when `now > expiresAt` and no active job pins the corpus. Reuse via new job with `corpusId` resets `lastUsedAt`.

---

### CloudCheckpoint (remote + downloadable)

| Field | Description |
|-------|-------------|
| `checkpointId` | Unique within job |
| `jobId` | Parent job |
| `step` / `stage` | Identity for UI |
| `createdAt` | Publish time |
| `downloadUrl` | Time-limited signed URL (or API path requiring auth) |
| `localPath` | After download (plugin-side) |

---

### FinalCloudArtifact (remote + downloadable)

| Field | Description |
|-------|-------------|
| `jobId` | Parent |
| `artifactKind` | TorchScript / weight bundle suitable for Gold load |
| `downloadUrl` | Signed/auth download |
| `localPath` | After download |

**Auto-load**: Only if local session has `isSubmitter == true` for this `jobId` and status is `succeeded`.

---

### JobMonitorSession (local)

| Field | Description |
|-------|-------------|
| `jobId` | Attached job |
| `isSubmitter` | True iff this instance submitted the job (persisted across restart on that machine) |
| `destination` | `cloud` |
| `lastStatus` | Mirrored status for UI |
| `offline` | True when poll fails due to network (job not implied cancelled) |

**Rediscovery**: On startup / token present → `GET /jobs` (active + recent); attach without requiring local job id if token-wide list returns the job. `isSubmitter` remains true only if local persistence records submit for that `jobId` (other machines: `isSubmitter=false`).

---

### SoftUploadWarning (local derived)

| Field | Description |
|-------|-------------|
| `totalBytes` | Sum of files that would be uploaded |
| `thresholdBytes` | Default `2147483648` (2 GiB) |
| `warned` | UI has shown soft warning for this submit attempt |

Does not block submit.

## Relationships (summary)

```text
CloudApiToken ──authorizes──► CloudTrainingJob
CloudTrainingJob ──uses──► RetainedCloudCorpus
CloudTrainingJob ──publishes──► CloudCheckpoint*
CloudTrainingJob ──publishes──► FinalCloudArtifact?
JobMonitorSession ──mirrors──► CloudTrainingJob
JobPackage ──creates/updates──► RetainedCloudCorpus + CloudTrainingJob
```
