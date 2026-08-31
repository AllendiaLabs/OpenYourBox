# Data Model: Cloud Training

**Feature**: `017-cloud-training`  
**Date**: 2026-08-31

## Entities

### PlatformCustomerLink (local)

| Field | Description |
|-------|-------------|
| `accessToken` | Linked session Bearer credential (stored locally; never shown in full after save) |
| `refreshToken` | Optional refresh credential if issued |
| `customerIdHint` | Optional opaque customer id for UI/debug (never a password) |
| `linked` | Whether a non-empty valid session is saved |
| `baseUrlOverride` | Optional staging/dev API base URL; empty → product default |
| `storefrontUrlOverride` | Optional staging storefront URL; empty → product default |

**Validation**: Cloud Run requires `linked` and successful entitlement check; disconnect clears cloud capability for remote jobs only.

**Relationships**: Authorizes all cloud API calls; scopes job ownership to the platform customer account.

---

### CloudEntitlement (remote + local cache)

| Field | Description |
|-------|-------------|
| `sufficient` | Whether a new cloud job may be submitted now |
| `balanceHint` | Optional display string/units from server (ops-defined) |
| `checkedAt` | Last successful probe time (local cache) |
| `source` | Server/storefront authoritative; local cache advisory |

**Validation**: `POST /v1/jobs` requires `sufficient == true` server-side. Stale local cache MUST NOT override a server `insufficient_entitlement` response.

**Relationships**: Belongs to platform customer account; consumed/reserved per ops rules at job accept.

---

### TrainDestination (local UI state)

| Value | Meaning |
|-------|---------|
| `local` | Existing ChildProcess `train_worker` path |
| `cloud` | Remote job via `CloudTrainClient` |

**Persistence**: Last-used destination may be remembered per plugin instance (optional UX); default `local` so unlinked users are unaffected.

---

### CloudTrainingJob (remote + local mirror)

| Field | Description |
|-------|-------------|
| `jobId` | Server-assigned unique id |
| `customerSubject` | Opaque platform customer identity (server-side; not raw credentials in logs) |
| `status` | `queued` \| `running` \| `paused` \| `succeeded` \| `failed` \| `stopped` |
| `objective` | `mapping` \| `reconstruction` |
| `step` / `totalSteps` | Progress counters when running |
| `stage` | Optional; `representation` \| `quality` for reconstruction |
| `loss` | Latest scalar loss when reported |
| `errorCode` / `errorMessage` | Set on failure or reject |
| `corpusId` | Retained corpus reference |
| `createdAt` / `updatedAt` | Timestamps |
| `submitterClientInstanceId` | Id of submitting plugin instance (informational on server) |

**Active job**: status ∈ {`queued`, `running`, `paused`}. At most one active job per platform customer account.

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

**Validation**: Same local gates before build (copyright, objective eligibility, mixed SR, etc.). Soft warn if total upload bytes &gt; `cloudSoftUploadWarnBytes` (default 2 GiB). Cloud path also requires linked account + entitlement.

---

### RetainedCloudCorpus (remote)

| Field | Description |
|-------|-------------|
| `corpusId` | Unique id |
| `customerSubject` | Owning platform customer |
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

**Rediscovery**: On startup / linked account present → `GET /v1/jobs` (active + recent); attach without requiring local job id if account-wide list returns the job. `isSubmitter` remains true only if local persistence records submit for that `jobId` (other machines: `isSubmitter=false`).

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
PlatformCustomerLink ──authorizes──► CloudTrainingJob
PlatformCustomerLink ──has──► CloudEntitlement
CloudTrainingJob ──uses──► RetainedCloudCorpus
CloudTrainingJob ──publishes──► CloudCheckpoint*
CloudTrainingJob ──publishes──► FinalCloudArtifact?
JobMonitorSession ──mirrors──► CloudTrainingJob
JobPackage ──creates/updates──► RetainedCloudCorpus + CloudTrainingJob
```
