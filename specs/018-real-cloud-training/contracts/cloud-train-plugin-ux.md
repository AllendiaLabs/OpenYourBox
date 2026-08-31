# Contract: Cloud Train Plugin UX (Run / Stop)

**Feature**: `018-real-cloud-training`  
**Supersedes**: Pause/Resume portions of `specs/017-cloud-training/contracts/cloud-train-plugin-ux.md`  
**Baseline**: Destination Local|Cloud, account link, entitlement messages, soft size warn, submitter auto-load from `017`

## Train control surface

| Destination | Primary actions |
|-------------|-----------------|
| Local | **Run**, **Stop** |
| Cloud | **Run**, **Stop** |

**Must not show** Pause or Resume for either destination in this feature.

## Status mapping (cloud)

| Remote status | Panel treatment |
|---------------|-----------------|
| `queued` | Queued / waiting |
| `running` | Training… + real step/loss/stage |
| `succeeded` | Success path (submitter auto-load / non-submitter manual) |
| `failed` | Failure message (`worker_lost` readable) |
| `stopped` | Stopped; no success auto-load |

If a legacy `paused` status is ever received, treat as non-controllable anomaly (show message / map to running or failed)—do not offer Resume.

## Cloud client

- Support `stop` only among control verbs.
- Remove or no-op pause/resume client methods; UI must not call them.
- Poll/download/checkpoint behavior unchanged from `017`.

## Local worker commands

- UI writes **stop** only (no pause/resume commands from the panel).
- Existing worker pause handling may remain unused dead code until cleaned; product surface must not expose it.

## Staging pointer

- No API URL fields in the main Train hyperparameter chrome (per `017`).
- Operators set `apiBaseUrlOverride` / `storefrontUrlOverride` via the Allendia
  account panel (**Cloud endpoint (staging / RunPod)**) or by editing `cloud.xml`
  under the user data root, so Cloud Train hits staging / RunPod
  `CLOUD_API_PUBLIC_URL`.

## Audio thread

- Run/Stop initiation, packaging, HTTP, and downloads remain off `processBlock`.
