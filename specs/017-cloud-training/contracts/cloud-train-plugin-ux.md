# Contract: Cloud Training Plugin UX

**Feature**: `017-cloud-training`  
**Applies to**: `TrainPanel`, cloud settings UI, `TrainCoordinator` / `CloudTrainClient` wiring

## Settings

- Actions: **Link account** (starts storefront link flow), **Disconnect**, optional advanced Base URL / Storefront URL.
- During link: show user code (if device-code) and open verification URL in system browser; poll until linked, cancelled, or expired.
- Status line: Not linked | Linked | Auth error | Entitlement unavailable (last probe).
- Affordance: **Open storefront** (account / credits) — launches storefront URL outside the VST (FR-002a).
- No in-plugin checkout, cart, or payment form.

## Train panel

### Destination

- Control: **Local** | **Cloud** (same panel as objective / Run / Pause / Stop).
- Cloud + not linked → Run disabled or Run shows link-account error (FR-005a, FR-015).
- Cloud + linked + insufficient entitlement → Run refused with entitlement message; MAY offer Open storefront (FR-005a, US5).
- Local ignores account link (FR-017).

### Soft size warning

- Before/at Cloud Run, if selected upload bytes &gt; 2 GiB (default), show non-blocking warning; user may proceed (FR-016a).

### Busy rules

- One active train job per instance (local or cloud) (FR-013).
- Cloud submit may additionally fail with `one_job_per_account` (FR-013a) — show that message clearly.

### Progress

- Map remote statuses into existing Train chrome; show `queued` distinctly.
- Poll while attached and online; on network loss show offline/reconnect **without** claiming job cancelled (US3).
- Loss / stage display mirrors local when fields present.

### Checkpoints

- List remote checkpoints; Download stores locally; optional Load uses existing hear-while-training path; job keeps running (FR-009).

### Success / failure

- **Submitter** (`isSubmitter`): on `succeeded`, download final artifact and auto-load Gold (same policy as local) (FR-010).
- **Non-submitter**: show success; offer Download / Load; **do not** auto-swap graph (FR-010a).
- `failed` / `stopped`: message; no success auto-load (FR-011).
- Download failure after server success: retryable error; do not partially corrupt graph.

### Rediscovery

- On open with linked account: list account jobs; allow attach to active/recent.
- Local persistence: remember `jobId` + `isSubmitter` for jobs this instance submitted so restart on same machine still auto-loads.

## Audio / threading

- No HTTP, file packing, browser-wait loops, or waits on the audio thread.
- Audible model unchanged until intentional success auto-load or optional checkpoint load (FR-006).

## Implementation anchors

- `OpenYourBox/Source/ui/TrainPanel.*`
- `OpenYourBox/Source/train/TrainCoordinator.*`
- `OpenYourBox/Source/train/CloudTrainClient.*`
- `OpenYourBox/Source/train/CloudSettings.*`
