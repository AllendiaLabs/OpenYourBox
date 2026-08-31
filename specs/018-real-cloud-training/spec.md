# Feature Specification: Real Cloud Training

**Feature Branch**: `018-real-cloud-training`

**Created**: 2026-08-31

**Status**: Draft

**Input**: User description: "let's implement real cloud training"

## Clarifications

### Session 2026-08-31

- Q: For this feature’s Definition of Done, is a working real-training staging path enough, or must a public production Allendia cloud host also be live and serving real training? → A: **Staging real-training E2E is enough to complete this feature**, but implementation and ops choices MUST remain **production-safe** (no staging-only shortcuts that would block or compromise a later public host). Public production going live is not required for Done.
- Q: When both mock advancement and real-training modes exist, what must the official product Cloud path always do? → A: **Official product Cloud path is always real training**; **remove mock/fake advancement mode** (and its code paths) from the cloud service—contract tests must exercise real job control against real-training behavior (or lightweight fixtures that are not a product “mock success” worker), never ship fake succeeded artifacts.
- Q: If the remote training worker crashes or the host reboots while a Cloud job is running, what should happen to that job? → A: **Fail the job clearly**; already-published checkpoints stay downloadable; the user resubmits if they want to continue—no auto-resume and no indefinite “running” zombie.
- Q: When the user presses Pause on a real Cloud training job, what should the remote worker do? → A: **No Pause/Resume for now**—local and cloud training expose only **Run** and **Stop**; Pause and Resume are out of scope for this feature (and matching local Train control surface).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cloud Job Produces a Real Trained Model (Priority: P1)

A sound designer with a linked Allendia account and sufficient credits prepares a valid Train setup (armed graph, eligible library audio, copyright acknowledged), chooses destination **Cloud**, and presses Run. The remote job actually trains the submitted graph on the selected audio—the same mapping or reconstruction objectives as local Train—and returns a final artifact that loads as Gold and produces audible, model-shaped output (not a placeholder or synthetic “success” file). Progress metrics (steps, loss/stage) reflect genuine training, not fabricated advancement.

**Why this priority**: Without real remote training, Cloud destination is only a UI rehearsal; this is the product value of Phase 4 cloud.

**Independent Test**: Submit a short Cloud mapping or reconstruction job against a real-training service endpoint; wait for success; confirm the submitting instance auto-loads Gold and the result behaves like a locally trained model for that objective (audible change / expected surface), not an empty or dummy weight file.

**Acceptance Scenarios**:

1. **Given** a valid Cloud-ready setup and a real-training cloud service, **When** the user submits a short mapping job and it succeeds, **Then** the submitting instance downloads and Gold-loads an artifact that is usable like a successful local mapping train
2. **Given** a valid Cloud-ready setup for reconstruction, **When** the user submits a short reconstruction job and it succeeds, **Then** the submitting instance downloads and Gold-loads an artifact usable like a successful local reconstruction train
3. **Given** a running real cloud job, **When** the Train panel shows progress, **Then** step/loss (or reconstruction stage detail) change in a way consistent with actual training advancing—not a fixed fake curve disconnected from work done
4. **Given** destination Local, **When** the user trains the same setup, **Then** local training remains unchanged and still requires no platform account

---

### User Story 2 - Checkpoints Are Real Mid-Run Artifacts (Priority: P1)

While a real cloud job runs, the service publishes intermediate checkpoints that the user can list and download. Loading a checkpoint into the live path lets them hear mid-training behavior without stopping the remote job. On success, the final artifact is the authoritative trained result; **Stop** or failure never auto-loads as success. Train controls for this feature are **Run** and **Stop** only (no Pause/Resume), matching local Train.

**Why this priority**: Hear-while-training and recovery from long jobs depend on real checkpoints, not dummy files.

**Independent Test**: Run a job long enough to publish at least one checkpoint; download and optionally load it; confirm the remote job continues; on success, confirm final Gold load; on Stop, confirm no success auto-load.

**Acceptance Scenarios**:

1. **Given** a real cloud job that has published a checkpoint, **When** the user downloads it, **Then** the file is a loadable training artifact (same role as a local mid-run checkpoint), not an empty placeholder
2. **Given** a downloaded checkpoint, **When** the user optionally loads it, **Then** the live path can use it while the remote job remains running
3. **Given** the user Stops a real cloud job, **When** the terminal state is shown, **Then** no success auto-load occurs and any partial checkpoints remain optionally downloadable if the service still exposes them

---

### User Story 3 - Operators Can Run Real Training Locally for Staging (Priority: P1)

A developer or operator stands up the cloud training service in a staging configuration on a machine they control. With the plugin pointed at that staging endpoint (and staging account link as already supported for cloud client testing), Cloud Run executes **real** training recipes. They can verify end-to-end: submit → real progress → real artifact → Gold load, before any public production host is required. There is **no** product mock/fake advancement mode that reports success without training.

**Why this priority**: The team must be able to validate real cloud training without waiting for a fully public production deployment.

**Independent Test**: Start the staging cloud service; point the plugin at it; complete User Story 1 on a short job; confirm results are real. Confirm no mock/fake worker path remains that can mark a job succeeded with a non-trainable artifact.

**Acceptance Scenarios**:

1. **Given** the staging service is running, **When** a Cloud job is accepted, **Then** the worker performs actual training on the submitted package and publishes real progress and artifacts
2. **Given** automated contract tests for auth, entitlement, and job control (Run/Stop), **When** those tests run, **Then** they do so without relying on a product mock/fake advancement worker that fabricates success artifacts
3. **Given** a job package invalid for training (same classes of failure local Train would reject after accept where applicable), **When** the job runs, **Then** it fails with a clear reason and does not report success with a dummy artifact

---

### User Story 4 - Parity and Failure Honesty (Priority: P2)

Users and operators trust that a Cloud success means the same creative outcome class as Local success for that objective. If remote training fails (bad package, resource exhaustion, interrupted compute), the plugin shows a clear failure—never a fake “succeeded” with a non-trainable placeholder. Capacity and one-job-per-account rules from official cloud access still apply.

**Why this priority**: False success destroys trust faster than slow real training.

**Independent Test**: Force a known failing train package or stop mid-run; confirm failed/stopped states and no Gold auto-load; compare a successful short Cloud job’s loadability to a successful short Local job for the same objective.

**Acceptance Scenarios**:

1. **Given** a cloud job that cannot complete real training, **When** the worker terminates in failure, **Then** status is failed with a readable reason and the submitting instance does not auto-load Gold
1a. **Given** a running cloud job, **When** the training worker crashes or the host reboots, **Then** the job becomes failed (not left running), any already-published checkpoints remain downloadable, and no success auto-load occurs
2. **Given** two short successful jobs—one Local, one Cloud—for the same objective and comparable setup, **When** both auto-load Gold, **Then** both produce loadable trained models suitable for hearing the trained chain (parity of outcome class, not bit-identical weights)
3. **Given** an account already has an active real cloud job, **When** another Cloud submit is attempted under that account, **Then** it is refused with the existing one-job-per-account message

---

### Edge Cases

- What if the staging machine has no GPU? Real training still runs on available compute (CPU allowed); the job may be slower, but progress and artifacts remain real.
- What if real training runs out of disk or memory on the worker? The job fails clearly; no success auto-load; corpus retention rules from official cloud still apply to uploaded audio.
- What if the remote job succeeds but the artifact cannot be loaded in the VST? Treat as download/prepare failure after remote success (retryable); do not partially corrupt the live graph—same honesty rules as existing cloud client behavior.
- What if Local and Cloud recipe support diverge for an element? Cloud MUST reject or fail jobs that use unsupported elements before claiming success; it MUST NOT invent a fake successful model.
- What if the remote training worker crashes or the host reboots mid-job? The job MUST transition to **failed** with a clear recoverable message; already-published checkpoints MUST remain downloadable when still available; the service MUST NOT auto-resume the same job id and MUST NOT leave it stuck as running indefinitely.
- Network loss during a real job: remote training continues; plugin shows offline/reconnect; does not imply cancel (existing cloud monitor behavior).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Accepted Cloud jobs MUST execute actual training for the submitted job package using the same objective families as local Train (**mapping** and **reconstruction**).
- **FR-002**: Successful Cloud jobs MUST produce final artifacts that are eligible for the same Gold auto-load success path as local Train for that objective on the submitting instance.
- **FR-003**: The cloud service MUST NOT report job status `succeeded` with a placeholder, empty, or synthetic non-trainable artifact.
- **FR-004**: While a Cloud job runs, reported progress (steps and loss or reconstruction stage metrics) MUST reflect actual worker advancement.
- **FR-005**: Cloud jobs MUST publish intermediate checkpoints when the job’s train options request them (same user-visible checkpoint cadence intent as local Train), and those checkpoints MUST be downloadable as loadable artifacts.
- **FR-006**: Users MUST be able to **Stop** an in-flight Cloud (and Local) training job from the Train panel; Stop MUST end without success auto-load. **Pause** and **Resume** MUST NOT be offered for local or cloud training in this feature.
- **FR-006a**: Local and cloud Train control surfaces MUST expose the same action set for this slice: **Run** and **Stop** only.
- **FR-007**: The cloud service MUST **not** ship a mock/fake advancement worker or mode that fabricates job success or non-trainable success artifacts; any legacy mock advancement code MUST be removed. Automated tests MAY use fixtures or short real jobs, but MUST NOT depend on a product fake-success path.
- **FR-008**: Staging MUST allow pointing the existing plugin cloud client at a real-training service endpoint so end-to-end Cloud Run → real artifact → Gold load can be validated without a public production host.
- **FR-008a**: Real-training delivery for this feature MUST NOT introduce staging-only shortcuts (auth, entitlement, artifact honesty, retention, or security posture) that would be unsafe or incompatible for a later public production host; public production going live is **not** required for this feature’s Done bar.
- **FR-009**: Real cloud training MUST preserve existing official cloud access rules: linked platform customer account, entitlement at submit, one active cloud job per account, account-wide monitor/control (**Stop** and status/download; no Pause/Resume), submitter-only success auto-load, corpus retention policy, and non-blocking audio.
- **FR-010**: Local destination training MUST remain fully usable without a platform account and MUST NOT regress because real cloud training was added.
- **FR-011**: If real training cannot complete, the service MUST mark the job failed (or stopped if user-requested) with a user-readable reason when available; the plugin MUST NOT treat those outcomes as success.
- **FR-011a**: If the training worker crashes or the host reboots while a job is active, the service MUST mark that job **failed** with a clear recoverable message, MUST keep already-published checkpoints downloadable when available, MUST NOT auto-resume the same job, and MUST NOT leave the job indefinitely in a running state.
- **FR-012**: Cloud training MUST accept the same job package contents the plugin already submits for Cloud (armed graph snapshot, selected corpus, train options); unsupported or invalid packages MUST fail validation or training honestly.
- **FR-013**: Cloud training SHOULD use accelerator hardware when available and MAY fall back to CPU; absence of a GPU MUST NOT enable fake/synthetic success behavior.

### Key Entities

- **Cloud Training Service**: Hosted (or staging) remote executor for official Cloud jobs; always performs real training—no product mock/fake advancement mode.
- **Real Cloud Job**: An account-owned remote job that executes real training for a submitted package; statuses and controls match the existing cloud job model.
- **Real Checkpoint / Final Artifact**: Mid-run or final trainable outputs produced by real training; suitable for download and optional or success Gold load.
- **Staging Endpoint**: Operator-controlled cloud service URL used by the plugin for end-to-end real-training validation before public hosting.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In guided staging tests, at least 90% of short eligible Cloud mapping jobs that the service accepts reach success with a Gold-loadable artifact on the submitting instance (excluding intentional Stop and induced resource failures).
- **SC-002**: In the same staging regime, at least 90% of short eligible Cloud reconstruction jobs that the service accepts reach success with a Gold-loadable artifact on the submitting instance.
- **SC-003**: In 100% of guided tests, a reported `succeeded` job yields an artifact that the submitting instance can prepare/load as Gold without substituting a known-empty placeholder file.
- **SC-004**: During an active Cloud run, Train panel progress updates while online at least once every 5 seconds on average, and observed step/loss (or stage) changes correlate with wall-clock training work (not a pre-scripted fake curve alone).
- **SC-005**: Optional checkpoint download during a real run leaves the remote job running in 100% of tests where the job had not finished; downloaded checkpoints are loadable in at least 90% of guided mid-run attempts when the job had published them.
- **SC-006**: Induced training failures, worker/host crash mid-job, and user Stop result in no success auto-load in 100% of guided negative tests; crash cases show failed (not stuck running) and retain downloadable checkpoints when any were published.
- **SC-006a**: Guided tests confirm Train offers Run and Stop only for both Local and Cloud destinations (no Pause/Resume controls) in 100% of UI control-surface checks for this feature.
- **SC-007**: Operators can complete one full real Cloud success path (submit → progress → Gold hear) on staging in a single session without needing a public production host; guided review confirms no known staging-only shortcuts that would block a later production host (auth/entitlement gates, real artifacts only on success, retention and one-job rules intact).
- **SC-008**: No mock/fake advancement worker or mode remains in the cloud service that can mark a job `succeeded` with a non-trainable artifact; automated auth/entitlement/job-control (Run/Stop) tests pass without depending on such a path.
- **SC-009**: Continuous DAW playback during real cloud jobs shows no cloud-induced audio dropouts attributable to submit/monitor/download; audible model stays pre-job until intentional load.
- **SC-010**: Users who never use Cloud can still complete a local Train success path with no cloud dependency.

## Assumptions

- Feature `017-cloud-training` (plugin Cloud destination, account link, entitlement gates, job package submit/monitor/download UX) is the baseline; this feature upgrades the **remote execution** from fake/mock results to real training rather than redesigning Train UI, and **removes** mock/fake advancement worker code from the cloud service.
- Constitution Phase 4 still applies: VST is the only training UI; official cloud remains account- and entitlement-gated; local train stays account-free; hosted cloud backend remains proprietary for the service.
- “Real” means the same creative outcome class as local Train (actual optimization over submitted audio/graph, loadable Gold artifact)—not bit-identical weights to a local run.
- Public production hosting (DNS, TLS, storefront billing at scale) may follow staging validation and is **not** required for this feature’s Done bar; this feature’s minimum bar is a real-training service mode exercisable from the plugin end-to-end on staging, implemented in a **production-safe** way (no deliberate staging-only compromises that would block or weaken a later public host).
- Short guided jobs (reduced step counts) are acceptable for acceptance tests; long production-quality trains are not required to prove the feature. Automated cloud contract tests may use short real jobs or non-success fixtures, but not a product fake-success worker.
- Mid-job worker or host loss fails the job (no auto-resume of that job id); published checkpoints remain downloadable when available.
- Local and cloud training control surfaces for this feature are **Run** and **Stop** only; Pause/Resume are deferred.
- Existing soft upload warning, corpus retention (30 days from last use with extend-on-reuse), and one-job-per-account limits remain as specified for official cloud.
- Marketplace selling and in-plugin checkout remain out of scope.
