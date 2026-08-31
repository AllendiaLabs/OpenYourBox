# Feature Specification: Cloud Training

**Feature Branch**: `017-cloud-training`

**Created**: 2026-08-31

**Status**: Draft

**Input**: User description: "I now want to implement cloud training before connecting to my wordpress website so that users can add credits for cloud. Let's start with cloud training"

## Clarifications

### Session 2026-08-31

- Q: What should “cloud training” cover in this first slice (before WordPress credits)? → A: Submit local graph/project to a remote GPU job, monitor progress, download checkpoints. WordPress payment and credit purchase are deferred.
- Q: Without WordPress yet, how should users identify themselves to the cloud training service? → A: Developer/beta API token entered once in plugin settings.
- Q: Which training jobs should cloud support in this first slice? → A: Same as local today: mapping and reconstruction (RAVE) objectives, same Library and Train panel.
- Q: After a cloud training job finishes (success, failure, or stop), how long should the service keep the uploaded training audio on the server? → A: Keep for a fixed base period of **30 days**; **extend (reset) that retention window every time the retained corpus is used again** (e.g. re-run training from it, or other product-defined reuse). Idle unused corpus expires after 30 days from last use.
- Q: How many cloud training jobs may run at the same time under one API token? → A: **One** active cloud job per token for now; additional submits are rejected or clearly refused until the active job reaches a terminal state.
- Q: If a user starts a cloud job on one computer, can they monitor and control that same job from another computer using the same API token? → A: **Yes (token-wide)** — any plugin instance with the same valid token can list, monitor, pause/resume/stop, and download for jobs owned by that token.
- Q: Should cloud submit enforce a maximum size for the selected training audio upload in this first slice? → A: **Soft warning** above a size threshold, but still allow the upload (no hard refuse-before-upload cap in this slice).
- Q: When a cloud job succeeds, which plugin instance should automatically download and load the Gold model? → A: **Auto-load only on the machine/instance that originally submitted the job**; other instances with the same token can still manually download/load.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure Cloud Access Once (Priority: P1)

A sound designer opens plugin settings and enters a cloud training API token (issued for developer or beta access). They can save, clear, or replace the token. The plugin stores the token securely on the local machine and shows a clear connected / not-configured status without leaving the VST. WordPress login, account creation, and buying credits are not part of this story.

**Why this priority**: Without a way to identify the user to the remote service, no cloud job can be submitted.

**Independent Test**: Enter a valid token, confirm status shows ready; clear the token and confirm Cloud Train is unavailable; replace with an invalid token and confirm a clear rejection when testing or submitting.

**Acceptance Scenarios**:

1. **Given** no token is configured, **When** the user opens cloud-related settings, **Then** they can enter and save an API token
2. **Given** a saved token, **When** the user reopens the plugin later, **Then** the token remains available (masked in the UI) and cloud status reflects that a token is configured
3. **Given** a configured token, **When** the user clears it, **Then** cloud submission is disabled until a new token is saved
4. **Given** an invalid or revoked token, **When** the user attempts a cloud action that requires authentication, **Then** the plugin shows a clear authentication error and does not start a job

---

### User Story 2 - Start a Cloud Job From the Same Train Panel (Priority: P1)

With copyright acknowledgment complete, an armed trainable graph, and selected library audio, the user opens the existing Train panel and chooses **destination: Cloud** (instead of Local). They keep the same objective choice (mapping or reconstruction), Run controls, and library selection rules already used for local training. Starting Run packages the armed graph snapshot, selected training data, and train options, then submits them to the remote cloud service. Live DAW audio continues on the previously loaded model; the local machine is not required to keep a heavy training process running for the job to proceed once accepted by the cloud.

**Why this priority**: This is the core value of cloud training—same creative workflow, remote compute.

**Independent Test**: Prepare a valid local train setup, switch destination to Cloud with a valid token, press Run, confirm the job is accepted and shows as queued or running while audio keeps playing.

**Acceptance Scenarios**:

1. **Given** a valid local train setup (armed graph, eligible selected library entries, copyright acknowledged), **When** the user sets destination to Cloud and presses Run with a valid token, **Then** the plugin submits one cloud job and shows that submission succeeded
2. **Given** destination Cloud and no configured token, **When** the user presses Run, **Then** Train refuses with a clear prompt to configure the API token
3. **Given** destination Cloud, **When** the user chooses objective mapping or reconstruction, **Then** the same eligibility rules as local Train apply (e.g. mapping rejects unpaired-only selections; reconstruction accepts pairs and unpaired clips per existing library rules)
4. **Given** an active cloud job for this plugin instance, **When** the user tries to start another cloud or local train from the same instance, **Then** the plugin blocks the new start with a clear busy state (one active train job per instance)
4a. **Given** an active cloud job already running under this API token (even from another plugin instance or machine), **When** the user tries to submit another cloud job with the same token, **Then** the service/plugin refuses with a clear one-job-per-token message until the active job finishes, fails, or is stopped
5. **Given** cloud submission in progress or a job running remotely, **When** the DAW plays audio, **Then** the audible model remains the pre-job model until a successful result is intentionally loaded
6. **Given** destination Local, **When** the user trains, **Then** existing local training behavior is unchanged

---

### User Story 3 - Monitor Progress and Control a Remote Job (Priority: P1)

After a cloud job is accepted, the user watches progress inside the same Train panel: status (queued, running, paused, failed, succeeded, stopped), step/stage progress where applicable, and live loss (or equivalent training metric) when the remote job reports it. They can request Pause, Resume, and Stop for the remote job. Closing the plugin window or disconnecting briefly must not permanently lose the ability to reconnect to an in-flight job the user still owns when they return with network available.

**Why this priority**: Long remote jobs are useless without trustworthy monitoring and control from the VST.

**Independent Test**: Start a short cloud job; confirm status and metric updates appear; Pause then Resume; Stop and confirm the job ends without auto-loading a model.

**Acceptance Scenarios**:

1. **Given** a submitted cloud job, **When** the remote status changes (queued → running → …), **Then** the Train panel updates status within a short, user-noticeable interval while online
2. **Given** a running cloud job that reports loss/metrics, **When** training steps advance, **Then** the panel shows updating progress and loss (or stage detail for reconstruction) comparable to local Train feedback
3. **Given** a running cloud job, **When** the user presses Pause, **Then** the remote job pauses and the panel reflects paused; Resume continues the same job
4. **Given** a running or paused cloud job, **When** the user presses Stop, **Then** the job stops, Stop is not treated as success, and the live model is not replaced
5. **Given** a cloud job started from this machine, **When** the user closes and reopens the plugin (or temporarily loses then regains network), **Then** they can rediscover the in-flight or recently finished job and resume monitoring without resubmitting
5a. **Given** a cloud job started on machine A under a token, **When** the user opens the plugin on machine B with the same valid token, **Then** they can list that token’s active or recent job and monitor, pause/resume/stop, and download without resubmitting
6. **Given** network loss during monitoring, **When** the connection drops, **Then** the panel shows a clear offline/reconnect state without implying the remote job was cancelled solely due to the disconnect

---

### User Story 4 - Retrieve Checkpoints and Load Successful Results (Priority: P1)

While a cloud job runs, the user can list available intermediate checkpoints reported by the job and optionally download one to try in the live path (hear-while-training style), without ending the remote job. When the job succeeds, the **submitting** plugin instance downloads the final trained artifact and auto-loads it as a Gold BlackBox for the armed chain using the same success rules as local training (including mapping vs reconstruction surfaces). Other instances with the same token can download/load manually but do not auto-load. Failed jobs surface a readable reason and never auto-load.

**Why this priority**: Users need the same creative outcome as local Train—usable Gold models and optional mid-run listening.

**Independent Test**: During a run, download an intermediate checkpoint and optionally load it; on full success, confirm auto-load Gold; on failure, confirm no auto-load and an error message.

**Acceptance Scenarios**:

1. **Given** a running cloud job that has published at least one checkpoint, **When** the user views checkpoints, **Then** they see a list with enough identity (e.g. step/stage/time) to choose among them
2. **Given** a listed checkpoint, **When** the user downloads it, **Then** it is stored locally and can be optionally loaded into the live path while the cloud job continues
3. **Given** a successful cloud job, **When** success is reported and the final artifact is available, **Then** the **submitting** plugin instance downloads it and auto-loads the armed chain as Gold under the same success policy as local Train
3a. **Given** a successful cloud job viewed from a non-submitting instance with the same token, **When** success is shown, **Then** that instance does **not** auto-load Gold into its graph, but the user can manually download and load the artifact
4. **Given** the user Stopped the job or the job Failed, **When** the terminal state is shown, **Then** no success auto-load occurs
5. **Given** a downloaded cloud artifact, **When** Weights / checkpoint browsing is used later, **Then** the user can treat it like other trained weight files produced locally

---

### User Story 5 - Understand Limits Before Credits Exist (Priority: P2)

Until WordPress credit purchase exists, the user sees that cloud training is a beta/developer capability gated by API token—not by a spendable credit balance. The UI does not offer “add credits” or open a WordPress storefront. Optional informational copy may state that paid credits will come later. The service enforces **at most one active cloud job per API token**; hitting that limit MUST be explained clearly.

**Why this priority**: Prevents confusion and false expectations about billing while still allowing cloud compute to ship.

**Independent Test**: With a valid token and no credit UI, complete a cloud submit path; confirm no purchase/credits flow appears; trigger a concurrency limit if available and confirm a clear message.

**Acceptance Scenarios**:

1. **Given** this feature slice, **When** the user explores Train and settings, **Then** there is no path to purchase or top up credits via WordPress
2. **Given** the service rejects a job due to a capacity or concurrency limit, **When** submission fails, **Then** the user sees a clear, actionable message (not a generic failure)
3. **Given** local training, **When** the user prefers not to use cloud, **Then** they can continue training entirely on-device with destination Local

---

### Edge Cases

- What happens when selected library audio is very large? The plugin shows a **soft warning** when the selected corpus exceeds a documented size threshold, but **still allows** packaging and upload; transfer progress remains visible. Upload may still fail for network/service reasons with a clear error—never hang the UI indefinitely with no status.
- What happens if the armed graph changes after Run? The cloud job uses the snapshot taken at submit; live edits do not alter the remote job.
- What happens if the token is cleared while a job is running? Monitoring may fail authentication; the remote job should not be silently deleted solely because the local token was cleared—user is warned they may lose local ability to control the job until the token is restored.
- What happens if download of the final artifact fails after remote success? The panel shows success-on-server plus a retryable download/load error; auto-load does not partially corrupt the live graph.
- What happens if two plugin instances on the same machine both try cloud Train? Each instance may own at most one active job; **additionally, the token allows only one active cloud job total**, so the second instance’s cloud submit is refused with a clear one-job-per-token message.
- How are mixed sample-rate or ineligible library selections handled? Same gates as local Train before any upload begins.
- What if the copyright acknowledgment is missing? Cloud Run stays disabled exactly like local Run.
- What happens to uploaded training audio after the job ends? The service retains it for **30 days from last use**; any qualifying reuse (e.g. starting another cloud job from the same retained corpus) **resets** that 30-day window. After idle expiry, the corpus is deleted and is no longer available for re-run without a fresh upload.
- What if the user needs the audio after idle expiry? They must re-upload via a new cloud submit from their local library; the plugin does not promise permanent cloud storage of corpus audio.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to save, replace, and clear a cloud training API token from within the plugin settings.
- **FR-002**: The plugin MUST keep the token out of plain display after save (masked) and MUST NOT require a separate desktop application to configure cloud access.
- **FR-003**: The Train panel MUST offer a train destination choice of **Local** or **Cloud** without introducing a second Train product or separate capture/library workflow.
- **FR-004**: Cloud Train MUST support the same objectives as local Train in this product today: **mapping** and **reconstruction**, with the same library eligibility and copyright gates.
- **FR-005**: Cloud Run MUST submit a job package consisting of the armed trainable graph snapshot, selected library audio (or references resolved for transfer), and the same user-visible train options used for that objective locally.
- **FR-006**: Cloud submission and monitoring MUST NOT stall, glitch, or block the real-time audio path; the audible model remains the pre-job model until an intentional successful load or optional checkpoint load.
- **FR-007**: Users MUST be able to monitor cloud job status (at least: queued, running, paused, succeeded, failed, stopped) and training progress metrics when reported by the job.
- **FR-008**: Users MUST be able to request Pause, Resume, and Stop for their cloud job from the Train panel, with Stop never counting as success auto-load.
- **FR-009**: Users MUST be able to list and download intermediate checkpoints from a running or completed cloud job and optionally load a downloaded checkpoint into the live path while a job continues.
- **FR-010**: On cloud job success, the **submitting** plugin instance MUST download the final artifact and auto-load the armed chain as Gold under the same success rules as local training for that objective.
- **FR-010a**: Non-submitting plugin instances that share the same token MUST NOT auto-load the successful artifact into their graph; they MUST still be able to manually download and optionally load it.
- **FR-011**: On cloud job failure or user Stop, the plugin MUST show a clear reason when available and MUST NOT perform success auto-load.
- **FR-012**: Users MUST be able to rediscover and resume monitoring of an in-flight or recently finished cloud job they own after plugin restart or temporary network loss, when authentication still succeeds.
- **FR-012a**: Job ownership MUST be **token-wide**: any plugin instance authenticated with the same valid API token MUST be able to list, monitor, pause, resume, stop, and download checkpoints/artifacts for that token’s active or recent cloud job(s), including jobs submitted from a different machine.
- **FR-013**: At most one active train job (local or cloud) MUST be allowed per plugin instance; attempts to start another MUST be refused with a clear busy message.
- **FR-013a**: The cloud service MUST allow at most **one active cloud job per API token** at a time; additional cloud submits under that token MUST be refused with a clear one-job-per-token message until the active job reaches a terminal state (succeeded, failed, or stopped).
- **FR-014**: This slice MUST NOT include WordPress account linking, credit purchase, credit balance spending, or marketplace browsing.
- **FR-015**: When the remote service rejects a job (auth, capacity, concurrency, validation), the plugin MUST show a specific, user-readable error rather than failing silently.
- **FR-016**: Packaging/upload progress MUST be visible for non-trivial transfers so users know the submit step is still working.
- **FR-016a**: When the selected corpus exceeds a documented soft size threshold, the plugin MUST warn the user before or during submit but MUST still allow the upload unless a separate hard failure occurs (network, auth, or service rejection).
- **FR-017**: Local destination training MUST remain fully usable without any cloud token configured.
- **FR-018**: The cloud service MUST retain uploaded training audio for each job’s corpus for a **30-day** window measured from **last use**, and MUST **extend (reset)** that window whenever the retained corpus is used again for a product-defined reuse action (at minimum: starting another cloud training job from that retained corpus).
- **FR-019**: After the retention window expires without use, the service MUST delete the retained corpus audio; subsequent cloud jobs MUST require a fresh upload of selected library audio from the plugin.

### Key Entities

- **Cloud API Token**: Secret credential stored locally after user entry; authorizes cloud job actions for beta/developer access; not a spendable credit balance.
- **Train Destination**: User choice on the Train panel — Local (on-device worker) or Cloud (remote job).
- **Cloud Training Job**: Remote execution of a training recipe identified by a job id; holds status, progress, objective, owning token identity, and links to checkpoints and final artifact.
- **Job Package**: Snapshot of armed graph, selected library material, and train options captured at submit time.
- **Retained Cloud Corpus**: Server-side copy of uploaded training audio tied to a job/token context; subject to a **30-day from last use** retention policy with extension on reuse; deleted after idle expiry.
- **Cloud Checkpoint**: Intermediate trainable result published by a job; downloadable for optional local listen/load.
- **Final Cloud Artifact**: Successful job output suitable for Gold auto-load (same role as a successful local train artifact).
- **Job Monitor Session**: Local UI binding between a plugin instance and a cloud job id for status updates, control requests, and downloads; rediscovery is **token-scoped** (not limited to the submitting machine).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user with a valid token can go from an already-valid local train setup to a successfully submitted cloud job in under 2 minutes without leaving the plugin.
- **SC-002**: During cloud training, continuous DAW playback shows no cloud-induced audio dropouts attributable to submit/monitor/download, and the audible model stays the pre-job model until intentional load.
- **SC-003**: While online, status or progress updates for a running cloud job appear in the Train panel at least once every 5 seconds on average during active running state.
- **SC-004**: In guided tests, Pause then Resume succeeds for at least 95% of attempts on a running cloud job; Stop ends the job without success auto-load in 100% of Stop tests.
- **SC-005**: After plugin restart with network and valid token, a user can reattach to an in-flight job and see correct status within 1 minute without resubmitting.
- **SC-005a**: On a second machine with the same valid token, a user can discover and control an in-flight job started elsewhere under that token within 1 minute while online, without resubmitting.
- **SC-006**: On successful cloud completion, on the **submitting** instance, final artifact download and Gold auto-load complete so the user can hear the trained result within 2 minutes of success notification on a typical broadband connection for a single-model artifact (excluding extreme corpus sizes).
- **SC-006a**: On a non-submitting instance with the same token, success does not auto-swap the live graph; manual download/load remains available in guided tests.
- **SC-007**: At least 90% of first-time guided testers correctly distinguish Local vs Cloud destination and complete one cloud submit without attempting a credits purchase flow.
- **SC-008**: Invalid token, missing copyright acknowledgment, and ineligible library selection each produce a distinct blocking message before upload of training audio begins in 100% of negative tests.
- **SC-009**: Optional checkpoint download and load during a run leaves the remote job running in 100% of tests where the job had not yet finished.
- **SC-010**: Users who never configure a token can still complete a local Train success path with no cloud dependency.
- **SC-011**: In guided tests, a retained cloud corpus remains available for reuse within 30 days of last use; after a simulated idle expiry past 30 days without use, a re-run attempt requires a fresh upload and does not silently reuse deleted server audio.
- **SC-013**: When selected corpus size exceeds the soft threshold in guided tests, the user sees a warning and can still proceed with upload; the UI does not hard-block solely due to that threshold.

## Assumptions

- Phase 3 local training (unified Train panel, Library, Capture, mapping and reconstruction objectives, copyright gate, non-blocking audio, success Gold auto-load, hear-while-training checkpoints) is the baseline this feature extends.
- Constitution Phase 4 “Cloud Training” intent applies: optional remote training that returns usable trained artifacts into the VST; monetization via WordPress credits is a **later** feature.
- A remote cloud training service exists (or will be stood up alongside this feature) that can execute the same training recipes as the local worker for mapping and reconstruction; exact hosting vendor is an implementation concern outside this specification.
- API tokens are issued out-of-band for developers/beta users (email, dashboard, or manual provisioning)—token issuance UI inside the plugin is out of scope.
- No credit balance is checked or decremented in this slice; the service enforces **one active cloud job per token** as the fair-use concurrency limit for now.
- Network connectivity is required for submit, monitor, control, and download; offline users use Local destination.
- Users remain responsible for copyright of training material; acknowledgment stays local and still gates Cloud Run; this slice does not upload the acknowledgment record as a legal substitute for that local gate.
- Transfer of selected library audio to the cloud is required for remote training; users accept that selected corpus leaves the machine. Server-side copies are retained **30 days from last use**, with the window **reset on each qualifying reuse**; idle expiry deletes the corpus (not indefinite storage). Storage cost is accepted as an operational concern for beta; credit-based billing for retention is deferred with WordPress credits. Large selections trigger a **soft size warning** but are not hard-capped in this slice; the concrete warning threshold is set during planning/ops.
- One active train job per plugin instance; **one active cloud job per token** globally (stricter of the two applies for cloud).
- Cloud job monitor/control/download is **token-wide** across machines; submitting machine is not privileged for monitor/control, but **success Gold auto-load is submitter-only** (others use manual download/load).
- Marketplace, model selling, and in-VST storefront remain out of scope.
- WordPress website integration for accounts and credit top-up is explicitly deferred to a follow-on feature after cloud job plumbing works end to end.
