---
description: "Task list for Cloud Training"
---

# Tasks: Cloud Training

**Input**: Design documents from `specs/017-cloud-training/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Tests**: Spec does not require TDD. Include targeted C++/Python tests where they reduce auth, entitlement, concurrency, retention, and client-mapping risk; validate via `quickstart.md`.

**Organization**: Tasks grouped by user story for independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`US1`–`US5`) on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in train/UI: `OpenYourBox/Source/train/`, `OpenYourBox/Source/ui/`
- Library: `OpenYourBox/Source/library/`
- Editor/processor: `OpenYourBox/Source/PluginEditor.*`, `PluginProcessor.*`
- Local worker: `Backend/train_worker.py`
- Cloud service: `CloudService/`
- Tests: `Tests/`, `CloudService/tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Scaffold proprietary cloud service tree and wire new plug-in sources into the build.

- [X] T001 Create `CloudService/` layout (`api/`, `worker/`, `storefront/`, `tests/`) and README noting proprietary Phase 4 backend per `specs/017-cloud-training/plan.md`
- [X] T002 [P] Add Python deps stub for the cloud API (web framework + test extras) in `CloudService/requirements.txt`
- [X] T003 [P] Add new plug-in sources `CloudSettings`, `CloudTrainClient`, and job-package helpers to the OpenYourBox target in `CMakeLists.txt`
- [X] T004 Confirm design artifact paths and contracts are linked from `specs/017-cloud-training/plan.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared types, linked-session persistence, destination enum, HTTP client skeleton, account-auth middleware, entitlement stub, and job store required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Define `TrainDestination` (`local`|`cloud`), soft-warn constant default `2147483648`, and cloud error-code string helpers (`unauthorized`, `insufficient_entitlement`, `one_job_per_account`, …) in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T006 [P] Implement `CloudSettings` (masked linked session + optional refresh, disconnect, base URL override, storefront URL override, PropertiesFile persistence) in `OpenYourBox/Source/train/CloudSettings.h` and `CloudSettings.cpp`
- [X] T007 [P] Scaffold `CloudTrainClient` with off-audio-thread HTTPS request hooks and Bearer linked-session header injection in `OpenYourBox/Source/train/CloudTrainClient.h` and `CloudTrainClient.cpp`
- [X] T008 Extend `TrainCoordinator` with destination-aware busy gate (one active local-or-cloud job per instance) without changing local ChildProcess path in `OpenYourBox/Source/train/TrainCoordinator.h` and `TrainCoordinator.cpp`
- [X] T009 Implement linked-session Bearer auth middleware and standard `{error_code,error_message}` responses in `CloudService/api/auth.py`
- [X] T010 [P] Implement mock entitlement provider (`sufficient` flag + `balance_hint`) with storefront sync stub in `CloudService/storefront/entitlement.py`
- [X] T011 [P] Implement in-memory (or file-backed) job store with statuses `queued|running|paused|succeeded|failed|stopped` and one-active-job-per-account rule in `CloudService/api/jobs.py`
- [X] T012 Add `GET /v1/health`, auth/link router stubs, and session-gated router skeleton matching `specs/017-cloud-training/contracts/cloud-train-api.md` in `CloudService/api/app.py`
- [X] T013 Persist submitter map (`jobId` → `isSubmitter`) for the local instance in `OpenYourBox/Source/train/CloudSettings.cpp` (or adjacent state helper)
- [X] T014 Wire `CloudSettings` + default cloud API base URL + default storefront URL into processor/editor lifecycle in `OpenYourBox/Source/PluginProcessor.cpp` and `OpenYourBox/Source/PluginEditor.cpp`

**Checkpoint**: Linked-session storage, client skeleton, coordinator busy gate, cloud API auth/entitlement/job store ready — story work can begin.

---

## Phase 3: User Story 1 - Link Platform Customer Account (Priority: P1) 🎯 MVP

**Goal**: In-plugin settings to link/disconnect a platform customer account via storefront device-code (or equivalent) flow, masked credentials, clear Linked / Not linked / Auth error status, and Open storefront affordance; no in-plugin checkout.

**Independent Test**: Quickstart scenario 1 — link (masked), disconnect disables cloud, revoked session shows auth error; Open storefront launches external URL.

### Implementation for User Story 1

- [X] T015 [P] [US1] Implement `POST /v1/auth/link/start` and `POST /v1/auth/link/token` (device-code style) in `CloudService/api/auth_link.py` wired from `CloudService/api/app.py`
- [X] T016 [P] [US1] Add mock storefront verification page/handler for link completion in `CloudService/storefront/link_mock.py` (dev/staging)
- [X] T017 [US1] Implement link-start, poll-for-token, refresh (optional), and logout client methods in `OpenYourBox/Source/train/CloudTrainClient.cpp`
- [X] T018 [P] [US1] Add cloud settings UI (Link account, Disconnect, masked status, Open storefront, optional advanced URLs) in `OpenYourBox/Source/ui/` settings/prefs panel used by the editor
- [X] T019 [US1] Connect settings UI to `CloudSettings` + link flow (show user code, open `verification_url`, poll until linked/expired/cancelled) in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T020 [US1] Surface `unauthorized` / `link_expired` as clear authentication errors in settings/Train status without starting a job in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T021 [US1] Ensure no in-plugin checkout/cart/payment UI exists; only Open storefront URL launch in `OpenYourBox/Source/ui/` cloud settings panel (same file as T018)

**Checkpoint**: US1 complete — account link/disconnect/auth-error/open-storefront works independently.

---

## Phase 4: User Story 2 - Start a Cloud Job From the Same Train Panel (Priority: P1)

**Goal**: Destination Local|Cloud on Train panel; Cloud Run packages graph + corpus, gates on linked account + entitlement, submits job; soft size warn; Local unchanged without account.

**Independent Test**: Quickstart scenarios 2–5 — entitlement refuse, successful submit with progress, soft warn, one-job-per-account, Local without link (SC-010).

### Implementation for User Story 2

- [X] T022 [P] [US2] Implement `GET /v1/entitlement` using storefront entitlement provider in `CloudService/api/entitlement_routes.py` (wired in `CloudService/api/app.py`)
- [X] T023 [US2] Implement `POST /v1/jobs` with gates auth → entitlement → one-job-per-account → validation and multipart/corpus_id ingest per `specs/017-cloud-training/contracts/cloud-job-package.md` in `CloudService/api/jobs.py`
- [X] T024 [P] [US2] Add mock/CPU worker that accepts queued jobs and advances status for submit smoke tests in `CloudService/worker/mock_worker.py`
- [X] T025 [P] [US2] Build job-package assembler (manifest + selected library files / soft byte sum) in `OpenYourBox/Source/train/CloudJobPackage.h` and `CloudJobPackage.cpp`
- [X] T026 [US2] Add Local|Cloud destination control and soft-upload warning to Train panel in `OpenYourBox/Source/ui/TrainPanel.h` and `TrainPanel.cpp`
- [X] T027 [US2] Fork Cloud Run in `TrainCoordinator` (package → upload → accept; refuse when unlinked or entitlement insufficient with distinct messages) in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T028 [US2] Probe entitlement via `CloudTrainClient` before/at Cloud Run and map `insufficient_entitlement` / `one_job_per_account` errors in `OpenYourBox/Source/train/CloudTrainClient.cpp`
- [X] T029 [US2] Keep Local path requiring zero platform account; verify Cloud disabled when disconnected in `OpenYourBox/Source/ui/TrainPanel.cpp` and `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T030 [US2] Show packaging/upload progress on Cloud submit without touching the audio thread in `OpenYourBox/Source/ui/TrainPanel.cpp` and `OpenYourBox/Source/train/CloudTrainClient.cpp`
- [X] T031 [P] [US2] Add Python tests for entitlement gate + one-job-per-account in `CloudService/tests/test_jobs_gates.py` (or `Tests/test_cloud_api.py`)

**Checkpoint**: US2 complete — Cloud submit works with account+entitlement; Local works without account.

---

## Phase 5: User Story 3 - Monitor Progress and Control a Remote Job (Priority: P1)

**Goal**: Poll status/metrics in Train panel; Pause/Resume/Stop; rediscovery and account-wide control across machines; offline/reconnect without implying cancel.

**Independent Test**: Quickstart scenarios 6–7, 11 — status updates, pause/resume/stop, second-machine attach, restart rediscovery, offline banner.

### Implementation for User Story 3

- [X] T032 [P] [US3] Implement `GET /v1/jobs`, `GET /v1/jobs/{id}`, and `POST .../pause|resume|stop` in `CloudService/api/jobs.py`
- [X] T033 [US3] Drive mock worker progress fields (`step`, `loss`, `stage`) while running in `CloudService/worker/mock_worker.py`
- [X] T034 [US3] Add background poll timer and status mapping into existing `TrainStatus` / panel messages in `OpenYourBox/Source/train/CloudTrainClient.cpp` and `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T035 [US3] Wire Pause/Resume/Stop for cloud destination to remote control endpoints in `OpenYourBox/Source/ui/TrainPanel.cpp` and `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T036 [US3] On plugin open with linked account, list/attach active or recent jobs (account-wide) in `OpenYourBox/Source/train/CloudTrainClient.cpp` and `OpenYourBox/Source/PluginEditor.cpp`
- [X] T037 [US3] Show offline/reconnect UI when poll fails without claiming job cancelled in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T038 [P] [US3] Add Python tests for pause/resume/stop state machine and illegal transitions in `CloudService/tests/test_job_control.py`

**Checkpoint**: US3 complete — monitor/control/rediscovery works with linked account.

---

## Phase 6: User Story 4 - Retrieve Checkpoints and Load Successful Results (Priority: P1)

**Goal**: List/download checkpoints during run; submitter success auto-load Gold; non-submitter manual download/load only; Stop/fail never auto-load.

**Independent Test**: Quickstart scenario 8 — mid-run checkpoint load; submitter auto-load; non-submitter no auto-swap.

### Implementation for User Story 4

- [X] T039 [P] [US4] Implement checkpoint list + signed download and final artifact download endpoints in `CloudService/api/artifacts.py` (wired in `CloudService/api/app.py`)
- [X] T040 [US4] Publish intermediate checkpoints and final artifact from mock/real worker on success only in `CloudService/worker/mock_worker.py` (and later GPU worker)
- [X] T041 [US4] List/download checkpoints to local artifact dirs and optional live load (hear-while-training) in `OpenYourBox/Source/train/CloudTrainClient.cpp` and `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T042 [US4] On `succeeded`, auto-download + Gold auto-load **only if** local `isSubmitter` for that `jobId` in `OpenYourBox/Source/train/TrainCoordinator.cpp` and `OpenYourBox/Source/PluginProcessor.cpp`
- [X] T043 [US4] Non-submitter success: show Download/Load actions without auto graph swap in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T044 [US4] Ensure `stopped`/`failed` never trigger success auto-load; retryable download error after server success in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T045 [P] [US4] Add C++ unit tests for submitter-only auto-load flag and soft-warn byte math in `Tests/CloudTrainClientTests.cpp`

**Checkpoint**: US4 complete — checkpoints + submitter-only Gold auto-load.

---

## Phase 7: User Story 5 - Entitlement Visibility and Storefront Path (Priority: P2)

**Goal**: High-level entitlement status in settings/Train; clear path to Open storefront when entitlement missing; concurrency messaging remains clear; no marketplace.

**Independent Test**: Quickstart scenario 2 + US5 acceptance — insufficient entitlement shows storefront path; capacity/concurrency messages actionable.

### Implementation for User Story 5

- [X] T046 [P] [US5] Surface entitlement status (available / unavailable / unknown) from last probe in cloud settings and Train panel in `OpenYourBox/Source/ui/TrainPanel.cpp` and settings panel under `OpenYourBox/Source/ui/`
- [X] T047 [US5] On entitlement refuse, show actionable copy + Open storefront button/link in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T048 [US5] Refresh entitlement probe on settings open and before Cloud Run in `OpenYourBox/Source/train/CloudTrainClient.cpp` and `OpenYourBox/Source/PluginEditor.cpp`
- [X] T049 [US5] Confirm marketplace browsing/selling remains absent from Train/settings surfaces in `OpenYourBox/Source/ui/TrainPanel.cpp` and settings panel

**Checkpoint**: US5 complete — entitlement visibility + storefront path without in-plugin checkout.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Retention, real GPU worker wiring, security hardening, and quickstart validation across stories.

- [X] T050 Implement corpus retention sweeper (30-day from `last_used_at`, extend on `corpus_id` reuse) in `CloudService/api/retention.py`
- [X] T051 [P] Add retention/reuse/expiry tests in `CloudService/tests/test_retention.py`
- [X] T052 Wire GPU (or shared-recipe) worker to local train recipes from `Backend/train_worker.py` in `CloudService/worker/train_runner.py`
- [X] T053 [P] Document staging env vars (API URL, storefront URL, mock entitlement) in `CloudService/README.md`
- [X] T054 Security pass: TLS assumptions, no secrets in query strings/logs, signed download URLs in `CloudService/api/` and `OpenYourBox/Source/train/CloudTrainClient.cpp`
- [X] T055 Run through `specs/017-cloud-training/quickstart.md` scenarios 1–11 and record gaps in `specs/017-cloud-training/checklists/requirements.md` notes (or adjacent QA note)
- [X] T056 Confirm audio-thread rules: no HTTP/pack/link waits on `processBlock` in `OpenYourBox/Source/PluginProcessor.cpp` and train client call sites

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: After Foundational — MVP account link
- **US2 (Phase 4)**: After US1 link + Foundational entitlement/job APIs (submit needs linked session)
- **US3 (Phase 5)**: After US2 (needs an accepted job to monitor); control APIs can be stubbed earlier
- **US4 (Phase 6)**: After US3 progress path (checkpoints during/after run)
- **US5 (Phase 7)**: After US1 + entitlement probe from US2; can partially overlap US2 UI
- **Polish (Phase 8)**: After desired stories complete

### User Story Dependencies

- **US1 (P1)**: After Foundational — no dependency on other stories
- **US2 (P1)**: Needs US1 link flow for real submits; mock linked session acceptable for API-only testing
- **US3 (P1)**: Needs US2 submit (or seeded job) for end-to-end monitor tests
- **US4 (P1)**: Needs US3 running job + artifact endpoints
- **US5 (P2)**: Needs US1 Open storefront + US2 entitlement probe; independently testable as UX polish

### Parallel Opportunities

- Phase 1: T002 ∥ T003
- Phase 2: T006 ∥ T007; T010 ∥ T011
- US1: T015 ∥ T016 ∥ T018
- US2: T022 ∥ T024 ∥ T025; T031 after gates exist
- US3: T032 ∥ early mock progress; T038 after control API
- US4: T039 ∥ T045
- US5: T046 can start once entitlement probe exists
- Polish: T051 ∥ T053

### Parallel Example: User Story 1

```bash
# After Foundational:
Task: "Implement POST /v1/auth/link/start and token exchange in CloudService/api/auth_link.py"
Task: "Add mock storefront link completion in CloudService/storefront/link_mock.py"
Task: "Add cloud settings UI (Link/Disconnect/Open storefront) under OpenYourBox/Source/ui/"
```

### Parallel Example: User Story 2

```bash
Task: "GET /v1/entitlement in CloudService/api/entitlement_routes.py"
Task: "Mock/CPU worker in CloudService/worker/mock_worker.py"
Task: "CloudJobPackage assembler in OpenYourBox/Source/train/CloudJobPackage.cpp"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup  
2. Complete Phase 2: Foundational  
3. Complete Phase 3: US1 account link  
4. **STOP and VALIDATE**: Quickstart scenario 1  
5. Demo link/disconnect/open-storefront before cloud submit work

### Incremental Delivery

1. Setup + Foundational → foundation ready  
2. US1 → linked account MVP  
3. US2 → Cloud submit with entitlement  
4. US3 → monitor/control  
5. US4 → checkpoints + Gold auto-load  
6. US5 → entitlement UX polish  
7. Polish → retention, real worker, quickstart sign-off  

### Parallel Team Strategy

1. Team completes Setup + Foundational together  
2. Then:  
   - Dev A: US1 UI + client link  
   - Dev B: CloudService auth/link + entitlement + jobs  
   - Dev C: package builder + Train destination (US2 prep)  
3. Integrate on US2 submit, then US3–US5  

---

## Notes

- [P] = different files, no incomplete-task dependencies  
- [USn] maps to spec user stories  
- Local Train MUST remain usable with no linked account at every checkpoint after T029  
- Do not add in-plugin checkout; storefront owns purchases  
- Prior token-only `tasks.md` is superseded by this list  
- Commit after each task or logical group  
- Next command: `/speckit-implement`
