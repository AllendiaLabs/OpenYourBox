---
description: "Task list for Cloud Training"
---

# Tasks: Cloud Training

**Input**: Design documents from `specs/017-cloud-training/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Tests**: Spec does not require TDD. Include targeted C++/Python tests where they reduce auth, concurrency, retention, and client-mapping risk; validate via `quickstart.md`.

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

- [ ] T001 Create `CloudService/` layout (`api/`, `worker/`, `tests/`) and README noting proprietary Phase 4 backend per `specs/017-cloud-training/plan.md`
- [ ] T002 [P] Add Python deps stub for the cloud API (web framework + test extras) in `CloudService/requirements.txt`
- [ ] T003 [P] Add new plug-in sources `CloudSettings`, `CloudTrainClient`, and job-package helpers to the OpenYourBox target in `CMakeLists.txt`
- [ ] T004 Confirm design artifact paths and contracts are linked from `specs/017-cloud-training/plan.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared types, settings persistence, destination enum, HTTP client skeleton, and API auth/job state machine required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T005 Define `TrainDestination` (`local`|`cloud`), soft-warn constant default `2147483648`, and cloud error-code string helpers in `OpenYourBox/Source/graph/GraphTypes.h`
- [ ] T006 [P] Implement `CloudSettings` (save/clear/replace masked token, optional base URL override, PropertiesFile persistence) in `OpenYourBox/Source/train/CloudSettings.h` and `CloudSettings.cpp`
- [ ] T007 [P] Scaffold `CloudTrainClient` with off-audio-thread HTTPS request hooks and Bearer header injection in `OpenYourBox/Source/train/CloudTrainClient.h` and `CloudTrainClient.cpp`
- [ ] T008 Extend `TrainCoordinator` with destination-aware busy gate (one active local-or-cloud job per instance) without changing local ChildProcess path in `OpenYourBox/Source/train/TrainCoordinator.h` and `TrainCoordinator.cpp`
- [ ] T009 Implement Bearer token auth middleware and standard `{error_code,error_message}` responses in `CloudService/api/auth.py`
- [ ] T010 [P] Implement in-memory (or file-backed) job store with statuses `queued|running|paused|succeeded|failed|stopped` and one-active-job-per-token rule in `CloudService/api/jobs.py`
- [ ] T011 [P] Add `GET /v1/health` and token-gated router skeleton matching `specs/017-cloud-training/contracts/cloud-train-api.md` in `CloudService/api/app.py`
- [ ] T012 Persist submitter map (`jobId` → `isSubmitter`) for the local instance in `OpenYourBox/Source/train/CloudSettings.cpp` (or adjacent state helper)
- [ ] T013 Wire `CloudSettings` + default cloud base URL into processor/editor lifecycle in `OpenYourBox/Source/PluginProcessor.cpp` and `OpenYourBox/Source/PluginEditor.cpp`

**Checkpoint**: Token storage, client skeleton, coordinator busy gate, and cloud API auth/job store ready — story work can begin.

---

## Phase 3: User Story 1 - Configure Cloud Access Once (Priority: P1) 🎯 MVP

**Goal**: In-plugin settings to enter, save, clear, and replace a masked API token with clear configured / not-configured / auth-error status; no separate app.

**Independent Test**: Quickstart scenario 1 — save token (masked), clear disables cloud, invalid token shows auth error on probe/action.

### Implementation for User Story 1

- [ ] T014 [P] [US1] Add cloud settings UI (token field, Save, Clear, masked display, status line) in `OpenYourBox/Source/ui/` (settings/prefs panel used by the editor)
- [ ] T015 [US1] Connect settings UI to `CloudSettings` load/save/clear in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T016 [US1] Add optional “Test connection” / auth probe via `CloudTrainClient` against `GET /v1/jobs` (or health+auth) in `OpenYourBox/Source/train/CloudTrainClient.cpp`
- [ ] T017 [US1] Surface `unauthorized` as a clear authentication error in the settings/Train status without starting a job in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T018 [P] [US1] Ensure no WordPress/credits affordances appear in cloud settings copy in `OpenYourBox/Source/ui/` cloud settings panel (same file as T014)

**Checkpoint**: US1 complete — token configure/clear/auth-error works independently.

---

## Phase 4: User Story 2 - Start a Cloud Job From the Same Train Panel (Priority: P1)

**Goal**: Local|Cloud destination; Cloud Run packages graph + corpus, uploads with progress, respects copyright/library gates, soft 2 GiB warn, and one-job-per-token; local path unchanged.

**Independent Test**: Quickstart scenarios 2–4 — Cloud submit → queued/running; soft warn; second submit under same token refused; audio stays on pre-job model.

### Implementation for User Story 2

- [ ] T019 [P] [US2] Add Local|Cloud destination control and last-used preference (default Local) in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T020 [US2] Gate Cloud Run on token configured + existing copyright/library/objective eligibility before any upload in `OpenYourBox/Source/ui/TrainPanel.cpp` and `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [ ] T021 [P] [US2] Compute selected corpus byte sum and show soft warning above 2 GiB without blocking in `OpenYourBox/Source/library/` helper and `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T022 [US2] Build cloud `manifest.json` + multipart file parts (or archive) per `specs/017-cloud-training/contracts/cloud-job-package.md` in `OpenYourBox/Source/train/` (job package builder beside `TrainCoordinator`)
- [ ] T023 [US2] Implement `POST /v1/jobs` multipart ingest, corpus storage, `corpus_id` assignment, and enqueue in `CloudService/api/jobs.py`
- [ ] T024 [US2] Return `409` + `one_job_per_token` when an active job exists for the token in `CloudService/api/jobs.py`
- [ ] T025 [US2] Submit package via `CloudTrainClient` with upload progress callbacks (message thread only) in `OpenYourBox/Source/train/CloudTrainClient.cpp`
- [ ] T026 [US2] Fork `TrainCoordinator::start` for `destination=cloud` (package → upload → mark submitter) while keeping local ChildProcess path intact in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [ ] T027 [US2] Map submit success to Train panel busy/queued state and show upload progress in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T028 [US2] Map API errors (`unauthorized`, `one_job_per_token`, `validation_failed`, `capacity`) to distinct user-readable messages in `OpenYourBox/Source/train/CloudTrainClient.cpp` and `TrainPanel.cpp`
- [ ] T029 [P] [US2] Add mock/CPU worker that accepts a queued job and transitions `queued→running` for dev in `CloudService/worker/mock_worker.py`
- [ ] T030 [P] [US2] Add Python tests for auth reject and one-job-per-token in `CloudService/tests/test_jobs_concurrency.py`

**Checkpoint**: US2 complete — cloud submit works; local Train unaffected.

---

## Phase 5: User Story 3 - Monitor Progress and Control a Remote Job (Priority: P1)

**Goal**: Poll status/loss/stage; Pause/Resume/Stop; offline/reconnect UX; token-wide list/attach across machines; plugin restart rediscovery.

**Independent Test**: Quickstart scenarios 5–6 and 10 — updates while online; pause/resume/stop; attach from second machine; restart reattach; offline does not imply cancel.

### Implementation for User Story 3

- [ ] T031 [P] [US3] Implement `GET /v1/jobs` and `GET /v1/jobs/{job_id}` progress payloads (status, step, loss, stage) in `CloudService/api/jobs.py`
- [ ] T032 [P] [US3] Implement `POST .../pause|resume|stop` with legal state transitions in `CloudService/api/jobs.py`
- [ ] T033 [US3] Drive mock/real worker to emit step/loss/stage updates while running in `CloudService/worker/mock_worker.py` (and later real worker)
- [ ] T034 [US3] Poll job detail on a timer from `CloudTrainClient` / coordinator (≥1 update / 5 s target) off the audio thread in `OpenYourBox/Source/train/CloudTrainClient.cpp`
- [ ] T035 [US3] Map remote statuses into `TrainStatus` + status message (including `queued` and offline/reconnect) in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [ ] T036 [US3] Wire Train panel Pause/Resume/Stop to cloud control endpoints when destination is cloud in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T037 [US3] Ensure Stop yields terminal `stopped` with no success auto-load path invoked in `OpenYourBox/Source/train/TrainCoordinator.cpp` and `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T038 [US3] On editor open with token, call `GET /v1/jobs` and offer attach to active/recent job in `OpenYourBox/Source/PluginEditor.cpp` and `TrainPanel.cpp`
- [ ] T039 [US3] Preserve `isSubmitter` only for locally recorded submitter jobs during rediscovery in `OpenYourBox/Source/train/CloudSettings.cpp`
- [ ] T040 [P] [US3] Add Python tests for pause/resume/stop state machine in `CloudService/tests/test_job_controls.py`

**Checkpoint**: US3 complete — monitor/control/rediscovery works token-wide.

---

## Phase 6: User Story 4 - Retrieve Checkpoints and Load Successful Results (Priority: P1)

**Goal**: List/download checkpoints; optional mid-run load; submitter success auto-load Gold; non-submitter manual download/load only; retryable download failure.

**Independent Test**: Quickstart scenario 7 — checkpoint download during run; submitter auto-load on success; non-submitter no auto-swap.

### Implementation for User Story 4

- [ ] T041 [P] [US4] Publish checkpoint metadata from worker and expose `GET /v1/jobs/{id}/checkpoints` plus download URL endpoint in `CloudService/api/artifacts.py`
- [ ] T042 [P] [US4] Publish final artifact on `succeeded` and expose artifact download endpoint in `CloudService/api/artifacts.py`
- [ ] T043 [US4] List remote checkpoints in Train UI and download to local train artifact dir via `CloudTrainClient` in `OpenYourBox/Source/ui/TrainPanel.cpp` and `CloudTrainClient.cpp`
- [ ] T044 [US4] Optional load of downloaded checkpoint into live path without stopping the remote job (reuse hear-while-training path) in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T045 [US4] On submitter + `succeeded`, download final artifact and auto-load Gold using existing local success path in `OpenYourBox/Source/train/TrainCoordinator.cpp` and `PluginEditor.cpp`
- [ ] T046 [US4] On non-submitter + `succeeded`, show success and enable manual Download/Load only (no graph auto-swap) in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T047 [US4] On download failure after server success, show retryable error and leave live graph unchanged in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [ ] T048 [US4] Wire real GPU worker to materialize package and invoke mapping/reconstruction recipes compatible with `Backend/train_worker.py` in `CloudService/worker/train_runner.py`
- [ ] T049 [P] [US4] Add Python tests for checkpoint list and success-only artifact availability in `CloudService/tests/test_artifacts.py`

**Checkpoint**: US4 complete — checkpoints and submitter-only Gold auto-load work.

---

## Phase 7: User Story 5 - Understand Limits Before Credits Exist (Priority: P2)

**Goal**: No credits/WordPress purchase path; clear capacity/concurrency messaging; Local remains available; retention 30-day extend-on-use + corpus reuse.

**Independent Test**: Quickstart scenarios 8–9 — no credits UI; corpus reuse within window; expiry requires re-upload; concurrency message clear.

### Implementation for User Story 5

- [ ] T050 [P] [US5] Audit Train/settings UI strings to ensure no add-credits / WordPress storefront path in `OpenYourBox/Source/ui/TrainPanel.cpp` and cloud settings UI
- [ ] T051 [US5] Add optional beta copy that paid credits come later (informational only) in `OpenYourBox/Source/ui/TrainPanel.cpp` and cloud settings UI under `OpenYourBox/Source/ui/`
- [ ] T052 [US5] Implement corpus `last_used_at` + 30-day expiry and extend-on-reuse when `POST /v1/jobs` references `corpus_id` in `CloudService/api/corpus.py`
- [ ] T053 [US5] Add retention sweeper deleting expired corpora not pinned by active jobs in `CloudService/api/retention.py`
- [ ] T054 [US5] Support plugin submit with `corpus_id` reuse (skip full re-upload) in `OpenYourBox/Source/train/` job package builder and `OpenYourBox/Source/train/CloudTrainClient.cpp`
- [ ] T055 [US5] Map `corpus_expired` and `capacity` to clear actionable Train messages in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T056 [P] [US5] Add Python tests for retention extend and idle expiry in `CloudService/tests/test_retention.py`

**Checkpoint**: US5 complete — beta limits and retention behavior validated; still no billing UI.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Build wiring, security hygiene, and end-to-end quickstart validation.

- [ ] T057 [P] Register C++ soft-warn / destination gating tests in `Tests/CloudTrainClientTests.cpp` (or equivalent) and CMake
- [ ] T058 [P] Ensure tokens never logged in full in `OpenYourBox/Source/train/CloudTrainClient.cpp` and `CloudService/api/auth.py`
- [ ] T059 Verify TLS/base-URL override path for staging in `OpenYourBox/Source/train/CloudSettings.cpp`
- [ ] T060 Confirm local Train path (no token) still succeeds end-to-end in `OpenYourBox/Source/train/TrainCoordinator.cpp` / DAW
- [ ] T061 Run `specs/017-cloud-training/quickstart.md` scenarios 1–10 and record results in `specs/017-cloud-training/checklists/requirements.md`
- [ ] T062 Update `specs/017-cloud-training/plan.md` status to implementation-ready / tasks complete note after T061

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: After Foundational — MVP (token settings)
- **US2 (Phase 4)**: After Foundational; practically needs US1 token UX for Cloud Run
- **US3 (Phase 5)**: Needs US2 submit (a job to monitor)
- **US4 (Phase 6)**: Needs US3 progress/control shell
- **US5 (Phase 7)**: Can overlap late US2+ for retention API; UI audit can run parallel after US1
- **Polish (Phase 8)**: After desired stories complete

### User Story Dependencies

| Story | Depends on | Independently testable deliverable |
|-------|------------|--------------------------------------|
| US1 | Foundation | Token save/clear/auth error |
| US2 | Foundation (+ US1 for real token UX) | Cloud submit + soft warn + concurrency refuse |
| US3 | US2 job exists | Poll, pause/resume/stop, rediscovery |
| US4 | US3 | Checkpoints + submitter auto-load |
| US5 | US1 UI + US2 API | No credits UI + retention reuse/expiry |

### Parallel Opportunities

- T002 ∥ T003 ∥ T004 (setup)
- T006 ∥ T007 ∥ T009–T011 (foundation client vs API)
- T014 ∥ T018 (US1 UI copy vs controls)
- T019 ∥ T021 ∥ T029 ∥ T030 (US2 UI/warn vs mock worker/tests)
- T031 ∥ T032 ∥ T040 (US3 API/tests)
- T041 ∥ T042 ∥ T049 (US4 artifacts/tests)
- T050 ∥ T056 (US5 audit vs retention tests)
- T057 ∥ T058 (polish)

---

## Parallel Example: User Story 2

```bash
# After foundational + US1 token path:
Task: "Add Local|Cloud destination in OpenYourBox/Source/ui/TrainPanel.cpp"
Task: "Soft-warn byte sum in library helper + TrainPanel.cpp"
Task: "Mock worker in CloudService/worker/mock_worker.py"
Task: "Concurrency tests in CloudService/tests/test_jobs_concurrency.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 Setup  
2. Complete Phase 2 Foundational  
3. Complete Phase 3 US1 (token settings)  
4. **STOP and VALIDATE** quickstart scenario 1  
5. Demo beta auth configuration inside the VST  

### Incremental Delivery

1. US1 → token configured  
2. US2 → first cloud submit (mock worker OK)  
3. US3 → monitor/control/rediscovery  
4. US4 → checkpoints + Gold auto-load + real worker  
5. US5 → retention + no-credits polish  
6. Phase 8 quickstart sign-off  

### Suggested MVP+ demo slice

**US1 + US2 + mock worker** proves cloud submit before full GPU worker and auto-load.

---

## Notes

- [P] = different files, no dependency on incomplete sibling tasks  
- Do not put HTTP/upload/poll on the audio thread  
- Local `Backend/train_worker.py` path must remain usable with no token  
- WordPress/credits/marketplace are explicitly out of scope — do not add tasks for them  
- Next command: `/speckit-implement`
