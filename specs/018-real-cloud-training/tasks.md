---
description: "Task list for Real Cloud Training"
---

# Tasks: Real Cloud Training

**Input**: Design documents from `specs/018-real-cloud-training/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Tests**: Spec does not require TDD. Include targeted Python/C++ tests where they reduce fake-success, Stop, crash→failed, and gate risk (per plan Testing + SC-008); validate via `quickstart.md`.

**Organization**: Tasks grouped by user story for independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`US1`–`US4`) on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in train/UI: `OpenYourBox/Source/train/`, `OpenYourBox/Source/ui/`
- Local recipes: `Backend/train_worker.py`
- Cloud service: `CloudService/`
- Tests: `Tests/`, `CloudService/tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Dependencies and docs so staging real-training can be installed and configured.

- [X] T001 Fill FastAPI/uvicorn/httpx/pytest (and related) dependencies in `CloudService/requirements.txt`
- [X] T002 [P] Update staging env docs: remove `CLOUD_MOCK_WORKER`; document real-only worker + `CLOUD_DATA_DIR` / `CLOUD_API_PUBLIC_URL` in `CloudService/README.md`
- [X] T003 [P] Align `specs/018-real-cloud-training/quickstart.md` setup commands with the README once deps are listed

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Runnable CloudService control plane (no fake train yet) + Run/Stop-only Train surface. Blocks all user stories.

**⚠️ CRITICAL**: No user story work begins until this phase is complete

- [X] T004 Update job model: drop product `paused` from active set; add `workerHeartbeatAt` / final-artifact honesty rules in `CloudService/api/state.py`
- [X] T005 [P] Implement Bearer session helpers and authenticated dependency in `CloudService/api/auth.py` (reuse `auth_link.py` / `state.py` sessions)
- [X] T006 [P] Implement retention sweeper helpers in `CloudService/api/retention.py` (30-day sliding; extend on corpus reuse)
- [X] T007 Implement job submit/list/get and **Stop-only** control (no pause/resume routes) in `CloudService/api/jobs.py` per `contracts/cloud-train-api.md`
- [X] T008 [P] Implement checkpoint list + signed/local download and final artifact download in `CloudService/api/artifacts.py`
- [X] T009 Wire routers (health, auth link, entitlement, jobs, artifacts, retention, storefront link) in `CloudService/api/app.py`
- [X] T010 Delete mock/fake advancement worker: empty or remove `CloudService/worker/mock_worker.py` and purge `CLOUD_MOCK_WORKER` / `MOCK_WORKER` imports from `CloudService/` and `CloudService/tests/`
- [X] T011 Provide pytest app client fixtures without fake-success worker in `CloudService/tests/conftest.py`
- [X] T012 Rewrite gate tests (entitlement, one-job-per-account) without mock worker ticks in `CloudService/tests/test_jobs_gates.py`
- [X] T013 Rewrite job-control tests for **Stop-only** (remove pause/resume) in `CloudService/tests/test_job_control.py`
- [X] T014 [P] Remove Pause/Resume UI; keep Run + Stop for Local and Cloud in `OpenYourBox/Source/ui/TrainPanel.cpp` and `OpenYourBox/Source/ui/TrainPanel.h`
- [X] T015 [P] Remove pause/resume cloud control verbs and local command writes from `OpenYourBox/Source/train/TrainCoordinator.cpp` and `OpenYourBox/Source/train/TrainCoordinator.h`
- [X] T016 [P] Restrict `CloudTrainClient` control API to stop (drop pause/resume) in `OpenYourBox/Source/train/CloudTrainClient.cpp` and `OpenYourBox/Source/train/CloudTrainClient.h`
- [X] T017 [P] Extend `Tests/CloudTrainClientTests.cpp` for Run/Stop-only expectations (no pause/resume client surface)

**Checkpoint**: API boots; auth/entitlement/one-job/stop contracts pass without fake train; Train UI is Run/Stop only

---

## Phase 3: User Story 1 - Cloud Job Produces a Real Trained Model (Priority: P1) 🎯 MVP

**Goal**: Accepted Cloud jobs run real mapping/reconstruction recipes and return Gold-loadable final artifacts with real progress.

**Independent Test**: Staging API + short Cloud mapping (then reconstruction) job → submitting instance Gold-loads a real artifact; Local still works without account.

### Implementation for User Story 1

- [X] T018 [US1] Implement package materialization (manifest + corpus files → work dir + `train_steerable` request) in `CloudService/worker/train_runner.py`
- [X] T019 [US1] Invoke shared `Backend/train_worker.py` recipes (mapping + reconstruction) from `CloudService/worker/train_runner.py` with CPU/accelerator device selection
- [X] T020 [US1] Mirror recipe progress events (step, loss, stage, device) into the job store from `CloudService/worker/train_runner.py` for `GET /v1/jobs/{id}`
- [X] T021 [US1] On recipe success, register a non-empty final artifact and set `succeeded` only when real bytes exist in `CloudService/worker/train_runner.py` + `CloudService/api/artifacts.py`
- [X] T022 [US1] On recipe failure, set `failed` with readable message and never publish a dummy success artifact in `CloudService/worker/train_runner.py`
- [X] T023 [US1] Claim queued jobs and start `train_runner` from the API/worker supervisor path in `CloudService/api/jobs.py` (and/or `CloudService/worker/train_runner.py` entry)
- [X] T024 [US1] Add short real-train smoke (tiny wavs, small `total_steps`) for mapping success in `CloudService/tests/test_real_train_mapping.py`
- [X] T025 [P] [US1] Add short real-train smoke for reconstruction success in `CloudService/tests/test_real_train_reconstruction.py`

**Checkpoint**: US1 MVP — real Cloud success → downloadable/loadable artifact; progress reflects real steps

---

## Phase 4: User Story 2 - Checkpoints Are Real Mid-Run Artifacts (Priority: P1)

**Goal**: Mid-run checkpoints are real exports; download/optional load while job runs; Stop does not auto-load success.

**Independent Test**: Short Cloud job with checkpoint interval → list/download checkpoint while running; Stop → no Gold auto-load; checkpoints may remain.

### Implementation for User Story 2

- [X] T026 [US2] Publish real intermediate checkpoints from recipe export events into the artifact store in `CloudService/worker/train_runner.py`
- [X] T027 [US2] Ensure checkpoint list/download returns real bytes (not empty placeholders) in `CloudService/api/artifacts.py`
- [X] T028 [US2] Honor cooperative Stop: terminal `stopped`, no final success artifact, keep prior checkpoints in `CloudService/worker/train_runner.py` and `CloudService/api/jobs.py`
- [X] T029 [US2] Add checkpoint + Stop mid-run tests in `CloudService/tests/test_checkpoints_stop.py`
- [X] T030 [P] [US2] Confirm plugin checkpoint list/download path still works against real ids in `OpenYourBox/Source/train/CloudTrainClient.cpp` (adjust parsing only if needed)

**Checkpoint**: US2 — real checkpoints downloadable during run; Stop clean

---

## Phase 5: User Story 3 - Operators Can Run Real Training on Staging (Priority: P1)

**Goal**: Operators can run staging E2E with plugin overrides; no mock-success path remains; contract tests do not depend on fake advancement.

**Independent Test**: Follow `quickstart.md` setup; point `cloud.xml` at staging; complete one real Cloud success; confirm no `CLOUD_MOCK_WORKER` fake-success code.

### Implementation for User Story 3

- [X] T031 [US3] Document `cloud.xml` `apiBaseUrlOverride` / `storefrontUrlOverride` staging pointer steps in `CloudService/README.md` and keep in sync with `specs/018-real-cloud-training/quickstart.md`
- [X] T032 [US3] Add retention reuse/expiry tests without mock worker in `CloudService/tests/test_retention.py`
- [X] T033 [US3] Grep/assert absence of fake-success advancement APIs in `CloudService/tests/test_no_mock_worker.py` (or equivalent assertion in existing suite)
- [X] T034 [P] [US3] Verify storefront staging link helpers do not fabricate train success in `CloudService/storefront/link_mock.py` (link-only; rename/comment if needed to avoid “mock train” confusion)

**Checkpoint**: US3 — staging docs + tests prove real-only service operable without public production host

---

## Phase 6: User Story 4 - Parity and Failure Honesty (Priority: P2)

**Goal**: Failures and worker loss are honest; one-job-per-account still holds; Local ungated; no false `succeeded`.

**Independent Test**: Induce fail / kill worker → `failed` + downloadable checkpoints if any; second Cloud submit while active → refused; Local train without account still succeeds.

### Implementation for User Story 4

- [X] T035 [US4] Implement worker heartbeat updates during training in `CloudService/worker/train_runner.py`
- [X] T036 [US4] Implement reconciler: stale heartbeat / dead process → `failed` + `worker_lost`, clear active slot, keep checkpoints in `CloudService/api/jobs.py` and/or `CloudService/api/state.py`
- [X] T037 [US4] Add crash/reconciler tests in `CloudService/tests/test_worker_lost.py`
- [X] T038 [P] [US4] Add invalid-package / induced failure tests that assert no dummy `succeeded` in `CloudService/tests/test_train_failure_honesty.py`
- [X] T039 [P] [US4] Confirm Local destination path untouched (no account required) in `OpenYourBox/Source/train/TrainCoordinator.cpp` start-local path
- [X] T040 [US4] Surface `worker_lost` / failure messages clearly in cloud status mapping in `OpenYourBox/Source/train/TrainCoordinator.cpp`

**Checkpoint**: US4 — crash→failed, honest failures, Local intact, concurrency gate intact

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Production-safe staging review and quickstart sign-off.

- [X] T041 [P] Security pass: no secrets in query strings/logs; signed or path-protected downloads; TLS notes for prod in `CloudService/README.md` and `CloudService/api/`
- [X] T042 Confirm audio-thread rules: no HTTP/pack waits on `processBlock` in `OpenYourBox/Source/PluginProcessor.cpp` and cloud call sites in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T043 Run through `specs/018-real-cloud-training/quickstart.md` scenarios 1–9; record gaps in `specs/018-real-cloud-training/checklists/requirements.md` notes
- [X] T044 [P] Run `ctest -R CloudTrain` and `PYTHONPATH=. pytest CloudService/tests -q`; fix regressions

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: After Foundational — MVP real train success
- **US2 (Phase 4)**: After US1 runner can complete (needs real recipe events for checkpoints)
- **US3 (Phase 5)**: After Foundational; ideally after US1 so staging E2E is meaningful
- **US4 (Phase 6)**: After US1 runner exists (heartbeat/crash need a live train process)
- **Polish (Phase 7)**: After desired stories complete

### User Story Dependencies

- **US1 (P1)**: After Foundational — no dependency on US2–US4
- **US2 (P1)**: Needs US1 `train_runner` progress/export hooks
- **US3 (P1)**: Docs/tests can start after Foundational; full E2E needs US1
- **US4 (P2)**: Needs US1 runner for heartbeat/crash; Stop honesty overlaps US2

### Parallel Opportunities

- Phase 1: T002 ∥ T003
- Phase 2: T005 ∥ T006 ∥ T008; T014 ∥ T015 ∥ T016 ∥ T017 (plugin control surface)
- US1: T024 then T025 [P] after runner works
- US2: T030 [P] alongside server checkpoint work once ids exist
- US3: T034 [P] with docs/tests
- US4: T038 ∥ T039 while reconciler lands
- Polish: T041 ∥ T044

---

## Parallel Example: Foundational control-surface strip

```bash
Task: "Remove Pause/Resume UI in OpenYourBox/Source/ui/TrainPanel.cpp"
Task: "Remove pause/resume from OpenYourBox/Source/train/TrainCoordinator.cpp"
Task: "Restrict CloudTrainClient to stop in OpenYourBox/Source/train/CloudTrainClient.cpp"
Task: "Extend Tests/CloudTrainClientTests.cpp for Run/Stop-only"
```

## Parallel Example: User Story 1 smokes

```bash
Task: "Short real mapping smoke in CloudService/tests/test_real_train_mapping.py"
Task: "Short real reconstruction smoke in CloudService/tests/test_real_train_reconstruction.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup  
2. Complete Phase 2: Foundational (API + Run/Stop + no mock worker)  
3. Complete Phase 3: US1 real train → Gold artifact  
4. **STOP and VALIDATE** staging short mapping success  
5. Continue US2 → US3 → US4 → Polish  

### Incremental Delivery

1. Setup + Foundational → runnable gated API, honest controls  
2. US1 → real Cloud models (MVP)  
3. US2 → real checkpoints + Stop  
4. US3 → staging operator path + no-mock proof  
5. US4 → crash/fail honesty  
6. Polish → quickstart sign-off  

### Suggested MVP scope

**US1 only** (plus Foundational): enough to “try real cloud training” on staging.

---

## Notes

- Baseline client packaging/auth UX from `017` is assumed present; this feature completes server real training and strips Pause/Resume / mock advancement  
- Public production DNS go-live is out of Done (FR-008a)  
- [P] = different files, no incomplete-task dependencies  
- Commit after each task or logical group  
- Prefer short `total_steps` in automated real-train tests (CPU OK)  
