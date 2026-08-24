---
description: "Task list for Steerable Discovery & Training (Phase 3)"
---

# Tasks: Steerable Discovery & Training (Phase 3)

**Input**: Design documents from `specs/004-steerable-discovery-training/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Tests**: Spec does not request TDD. Include targeted regression/worker tests where they reduce recipe risk; validate via `quickstart.md`.

**Organization**: Tasks grouped by user story for independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`US1`–`US5`) on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- Graph: `OpenYourBox/Source/graph/`
- DSP: `OpenYourBox/Source/dsp/`
- Capture/Train/Library: `OpenYourBox/Source/capture/`, `train/`, `library/`
- UI: `OpenYourBox/Source/ui/`
- Backend: `Backend/`
- Tests: `Tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design artifacts and scaffolding before code changes.

- [X] T001 Verify Phase 3 design artifact cross-links and path conventions in `specs/004-steerable-discovery-training/plan.md`
- [X] T002 [P] Add implementation anchor notes for pairing/capture in `specs/004-steerable-discovery-training/contracts/instance-pairing-capture-contract.md`
- [X] T003 [P] Add implementation anchor notes for train IPC in `specs/004-steerable-discovery-training/contracts/train-worker-ipc.md`
- [X] T004 [P] Add implementation anchor notes for library UI in `specs/004-steerable-discovery-training/contracts/training-library-ui-contract.md`
- [X] T005 [P] Add implementation anchor notes for steerable graph/Train UI in `specs/004-steerable-discovery-training/contracts/steerable-graph-ui-contract.md`
- [X] T006 Create `OpenYourBox/Source/capture/`, `OpenYourBox/Source/train/`, and `OpenYourBox/Source/library/` directories and wire them into `CMakeLists.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared graph fields, persistence hooks, and module skeletons required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T007 Extend `GraphTypes.h` with FiLM pin metadata, `dilationGrowth`, `residual`, `prelu` activation, `armedForTraining`, and Weights provenance fields in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T008 [P] Extend TCN/Activation factory defaults (growth default 2, arm default true for trainable nodes) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T009 Persist new Phase 3 node fields in graph `ValueTree` serialization in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T010 [P] Add FiLM conditioning input pin creation for TCN in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T011 Extend connection validation for TCN FiLM pin (`signalKind` conditioning) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T012 [P] Scaffold `TrainingLibrary` store API (paths, index load/save) in `OpenYourBox/Source/library/TrainingLibrary.h`
- [X] T013 [P] Scaffold `TrainCoordinator` ChildProcess shell mirroring freeze in `OpenYourBox/Source/train/TrainCoordinator.h`
- [X] T014 [P] Scaffold pairing session types (master/slave, roles, sync state) in `OpenYourBox/Source/capture/CapturePairing.h`
- [X] T015 Add copyright-acknowledgment local persistence helper in `OpenYourBox/Source/library/CopyrightAcknowledgment.h`
- [X] T016 Wire new modules into editor/processor includes in `OpenYourBox/Source/PluginEditor.h` and `OpenYourBox/Source/PluginProcessor.h`
- [X] T017 Add empty `Backend/train_worker.py` entrypoint with JSON stdin/stdout handshake stub in `Backend/train_worker.py`
- [X] T018 Embed/package `train_worker.py` the same way as freeze worker in `CMakeLists.txt` (or existing BinaryData pipeline)

**Checkpoint**: Types, persistence, and module skeletons ready — story work can begin.

---

## Phase 3: User Story 1 - Capture & Training Library (Priority: P1) 🎯 MVP

**Goal**: Dual-instance Capture Samples (input x/y, default bypass, master/slave) and durable Training Library (import, select, preview, rename, delete) with copyright gate readiness.

**Independent Test**: Import a file pair into Library and/or capture via two instances; select pair; preview x/y; confirm slave reduced UI; Train remains gated without copyright ack.

### Implementation for User Story 1

- [X] T019 [P] [US1] Implement Training Library index persistence (entries, tags schema, selected IDs) in `OpenYourBox/Source/library/TrainingLibrary.cpp`
- [X] T020 [P] [US1] Implement pair file storage under plugin user-data and delete-with-confirm cleanup in `OpenYourBox/Source/library/TrainingLibrary.cpp`
- [X] T021 [US1] Implement file-pair import (align/crop length mismatch) in `OpenYourBox/Source/library/TrainingLibrary.cpp`
- [X] T022 [P] [US1] Implement localhost discovery registry + InterprocessConnection pairing in `OpenYourBox/Source/capture/CapturePairing.cpp`
- [X] T023 [US1] Implement master/slave role assignment and complementary Clean/Processed enforcement in `OpenYourBox/Source/capture/CapturePairing.cpp`
- [X] T024 [US1] Implement preallocated input-ring capture and record start/stop sync messages in `OpenYourBox/Source/capture/CaptureRecorder.cpp`
- [X] T025 [US1] Implement capture bypass flag (default on) with restore-on-exit in `OpenYourBox/Source/PluginProcessor.cpp`
- [X] T026 [US1] On successful stop, assemble x/y pair and append to Training Library in `OpenYourBox/Source/capture/CaptureRecorder.cpp`
- [X] T027 [P] [US1] Build Library panel list+detail UI (browse, multi-select, rename, delete) in `OpenYourBox/Source/ui/TrainingLibraryPanel.cpp`
- [X] T028 [US1] Add x/y in-plugin preview playback controls in `OpenYourBox/Source/ui/TrainingLibraryPanel.cpp`
- [X] T029 [US1] Build Capture Samples master menu and reduced slave menu in `OpenYourBox/Source/ui/CaptureSamplesPanel.cpp`
- [X] T030 [US1] Implement copyright acknowledgment modal + local log gate helper usage in `OpenYourBox/Source/ui/CopyrightModal.cpp`
- [X] T031 [US1] Orchestrate Library/Capture panels from editor in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T032 [US1] Block Train enablement until ack + ≥1 selected pair (stub Train button) in `OpenYourBox/Source/ui/TrainPanel.cpp`

**Checkpoint**: Library + Capture ingest work; file-only and DAW capture paths validated (quickstart scenarios 2–3).

---

## Phase 4: User Story 2 - Non-Blocking Train Worker (Priority: P1)

**Goal**: Master Train panel Run/Pause/Stop + live loss; background worker with fixed steerable-nafx recipe; RF-aware crops; audio keeps prior model.

**Independent Test**: Select library pair + copyright ack; Run Train; confirm loss updates and prior model audio; Pause/Resume/Stop without model swap.

### Implementation for User Story 2

- [X] T033 [P] [US2] Implement armed-subgraph JSON snapshot builder for train requests in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T034 [P] [US2] Implement `TrainCoordinator::start/pause/resume/stop` and progress event parsing in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T035 [US2] Stream progress (step, loss, lr, status) to UI without audio-thread work in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T036 [US2] Implement `Backend/train_worker.py` model build from graph fragment (FiLM-TCN capable) in `Backend/train_worker.py`
- [X] T037 [US2] Implement Adam + MultiStep LR schedule (80%/95%) and ca=0 conditioning in `Backend/train_worker.py`
- [X] T038 [US2] Implement MR-STFT loss with fft/win `{32,128,512,2048}` and hops `{16,64,256,1024}` in `Backend/train_worker.py`
- [X] T039 [US2] Implement RF-aware cropping and default segment length ~228308 (clamped) in `Backend/train_worker.py`
- [X] T040 [US2] Implement pause/resume/stop command handling and ~2500-step loop in `Backend/train_worker.py`
- [X] T041 [US2] On success, export TorchScript artifact path in response JSON in `Backend/train_worker.py`
- [X] T042 [US2] On stop/failure, ensure no auto-load signal to plugin in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T043 [US2] Build Train panel Run/Pause/Stop, loss/step, RF/window info line in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T044 [US2] Enforce mixed sample-rate selection block before start in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T045 [US2] Keep audio path on previously loaded model during job in `OpenYourBox/Source/PluginProcessor.cpp`
- [X] T046 [P] [US2] Add Python unit/smoke tests for recipe constants and RF crop helper in `Tests/test_train_worker.py`

**Checkpoint**: Train runs non-blocking with correct recipe; Stop leaves model unchanged (quickstart scenario 4).

---

## Phase 5: User Story 3 - Auto-Load Gold + Arm + Unfreeze Weights (Priority: P1)

**Goal**: On train success, replace armed trainable chain with Gold BlackBox; control sources stay Blue; arm/disarm; Unfreeze keeps trained weights.

**Independent Test**: Complete train; confirm Gold auto-load and free-c via existing Knob/XY; Unfreeze retains Weights path; disarm excludes a node from absorption.

### Implementation for User Story 3

- [X] T047 [P] [US3] Implement arm/disarm UI only on trainable-parameter nodes (default armed) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T048 [US3] Exclude control sources from arm and from train snapshot absorption in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T049 [US3] Off-thread prepare + atomic swap of trained TorchScript into Gold BlackBox for armed chain in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T050 [US3] Preserve `sourceSubgraph` + weight provenance for train-origin BlackBoxes in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T051 [US3] Keep Knob/XY/Audio sources Blue and rewire conditioning into Gold ports on auto-load in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T052 [US3] Extend Unfreeze to restore Blue nodes **with trained weight tensors/paths** in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T053 [US3] Add retry-load path when artifact exists but swap fails in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T054 [US3] Show Gold styling/lock affordance consistent with Phase 2 freeze in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T055 [US3] Ensure free-c live control after auto-load via existing Knob/XY runtime path in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`

**Checkpoint**: Train → Gold → steer c; Unfreeze keeps weights (quickstart scenarios 5–7).

---

## Phase 6: User Story 4 - FiLM, Residual, PReLU, Dilation Growth (Priority: P2)

**Goal**: Live steerable-nafx-equivalent TCN configuration: FiLM runtime, residual, PReLU, dilation growth UI with RF readout/presets.

**Independent Test**: Configure TCN with FiLM+residual+PReLU+growth 8/10; connect XY; confirm live audio and dilation/RF readout; graph shape rules hold.

### Implementation for User Story 4

- [X] T056 [P] [US4] Implement per-block FiLM in live TCN forward in `OpenYourBox/Source/dsp/TCNModel.cpp`
- [X] T057 [P] [US4] Replace power-of-two-only dilation with growth^n schedule in `OpenYourBox/Source/dsp/TCNModel.cpp`
- [X] T058 [US4] Implement optional residual path in TCN blocks in `OpenYourBox/Source/dsp/TCNModel.cpp`
- [X] T059 [US4] Add PReLU activation path for Activation and TCN in `OpenYourBox/Source/dsp/TCNModel.cpp` and related activation modules
- [X] T060 [US4] Expose residual checkbox, PReLU choice, and Dilation growth control in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T061 [US4] Show live dilations `1→G→G²→…` and RF (samples/ms) readout plus presets 2/8/10 in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T062 [US4] Apply missing FiLM input as c=0 in live engine in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T063 [US4] Align train_worker TCN construction with live growth/residual/FiLM/PReLU semantics in `Backend/train_worker.py`

**Checkpoint**: Users can build steerable-nafx-equivalent live graphs (quickstart scenario 1).

---

## Phase 7: User Story 5 - Weights Property (Priority: P2)

**Goal**: Weights shows seed N or file path; browse/load compatible weight files; Gold shows trained path.

**Independent Test**: Random TCN shows seed; after train Gold shows path; browse compatible file updates Weights; incompatible rejected.

### Implementation for User Story 5

- [X] T064 [P] [US5] Implement Weights property display (seed vs path) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T065 [US5] Implement browse file chooser and off-thread compatible weight load with atomic swap in `OpenYourBox/Source/dsp/WeightLoader.cpp`
- [X] T066 [US5] Reject incompatible files without changing active weights in `OpenYourBox/Source/dsp/WeightLoader.cpp`
- [X] T067 [US5] Update Weights path on train auto-load and preserve across Unfreeze in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T068 [US5] Clear to seed provenance on randomize in `OpenYourBox/Source/dsp/WeightRandomizer.cpp`

**Checkpoint**: Weights provenance and browse load work (quickstart scenario 5–6 extras).

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Hardening, packaging, and full quickstart validation.

- [X] T069 [P] Document auraloss/MR-STFT dependency install for train worker in `Backend/requirements.txt` (or project packaging docs)
- [X] T070 [P] Add C++ regression coverage for dilation growth^n and FiLM pin validation in `Tests/LiveGraphEngineTests.cpp`
- [X] T071 Add library SR-mismatch and selection-gate coverage in `Tests/` (new or existing suite)
- [X] T072 Verify UI stays responsive under train CPU load and no audio-thread allocations in capture/train paths in `OpenYourBox/Source/PluginProcessor.cpp`
- [X] T073 Run full `specs/004-steerable-discovery-training/quickstart.md` scenarios 1–8 and record results in `specs/004-steerable-discovery-training/checklists/requirements.md` Notes
- [X] T074 [P] Update `NOTICE` if auraloss/third-party train deps require attribution in `NOTICE`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Start immediately
- **Foundational (Phase 2)**: Depends on Setup — **blocks all stories**
- **US1 (Phase 3)**: After Foundational — MVP data path
- **US2 (Phase 4)**: After US1 library selection + copyright gate (needs selected pairs)
- **US3 (Phase 5)**: After US2 success artifact path
- **US4 (Phase 6)**: After Foundational; can overlap late US1/US2 but must finish before claiming steerable-nafx parity (align with T063)
- **US5 (Phase 7)**: After Foundational; integrates with US3 auto-load/Unfreeze
- **Polish (Phase 8)**: After desired stories complete

### User Story Dependencies

| Story | Depends on | Independently testable? |
|-------|------------|-------------------------|
| US1 Library/Capture | Foundational | Yes — import/capture without train |
| US2 Train | US1 (selected pairs) | Yes — Stop without auto-load |
| US3 Auto-load/Arm/Unfreeze | US2 success path | Yes — with stub/trained artifact |
| US4 FiLM/Growth/PReLU | Foundational | Yes — live graph only |
| US5 Weights | Foundational (+ US3 for train path) | Yes — seed/browse without train |

### Parallel Opportunities

- T002–T005 (contract anchors); T012–T015 (scaffolds); T019–T020; T022; T027; T033–T034; T056–T057; T064; T069–T070; T074

### Parallel Example: User Story 1

```bash
# After Foundational:
Task: "Implement Training Library index persistence in OpenYourBox/Source/library/TrainingLibrary.cpp"
Task: "Implement localhost discovery registry in OpenYourBox/Source/capture/CapturePairing.cpp"
Task: "Build Library panel list+detail UI in OpenYourBox/Source/ui/TrainingLibraryPanel.cpp"
```

---

## Implementation Strategy

### MVP First (US1)

1. Phase 1 Setup + Phase 2 Foundational  
2. Phase 3 US1 (Library + Capture + copyright modal)  
3. **STOP** — validate import/capture/select/preview  

### Incremental Delivery

1. US1 → data path demo  
2. US2 → train without auto-load (Stop)  
3. US3 → Gold + free c + Unfreeze weights  
4. US4 → steerable-nafx-equivalent live TCN  
5. US5 → Weights browse polish  
6. Phase 8 quickstart sign-off  

### Suggested MVP Scope

**US1 only** (Training Library + Capture Samples + copyright gate) is the smallest shippable increment. Full Phase 3 value needs US1–US3 minimum; US4–US5 complete steerable-nafx fidelity and provenance UX.

---

## Notes

- [P] = different files / no incomplete-task dependencies
- No TDD battery — spec did not request it; T046/T070/T071 are risk-focused
- Commit after each task or logical group
- Constitution: never block audio thread; Train worker is detached ChildProcess
- Reference notebook: `.ignore/steerable-nafx-main/steerable-nafx.ipynb`
