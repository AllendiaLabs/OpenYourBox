---
description: "Task list for RAVE Variational Bottleneck Parity"
---

# Tasks: RAVE Variational Bottleneck Parity

**Input**: Design documents from `specs/009-rave-vae-parity/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Tests**: Spec does not request TDD. Include targeted C++/Python regression tests per `plan.md` and `quickstart.md`; no test-first gate.

**Organization**: Tasks grouped by user story (US1–US4) for independent validation checkpoints.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- DSP: `OpenYourBox/Source/dsp/`
- Graph: `OpenYourBox/Source/graph/`
- UI: `OpenYourBox/Source/ui/`
- Backend: `Backend/`
- Tests: `Tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design artifacts and build wiring before bottleneck changes.

- [x] T001 Verify design artifact cross-links and execution order in `specs/009-rave-vae-parity/plan.md`
- [x] T002 [P] Register `Tests/VariationalBottleneckTests.cpp` in `CMakeLists.txt`
- [x] T003 [P] Confirm `Backend/train_worker.py` and `Backend/freeze_worker.py` remain embedded in plug-in bundle via `CMakeLists.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared constants, API surface, and removal of legacy 1×1 bottleneck paths.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T004 Add `defaultBottleneckKernelSize = 5` and document even-channel requirement in `OpenYourBox/Source/graph/GraphTypes.h`
- [x] T005 Refactor `OpenYourBox/Source/dsp/VariationalBottleneck.h` for grouped causal conv, softplus variance, separate `encodeMean` vs `encodeTrainSample` (or equivalent mode flag) with Doxygen
- [x] T006 [P] Remove legacy dual 1×1 weight initialization from `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` (`variationalBottleneck` randomize/prepare cases)
- [x] T007 [P] Remove legacy 1×1 `VariationalBottleneckLayer` (mean/logvar 1×1 convs) from `Backend/train_worker.py` in favour of placeholder stub wired in Phase 3

**Checkpoint**: Foundation ready — no code path emits legacy 1×1 bottleneck weights.

---

## Phase 3: User Story 1 — Reference-Equivalent Latent Sampling (Priority: P1) 🎯 MVP

**Goal**: Softplus variance, worker stochastic sampling during stage 1, live/Gold/checkpoint path always μ-only; KL consistent with parameterization.

**Independent Test**: Worker `train()` forward ≠ `eval()` forward on same weights/input; live encode is deterministic (bit-identical consecutive calls); no ε on audio thread during background train (spec US1 scenarios 1–5).

### Implementation for User Story 1

- [x] T008 [US1] Implement `VariationalBottleneckLayer` with softplus variance, reparameterized sampling in `train()` mode, and μ-only in `eval()` mode in `Backend/train_worker.py` per `specs/009-rave-vae-parity/contracts/variational-bottleneck-contract.md`
- [x] T009 [US1] Update `_bottleneck_kl()` for softplus/log-σ² parameterization in `Backend/train_worker.py`
- [x] T010 [P] [US1] Mirror parity `VariationalBottleneckLayer` forward modes in `Backend/freeze_worker.py`
- [x] T011 [US1] Implement softplus variance and μ-only `encodeMean` in `OpenYourBox/Source/dsp/VariationalBottleneck.cpp`
- [x] T012 [US1] Wire live variational bottleneck to call μ-only path (never sample) in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T013 [US1] Ensure Gold blackbox encode/forward applies μ-only before fidelity in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T014 [P] [US1] Add worker train-vs-eval forward tests in `Tests/test_train_worker.py`
- [x] T015 [P] [US1] Add live deterministic encode tests in `Tests/VariationalBottleneckTests.cpp`

**Checkpoint**: US1 complete — live path stable; worker samples only off audio thread.

---

## Phase 4: User Story 2 — Bottleneck Head Geometry (Priority: P1)

**Goal**: Grouped causal conv `groups=2` (μ branch + variance branch), configurable `kernel_size` default 5, shape gates, layout defaults.

**Independent Test**: Insert latest-continuous layout → kernel 5; illegal odd upstream channels refused; kernel_size edit propagates to prepare/train/export (spec US2 scenarios 1–5).

### Implementation for User Story 2

- [x] T016 [US2] Add grouped causal `Conv1d(groups=2)` head with per-group channel split in `Backend/train_worker.py` `VariationalBottleneckLayer`
- [x] T017 [US2] Implement grouped causal conv with preallocated `(kernel_size−1)` history (zero audio-thread alloc) in `OpenYourBox/Source/dsp/VariationalBottleneck.cpp`
- [x] T018 [US2] Update grouped weight randomization and `parameterCount` for k× grouped layout in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T019 [US2] Add `kernel_size` property (default 5) to `variationalBottleneck` palette in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T020 [US2] Read `kernel_size` during live prepare and pass to `VariationalBottleneck` in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T021 [US2] Add even input-channel shape gate with tooltip on connect/arm in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T022 [US2] Set `kernel_size=5` on bottleneck node in `OpenYourBox/Source/graph/RaveLayouts.cpp` for original and latest-continuous layouts
- [x] T023 [P] [US2] Add layout kernel default and odd-channel rejection tests in `Tests/LiveGraphEngineTests.cpp`

**Checkpoint**: US2 complete — reference head geometry end-to-end (Python + C++ + UI).

---

## Phase 5: User Story 3 — Compactness From Validation μ (Priority: P1)

**Goal**: 98/2 train/val split (seed 42, val cap 1000); PCA on validation μ at stage-1 end; buffers on Gold + Unfreeze; fidelity gated until ready.

**Independent Test**: Short reconstruction train → `compactness.ready: true` at stage-1 end; mid-stage-1 checkpoint fidelity inactive; Unfreeze copies buffers; fidelity sweep coarsens reconstruction (spec US3 scenarios 1–7).

### Implementation for User Story 3

- [x] T024 [US3] Implement `split_reconstruction_corpus()` (98% train / 2% val, seed 42, max 1000 val) in `Backend/train_worker.py` per `specs/009-rave-vae-parity/contracts/compactness-pca-contract.md`
- [x] T025 [US3] Restrict stage-1 minibatch sampling to train paths only in `Backend/train_worker.py`
- [x] T026 [US3] Replace single-batch PCA with full validation-pass μ collection at stage-1 end in `Backend/train_worker.py`
- [x] T027 [US3] Emit `compactness.ready` / `status` in train IPC events and checkpoint metadata in `Backend/train_worker.py`
- [x] T028 [US3] Embed `latent_mean`, `latent_pca`, `cumulative_variance` buffers in `_export_rave_scripted()` in `Backend/train_worker.py`
- [x] T029 [US3] Load compactness buffers into Gold runtime tensors in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T030 [US3] Copy compactness buffers and `compactnessReady` to Blue bottleneck on Unfreeze in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T031 [US3] Persist `kernel_size`, `compactnessReady`, and PCA tensor refs in graph ValueTree in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T032 [US3] Gate fidelity slider/display until `compactnessReady` in `OpenYourBox/Source/ui/TrainPanel.cpp` and bottleneck property UI in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T033 [US3] Clear `compactnessReady` and PCA tensors on bottleneck weight randomize in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T034 [P] [US3] Add split reproducibility, train-only sampler, and val-μ PCA source tests in `Tests/test_train_worker.py`

**Checkpoint**: US3 complete — fidelity/compactness matches acids-rave validation PCA rules.

---

## Phase 6: User Story 4 — End-to-End Parity Smoke (Priority: P2)

**Goal**: Practitioner-visible parity across sampling, head geometry, and compactness; quickstart validation.

**Independent Test**: Run `specs/009-rave-vae-parity/quickstart.md` scenarios 1–8; optional side-by-side acids-rave sign-off (SC-007).

### Implementation for User Story 4

- [x] T035 [US4] Add worker-eval μ vs live-encode μ parity test harness in `Tests/LiveGraphEngineTests.cpp`
- [x] T036 [US4] Add post-train fidelity monotonic rank assertion in `Tests/ProcessorIntegrationTests.cpp` or `Tests/test_train_worker.py`
- [x] T037 [P] [US4] Execute and tick off manual quickstart steps in `specs/009-rave-vae-parity/quickstart.md`

**Checkpoint**: US4 complete — end-to-end parity smoke documented.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Regression sweep, TODO hygiene, plan status.

- [x] T038 [P] Mark completed RAVE parity bullets in `TODO.md` referencing `specs/009-rave-vae-parity/`
- [x] T039 Run `ctest --test-dir OpenYourBox/Builds/Release -R "VariationalBottleneck|OpenYourBoxLiveGraph|OpenYourBoxProcessor" --output-on-failure` and `.venv/bin/python Tests/test_train_worker.py -v`
- [x] T040 Update plan status to "Ready for implement" in `specs/009-rave-vae-parity/plan.md` after all checkpoints pass

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **blocks all user stories**
- **US1 (Phase 3)**: Depends on Foundational — MVP
- **US2 (Phase 4)**: Depends on US1 worker layer stub (T008); C++ head (T017) can follow US1 C++ μ-only (T011)
- **US3 (Phase 5)**: Depends on US1 + US2 (needs correct encode μ path and grouped head)
- **US4 (Phase 6)**: Depends on US1–US3
- **Polish (Phase 7)**: Depends on desired story completion

### User Story Dependencies

| Story | Depends on | Independent test focus |
|-------|------------|------------------------|
| US1 | Foundational | Train vs eval sampling; live deterministic μ |
| US2 | US1 (worker layer) | kernel_size=5; grouped head; shape gate |
| US3 | US1, US2 | Val split; PCA; fidelity gate; Unfreeze |
| US4 | US1–US3 | Quickstart + parity harness |

### Parallel Opportunities

- **Phase 1**: T002 ∥ T003
- **Phase 2**: T006 ∥ T007 (after T005)
- **US1**: T010 ∥ T014 (after T008); T014 ∥ T015
- **US2**: T023 after T019–T022
- **US3**: T034 after T024–T026
- **US4**: T037 ∥ T035–T036 (after US3)
- **Polish**: T038 ∥ T039

### Parallel Example: User Story 1

```bash
# After T008 completes:
Task T010: Mirror layer in Backend/freeze_worker.py
Task T014: Tests/test_train_worker.py train-vs-eval tests
Task T015: Tests/VariationalBottleneckTests.cpp deterministic encode
```

### Parallel Example: User Story 3

```bash
# After T026 completes:
Task T034: Tests/test_train_worker.py split + PCA tests
Task T032: NodeGraph.cpp randomize clears compactness (different concern, same file — run sequentially)
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Phase 1 Setup + Phase 2 Foundational
2. Phase 3 US1 (softplus + sampling split)
3. **STOP and VALIDATE**: `Tests/test_train_worker.py` + `Tests/VariationalBottleneckTests.cpp`
4. Proceed to US2 → US3 for full parity

### Incremental Delivery

1. US1 → stable live μ-only + worker sampling
2. US2 → reference head geometry
3. US3 → validation PCA + fidelity (feature-complete for TODO)
4. US4 → smoke / practitioner sign-off

### Suggested Single-Developer Order

T001–T007 → T008–T015 → T016–T023 → T024–T034 → T035–T040

---

## Notes

- Breaking change (FR-013): do not restore legacy 1×1 checkpoints
- Live path **never** samples — even during stage-1 background train (clarify session)
- Fidelity inactive until stage-1 PCA completes (mid-checkpoint `compactness.ready: false`)
- Contracts: `specs/009-rave-vae-parity/contracts/variational-bottleneck-contract.md`, `specs/009-rave-vae-parity/contracts/compactness-pca-contract.md`
