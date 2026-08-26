---
description: "Task list for RAVE Architecture & Training"
---

# Tasks: RAVE Architecture & Training

**Input**: Design documents from `specs/005-rave-architecture-training/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Tests**: Spec does not request TDD. Include targeted C++/Python regression tests where they reduce recipe and shape risk; validate via `quickstart.md`.

**Organization**: Tasks grouped by user story for independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`US1`–`US6`) on story-phase tasks only
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

**Purpose**: Confirm design artifacts and CMake wiring before node/DSP work.

- [X] T001 Verify RAVE design artifact cross-links and path conventions in `specs/005-rave-architecture-training/plan.md`
- [X] T002 [P] Confirm `Backend/train_worker.py` is already embedded beside freeze in `CMakeLists.txt`
- [X] T003 Add new DSP/graph sources (`PqmfBank`, `RateConv`, `VariationalBottleneck`, `NoiseSynthesizer`, `RaveLayouts`) to the plug-in target in `CMakeLists.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared types, shape domains, persistence, and module skeletons required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Extend `NodeType`, `ShapeSignature` (`domain` audio|multiband|latent, `temporalRate`), and RAVE defaults (nBand 16, latent 128) in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T005 [P] Implement domain/rate/channel compatibility helpers on `ShapeSignature` in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T006 Scaffold factory + ValueTree persist for `pqmfAnalysis`, `pqmfSynthesis`, `rateConv`, `variationalBottleneck`, `noiseSynthesizer` in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T007 [P] Persist last-used Train `objective` (`mapping`|`reconstruction`, default mapping) on the processor in `OpenYourBox/Source/PluginProcessor.cpp`
- [X] T008 Extend Training Library index schema with `kind` (`pair`|`clip`) and system tags, defaulting existing entries to `pair`, in `OpenYourBox/Source/library/TrainingLibrary.h`
- [X] T009 [P] Add `train_options.objective` (default `mapping`) to start-request assembly in `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T010 Wire new DSP headers into live-engine includes in `OpenYourBox/Source/dsp/LiveGraphEngine.h`

**Checkpoint**: Types, persistence, and skeletons ready — story work can begin.

---

## Phase 3: User Story 1 - Assemble a RAVE Graph From Live Elements (Priority: P1) 🎯 MVP

**Goal**: Palette + live causal PQMF, rate conv, bottleneck, and noise synth; illegal audio/multiband/latent connections refused; random-weight graphs play; Freeze still works.

**Independent Test**: Quickstart scenario 1 (without requiring a layout insert from US6): add PQMF analysis/synthesis, rate convs, bottleneck; play if shapes legal; refuse illegal domain cables; existing TCN graphs unchanged.

### Implementation for User Story 1

- [X] T011 [P] [US1] Implement Kaiser cosine-modulated PQMF analysis/synthesis (default 16 bands, causal streaming) in `OpenYourBox/Source/dsp/PqmfBank.cpp`
- [X] T012 [P] [US1] Implement causal `rateConv` downsample/upsample with preallocated leftover/history rings in `OpenYourBox/Source/dsp/RateConv.cpp`
- [X] T013 [US1] Implement live variational bottleneck (reparameterize at inference with prior sample; store fidelity percent) in `OpenYourBox/Source/dsp/VariationalBottleneck.cpp`
- [X] T014 [P] [US1] Implement filtered-noise synthesizer addend in `OpenYourBox/Source/dsp/NoiseSynthesizer.cpp`
- [X] T015 [US1] Dispatch new node types with zero audio-thread allocations (prepare rings on GUI thread) in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T016 [P] [US1] Add palette items and properties (nBand, stride, direction, latentSize, noiseBands) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T017 [US1] Create pins and labels for RAVE nodes (audio/multiband/latent) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T018 [US1] Refuse illegal domain, temporal-rate, and nBand mismatches with tooltips in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T019 [P] [US1] Show causal delay (samples/ms) on RAVE processing nodes in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T020 [US1] Accept new node types in freeze graph fragments in `Backend/freeze_worker.py`
- [X] T021 [P] [US1] Add C++ tests for PQMF analysis→synthesis invertibility and causal rate-conv length in `Tests/LiveGraphEngineTests.cpp`

**Checkpoint**: Users can wire a RAVE-shaped live graph; TCN mapping graphs still work.

---

## Phase 4: User Story 2 - Same Library and Capture Shell (Priority: P1)

**Goal**: One library for pairs and unpaired clips; tags; Capture Pair|Single; reconstruction uses pair x and y; mapping errors if unpaired selected.

**Independent Test**: Quickstart scenarios 3–4: import clip, Single capture, Pair still works; mapping+clip errors; reconstruction selects pair (x+y).

### Implementation for User Story 2

- [X] T022 [P] [US2] Persist `kind`, system tags `pair`/`unpaired`, and optional user tags in `OpenYourBox/Source/library/TrainingLibrary.cpp`
- [X] T023 [P] [US2] Implement single-file import as `clip` entries in `OpenYourBox/Source/library/TrainingLibrary.cpp`
- [X] T024 [US2] Implement Capture Samples kind **Single** (no peer; input record; default bypass; append clip) in `OpenYourBox/Source/capture/CaptureRecorder.cpp` and `OpenYourBox/Source/ui/CaptureSamplesPanel.cpp`
- [X] T025 [US2] Show tags and objective-based warn/filter in `OpenYourBox/Source/ui/TrainingLibraryPanel.cpp`
- [X] T026 [US2] Block mapping Run with an error when any unpaired/clip is selected in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T027 [US2] Flatten selected pairs to both x and y plus clips when assembling reconstruction corpus in `OpenYourBox/Source/train/TrainCoordinator.cpp`

**Checkpoint**: Library/Capture remain one product; mapping pair path unchanged.

---

## Phase 5: User Story 3 - Unified Train Panel + Reconstruction Recipe (Priority: P1)

**Goal**: Same Train panel with explicit objective; reconstruction two-stage recipe in the existing worker; mapping recipe unchanged; graph validity gate; stage in progress UI.

**Independent Test**: Quickstart 5–6 and 8: reconstruction on TCN-only graph refused; reconstruction Run shows stage; Pause/Stop; mapping still ~2500-step MR-STFT.

### Implementation for User Story 3

- [X] T028 [P] [US3] Add mapping|reconstruction objective control to the existing Train panel in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T029 [US3] Restore last-used objective per instance from processor state in `OpenYourBox/Source/ui/TrainPanel.cpp` and `OpenYourBox/Source/PluginProcessor.cpp`
- [X] T030 [US3] Gate reconstruction Run on armed autoencoder path (bottleneck + decode to audio) in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T031 [US3] Send `objective`, `clips`, reconstruction step counts, and mixed-SR/channel gates per `contracts/unified-train-ipc.md` from `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T032 [US3] Parse and display `stage` (`representation`|`quality`) on progress events in `OpenYourBox/Source/train/TrainCoordinator.cpp` and `OpenYourBox/Source/ui/TrainPanel.cpp`
- [X] T033 [US3] Keep mapping path (`objective=mapping`) on the existing steerable recipe in `Backend/train_worker.py`
- [X] T034 [US3] Implement reconstruction stage 1 (dual spectral distance + KL warmup, causal graph build) in `Backend/train_worker.py`
- [X] T035 [US3] Implement reconstruction stage 2 (freeze encoder, hinge GAN + feature matching, train-only discriminators) in `Backend/train_worker.py`
- [X] T036 [US3] Compute compactness PCA after stage 1 (fallback full latent + status) in `Backend/train_worker.py`
- [X] T037 [P] [US3] Add Python tests for objective dispatch, stage counters, and corpus flatten (pair x+y) in `Tests/test_train_worker.py`

**Checkpoint**: One Train shell; recipes differ by objective only.

---

## Phase 6: User Story 4 - Auto-Load Gold Forward / Encode / Decode (Priority: P1)

**Goal**: Successful reconstruction auto-loads Gold with forward, encode, decode; Unfreeze keeps trained weights; Stop is not success; optional checkpoint load continues to work.

**Independent Test**: Quickstart 7 (success or shortened test steps): Gold methods/ports; Unfreeze keeps weights; Stop does not swap as success.

### Implementation for User Story 4

- [X] T038 [P] [US4] Add Gold RAVE audio/latent ports (forward default wiring) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T039 [US4] Invoke TorchScript `forward`/`encode`/`decode` from the live Gold path in `OpenYourBox/Source/dsp/TorchScriptBlackBox.cpp` and `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T040 [US4] Export encode/decode/forward when a bottleneck is in the freeze fragment in `Backend/freeze_worker.py`
- [X] T041 [US4] Export reconstruction success artifact with those methods plus compactness buffers in `Backend/train_worker.py`
- [X] T042 [US4] Auto-load RAVE Gold for the armed chain on reconstruction `success` only (not `stopped`) in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/train/TrainCoordinator.cpp`
- [X] T043 [US4] Unfreeze RAVE Gold back to Blue nodes with trained weight paths in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T044 [US4] Keep hear-while-training optional checkpoint swap off the success-auto-load path in `OpenYourBox/Source/train/TrainCoordinator.cpp`

**Checkpoint**: Reconstruction success matches Phase 3 Gold auto-load pattern with extra methods.

---

## Phase 7: User Story 5 - Fidelity Always-On (Priority: P2)

**Goal**: Fidelity 0–100 on Gold and live bottleneck without Unfreeze; Unfreeze preserves the setting; compactness applied internally so latent port width stays stable.

**Independent Test**: Quickstart 7 fidelity sweep on Gold; Unfreeze retains percent.

### Implementation for User Story 5

- [X] T045 [P] [US5] Expose Fidelity property on variational bottleneck in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T046 [US5] Expose Fidelity on Gold RAVE (always applies, FiLM-on-Gold pattern) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T047 [US5] Apply compactness crop inside encode/forward (full-width latent ports) in `OpenYourBox/Source/dsp/VariationalBottleneck.cpp` and `OpenYourBox/Source/dsp/TorchScriptBlackBox.cpp`
- [X] T048 [US5] Copy current fidelity onto restored bottleneck on Unfreeze in `OpenYourBox/Source/graph/NodeGraph.cpp`

**Checkpoint**: Fidelity is a live Gold control, not a train-only export flag.

---

## Phase 8: User Story 6 - Original and Latest Continuous Layouts (Priority: P2)

**Goal**: Insertable original vs latest-continuous graphs with mono|stereo choice; no silent host downmix; delay readout on insert.

**Independent Test**: Quickstart scenarios 1–2.

### Implementation for User Story 6

- [X] T049 [P] [US6] Build original RAVE layout (PQMF 16, strides 4-4-4-2, waveform×loudness+noise, latent 128) in `OpenYourBox/Source/graph/RaveLayouts.cpp`
- [X] T050 [P] [US6] Build latest-continuous layout (residual dilated stacks + amplitude modulation, same ratios) in `OpenYourBox/Source/graph/RaveLayouts.cpp`
- [X] T051 [US6] Add Insert RAVE layout menu (Original|Latest × Mono|Stereo) in `OpenYourBox/Source/graph/NodeRenderer.cpp` or `OpenYourBox/Source/PluginEditor.cpp`
- [X] T052 [US6] Refuse mono layout vs stereo host without explicit adapters in `OpenYourBox/Source/graph/NodeGraph.cpp`

**Checkpoint**: Users can start from either published lineage without a second trainer.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Attribution, tests, latent analysis, quickstart sign-off.

- [X] T053 [P] Add acids-rave / RAVE paper attribution when PQMF or recipe code is ported in `NOTICE`
- [X] T054 [P] Confirm latent-domain analysis plots all feature dimensions on shared axes in `OpenYourBox/Source/dsp/LiveGraphPublisher.cpp`
- [X] T055 Add C++ tests for illegal domain connections and reconstruction-path validity helper in `Tests/LiveGraphEngineTests.cpp`
- [X] T056 Extend Python tests for mapping unpaired rejection (plugin-equivalent flatten rules documented in worker helpers) in `Tests/test_train_worker.py`
- [X] T057 Run `specs/005-rave-architecture-training/quickstart.md` scenarios 1–9 (shortened reconstruction steps allowed) and record results in `specs/005-rave-architecture-training/checklists/requirements.md`
- [X] T058 Document new public DSP/graph APIs with Doxygen comments in `OpenYourBox/Source/dsp/PqmfBank.h`, `OpenYourBox/Source/dsp/RateConv.h`, `OpenYourBox/Source/dsp/VariationalBottleneck.h`, and `OpenYourBox/Source/graph/RaveLayouts.h`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Start immediately
- **Foundational (Phase 2)**: Depends on Setup — **blocks all stories**
- **US1 (Phase 3)**: After Foundational — live graph MVP
- **US2 (Phase 4)**: After Foundational; can overlap US1 (library vs DSP files)
- **US3 (Phase 5)**: After US1 (valid graph to train) and US2 (corpus rules)
- **US4 (Phase 6)**: After US3 success artifact path
- **US5 (Phase 7)**: After Foundational; Gold fidelity integrates with US4
- **US6 (Phase 8)**: After US1 node types exist
- **Polish (Phase 9)**: After desired stories complete

### User Story Dependencies

| Story | Depends on | Independently testable? |
|-------|------------|-------------------------|
| US1 Live RAVE graph | Foundational | Yes — wire/play without train |
| US2 Library/Capture | Foundational | Yes — ingest without RAVE nodes |
| US3 Reconstruction Train | US1 + US2 | Yes — Stop without auto-load |
| US4 Gold methods | US3 success path | Yes — with stub artifact |
| US5 Fidelity | Foundational (+ US4 for Gold) | Yes — bottleneck control without full train |
| US6 Layouts | US1 types | Yes — insert/play without train |

### Parallel Opportunities

- T002–T003; T004 vs T007–T009; T011–T012–T014; T022–T023; T049–T050; T053–T054
- After Foundational: US1 and US2 in parallel (different directories)

### Parallel Example: User Story 1

```bash
Task: "Implement Kaiser PQMF in OpenYourBox/Source/dsp/PqmfBank.cpp"
Task: "Implement causal rateConv in OpenYourBox/Source/dsp/RateConv.cpp"
Task: "Implement noise synthesizer in OpenYourBox/Source/dsp/NoiseSynthesizer.cpp"
```

---

## Implementation Strategy

### MVP First (US1)

1. Phase 1 Setup + Phase 2 Foundational  
2. Phase 3 US1 (live RAVE elements + shape gates)  
3. **STOP** — validate wiring and playback  

### Incremental Delivery

1. US1 → hear a RAVE-shaped random graph  
2. US2 → unpaired corpus in the same Library  
3. US3 → reconstruction Train in the same panel (Stop)  
4. US4 → Gold forward/encode/decode  
5. US5 → fidelity on Gold  
6. US6 → one-click layouts  
7. Phase 9 quickstart sign-off  

### Suggested MVP Scope

**US1 only** (live RAVE element set + shape integrity) is the smallest shippable increment. Full feature value needs US1–US4; US5–US6 complete compactness UX and layout speed.

---

## Notes

- [P] = different files / no incomplete-task dependencies
- No TDD battery — spec did not request it; T021/T037/T055/T056 are risk-focused
- Commit after each task or logical group
- Constitution: never block audio thread; Train worker stays detached ChildProcess
- Reconstruction defaults 1e6+1e6 steps; tests may shorten via IPC fields
- Mapping Train and Capture Pair must not regress
- Reference: `.ignore/repos/RAVE-master/` and `.ignore/papers/rave.pdf` (not shipped)
