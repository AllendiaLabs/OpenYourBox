---
description: "Task list for Generalize Training Graph"
---

# Tasks: Generalize Training Graph

**Input**: Design documents from `specs/019-generalize-training-graph/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Spec does not require TDD. Include targeted C++/Python tests where plan Testing reduces gate/path/IPC risk; validate via `quickstart.md`.

**Organization**: Tasks grouped by user story for independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`US1`–`US6`) on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in graph/UI/train: `OpenYourBox/Source/graph/`, `ui/`, `train/`, `library/`, `dsp/`
- Local recipes: `Backend/train_worker.py`
- Cloud: `CloudService/worker/train_runner.py`, `OpenYourBox/Source/train/CloudJobPackage.*`
- Tests: `Tests/`, `Tests/test_train_worker.py`
- Examples: `OpenYourBox/Resources/examples/training/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm layout and example resource location; no product behavior change yet.

- [ ] T001 Decide and create example resource directory `OpenYourBox/Resources/examples/training/` (graph templates + configs) and note it in `specs/019-generalize-training-graph/plan.md`
- [ ] T002 [P] List CMake/`Tests/` touchpoints for new graph/train tests in `specs/019-generalize-training-graph/quickstart.md` Prerequisites if missing

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared graph types, live exclusion of train-only nodes, and `train_graph` package skeleton. Blocks all user stories.

**⚠️ CRITICAL**: No user story work begins until this phase is complete

- [ ] T003 Add `dataLoader` and `loss` node types, property keys, and `nodeTypeName` serialization in `OpenYourBox/Source/graph/GraphTypes.h`
- [ ] T004 [P] Register Data Loader and Loss factory defaults (output count, loss_type, weight) in `OpenYourBox/Source/graph/FactoryPalette.h` (insertable stubs OK)
- [ ] T005 Persist/load new node types in graph ValueTree paths in `OpenYourBox/Source/graph/NodeGraph.cpp` (and related snapshot helpers)
- [ ] T006 Exclude `dataLoader` and `loss` from live audible compilation in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [ ] T007 Introduce `train_graph` request fields (`active_data_loader_id`, `data_loader_bindings`, `loss_schedule`, `schema_version`) skeleton in `OpenYourBox/Source/graph/NodeGraph.cpp` `createTrainRequest` (still may emit transitional data until later stories)
- [ ] T008 [P] Accept `operation: "train_graph"` in local start path in `OpenYourBox/Source/train/TrainCoordinator.cpp` / `TrainCoordinator.h`
- [ ] T009 [P] Accept `train_graph` in cloud package assembly in `OpenYourBox/Source/train/CloudJobPackage.cpp` / `CloudJobPackage.h`
- [ ] T010 Remove `TrainObjective` / `lastTrainObjective` as required product state from `OpenYourBox/Source/graph/GraphTypes.h` and `OpenYourBox/Source/PluginProcessor.cpp` / `PluginProcessor.h` (migrate callers to no-objective flow)
- [ ] T011 [P] Document foundational IPC shape against `specs/019-generalize-training-graph/contracts/generalized-train-ipc.md` (keep contract in sync if field names drift)

**Checkpoint**: Graph can insert stub Data Loader/Loss; live audio ignores them; train package can say `train_graph` without mapping/reconstruction enum

---

## Phase 3: User Story 1 - Train Whatever Is On The Graph (Priority: P1) 🎯 MVP

**Goal**: Train/inference workflows are architecture-agnostic—no RAVE/steerable/TCN recipe modes; Train UI and messages describe the graph and settings only.

**Independent Test**: Assemble a non-branded graph; open Train with no architecture mode selector; with valid loader+loss+config (from later stories or stubs), Run trains the graph subgraph without hard-coded architecture path; live audio follows live wiring.

### Implementation for User Story 1

- [ ] T012 [US1] Remove Mapping/Reconstruction objective combo and objective-specific library Start rules from `OpenYourBox/Source/ui/TrainPanel.cpp` / `TrainPanel.h`
- [ ] T013 [US1] Remove reconstruction graph gate (`hasReconstructionTrainPath` / armed-bottleneck requirement) from Start preflight in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T014 [US1] Merge former mapping + reconstruction hyperparameter controls into one general Train HP surface in `OpenYourBox/Source/ui/TrainPanel.cpp` per `contracts/train-panel-generalized-ux.md`
- [ ] T015 [US1] Replace Gold auto-load labels “Trained Steerable” / “Trained RAVE” with neutral naming (e.g. “Trained Graph”) in `OpenYourBox/Source/graph/NodeGraph.cpp` (absorb/success path)
- [ ] T016 [P] [US1] Neutralize MLflow/default tags (`steerable` / `rave` / objective tags) in `OpenYourBox/Source/ui/TrainPanel.cpp` and train request assembly in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T017 [US1] Update `Backend/train_worker.py` entry to prefer `train_graph` and stop requiring `train_options.objective` (temporary: if loss_schedule empty, fail clearly rather than mapping/reconstruction branch)
- [ ] T018 [P] [US1] Align `CloudService/worker/train_runner.py` materialization with `train_graph` (no objective required) per `contracts/generalized-train-ipc.md`
- [ ] T019 [US1] Strip user-facing architecture-mode copy from Train/Library messaging in `OpenYourBox/Source/ui/TrainingLibraryPanel.cpp` and related strings

**Checkpoint**: Train panel has no architecture/objective mode; Start no longer depends on RAVE/steerable gates; worker/cloud accept `train_graph`

---

## Phase 4: User Story 2 - Feed Training With A Data Loader Node (Priority: P1)

**Goal**: Data Loader node with renamable outputs, per-output bindings, equal-count at Start (connected only), utilities, active-loader picker, refuse Start without usable loader.

**Independent Test**: Insert Data Loader; two renamed outputs with equal counts; connect; Run (with loss/arm from other stories); mismatch blocks until copy/repeat; sole vs multi-loader active rules.

### Implementation for User Story 2

- [ ] T020 [US2] Implement Data Loader element UI (output count, rename outputs) in `OpenYourBox/Source/graph/NodeRenderer.cpp` and property panel paths
- [ ] T021 [US2] Implement per-output `TrainingMaterialBinding` storage (audio_list + constant_scalar) on Data Loader in `OpenYourBox/Source/graph/NodeGraph.cpp` / `GraphTypes.h`
- [ ] T022 [US2] Build binding editor UX (pick from Training Library / constant utility) in `OpenYourBox/Source/ui/` (new panel helper or `TrainingLibraryPanel.cpp` integration)
- [ ] T023 [US2] Implement copy/repeat and constant/scalar-across-examples utilities for Data Loader outputs in `OpenYourBox/Source/graph/NodeGraph.cpp` (and UI triggers)
- [ ] T024 [US2] Enforce equal-count **only at Start** for **connected** outputs of the active loader in `OpenYourBox/Source/PluginEditor.cpp` `handleTrainRun`
- [ ] T025 [US2] Add Train panel **active Data Loader picker** (sole loader defaults active; multi without selection refuses) in `OpenYourBox/Source/ui/TrainPanel.cpp` / `TrainPanel.h`
- [ ] T026 [US2] Refuse Start with clear message when no usable Data Loader / no active designation in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T027 [US2] Resolve `data_loader_bindings` into the train package from the active loader in `OpenYourBox/Source/graph/NodeGraph.cpp` / `PluginEditor.cpp`
- [ ] T028 [P] [US2] Worker reads `data_loader_bindings` and zips connected outputs by example index in `Backend/train_worker.py`
- [ ] T029 [P] [US2] Add C++ tests for equal-count gate and sole/multi active-loader rules in `Tests/` (new or existing graph/train test file)

**Checkpoint**: Data Loader bindings drive batches; equal-count and active-loader gates match FR-002/003/004/017

---

## Phase 5: User Story 3 - Train Only The Data-Loader Subgraph (Priority: P1)

**Goal**: Train forward path from active Data Loader; update only armed∩on-path; passthrough/Gold rules; external-only data-loader cables; distinct cable color/N/A RMS; Train-tab opacity; live ignores train-only nodes.

**Independent Test**: Audio In→A→B→C + Knob; data loader on A external + Knob; refuse loader on B’s internal pin; arm A+C not B; Run updates A+C only; Train tab opacity; missing knob feed fails Start.

### Implementation for User Story 3

- [ ] T030 [US3] Implement data-loader path discovery (downstream from active loader outputs) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T031 [US3] Change train snapshot / `armed_element_ids` to armed∩on-path; include on-path passthrough helpers in fragment in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T032 [US3] Enforce connection rules: allow data-loader on external/inference sources (incl. coexist with live); refuse internal upstream-fed pins in `OpenYourBox/Source/graph/NodeGraph.cpp` (connect API) + tooltip in `NodeRenderer.cpp`
- [ ] T033 [US3] At Start, require every external source on the trainable path to have a data-loader feed or constant utility; refuse with message in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T034 [US3] Keep arm checkbox; default new processing elements armed; refuse arming Gold; refuse Start if no armed on-path trainable in `OpenYourBox/Source/graph/NodeGraph.cpp` / `NodeRenderer.cpp` / `PluginEditor.cpp`
- [ ] T035 [US3] Train-tab opacity: armed on-path normal; passthrough/off-path slightly transparent in `OpenYourBox/Source/graph/NodeRenderer.cpp` (Train tab active flag from `PluginEditor.cpp`)
- [ ] T036 [US3] Distinct data-loader cable color and RMS N/A (skip fill) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [ ] T037 [P] [US3] Confirm live path ignores Data Loader/Loss (audible = live cables only) in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [ ] T038 [US3] Worker applies grads only to `armed_element_ids`; on-path others passthrough in `Backend/train_worker.py`
- [ ] T039 [P] [US3] C++ tests for connect refuse/allow and path∩arm selection in `Tests/`

**Checkpoint**: SC-003/013/015/016/017/018/019 behaviors hold for path, arm, cables, and live isolation

---

## Phase 6: User Story 4 - Supervise With Loss Nodes And Save Training Configs (Priority: P1)

**Goal**: Loss nodes + weights + stage schedules; no reconstruction/mapping mode; user config library + project snapshot; worker optimizes scheduled losses; example templates/configs; capability-class parity.

**Independent Test**: Wire loss to output + target; save/load config; same-data and different-data runs without mode toggle; multi-stage short run; examples loadable.

### Implementation for User Story 4

- [ ] T040 [US4] Complete Loss element UI (loss_type catalog, weight, pins) in `OpenYourBox/Source/graph/NodeRenderer.cpp` / `FactoryPalette.h` / `GraphTypes.h` per `contracts/loss-nodes-and-stages.md`
- [ ] T041 [US4] Validate loss wiring at Start (usable loss on path; refuse incomplete/off-path supervision) in `OpenYourBox/Source/PluginEditor.cpp` / `NodeGraph.cpp`
- [ ] T042 [US4] Add loss stage schedule editor to Train panel in `OpenYourBox/Source/ui/TrainPanel.cpp` / `TrainPanel.h`
- [ ] T043 [US4] Serialize `loss_schedule` into train package in `OpenYourBox/Source/graph/NodeGraph.cpp` / `PluginEditor.cpp`
- [ ] T044 [US4] Implement weighted single-stage and multi-stage loss evaluation in `Backend/train_worker.py` (reuse spectral/KL/GAN/FM helpers; no `objective` branch)
- [ ] T045 [P] [US4] Progress events include stage name/index for multi-stage in `Backend/train_worker.py` and display in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T046 [US4] Implement user `TrainingConfigLibrary` persistence (list/save/load/rename/delete) in `OpenYourBox/Source/library/TrainingConfigLibrary.h` / `TrainingConfigLibrary.cpp`
- [ ] T047 [US4] Wire Train panel Save/Load to user config library in `OpenYourBox/Source/ui/TrainPanel.cpp`
- [ ] T048 [US4] Persist/restore project training-config snapshot in `OpenYourBox/Source/PluginProcessor.cpp` / `PluginProcessor.h`
- [ ] T049 [US4] Forward-compatible config load (ignore unknown; defaults + warn for missing) in `OpenYourBox/Source/library/TrainingConfigLibrary.cpp` and Train panel
- [ ] T050 [P] [US4] Update cloud runner to pass loss schedule through unchanged in `CloudService/worker/train_runner.py`
- [ ] T051 [US4] Author mapping-style and reconstruction-style **example graph templates** under `OpenYourBox/Resources/examples/training/`
- [ ] T052 [P] [US4] Author matching **example training configs** under `OpenYourBox/Resources/examples/training/` (clearly labeled examples, not modes)
- [ ] T053 [US4] Expose example load entry points labeled as examples/templates in `OpenYourBox/Source/ui/TrainPanel.cpp` (and/or preset/library UI)
- [ ] T054 [P] [US4] Python tests for weighted losses, stages, and recipe-parity smoke in `Tests/test_train_worker.py`

**Checkpoint**: FR-008–012, FR-018, FR-020 and SC-008–011/014 satisfied for new graphs

---

## Phase 7: User Story 5 - Group A Single Box (Priority: P2)

**Goal**: Create a group from exactly one allowed box; save/reload/ungroup preserve structure.

**Independent Test**: Select one Conv1D → group → one member + hubs; save/reload; ungroup preserves connections.

### Implementation for User Story 5

- [ ] T055 [US5] Lower `createGroup` minimum from 2 to 1 (keep Audio I/O exclusion) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T056 [US5] Update group menu enablement (`canGroup`) for single selection in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [ ] T057 [P] [US5] C++ test for one-member group create/ungroup/persist in `Tests/`

**Checkpoint**: FR-013 / SC-004

---

## Phase 8: User Story 6 - Simplify Palette By Removing Redundant Blocks (Priority: P2)

**Goal**: Remove TCN and Linear from new-insert palette; equivalents via Conv1D remain.

**Independent Test**: Palette has no TCN/Linear; stacked Conv1D and stride=kernel=dilation=1 Conv1D still insertable.

### Implementation for User Story 6

- [ ] T058 [US6] Remove TCN and Linear from insertable palette in `OpenYourBox/Source/graph/FactoryPalette.h`
- [ ] T059 [P] [US6] Stop advertising TCN/Linear in any insert UI paths in `OpenYourBox/Source/graph/NodeRenderer.cpp` (and related menus)
- [ ] T060 [US6] Ensure live/worker no longer require TCN/Linear for new graphs (leave dead handlers or remove safely) in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` / `Backend/train_worker.py` without adding migration
- [ ] T061 [P] [US6] C++ test or assertion that palette insertion list excludes `tcn` and `linear` in `Tests/`

**Checkpoint**: FR-014 / SC-006; legacy migration still out of scope (FR-015)

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation, docs sync, cleanup across stories.

- [ ] T062 [P] Sync all five contracts under `specs/019-generalize-training-graph/contracts/` with final field names and UI copy
- [ ] T063 Run and fix gaps against `specs/019-generalize-training-graph/quickstart.md` scenarios A–F
- [ ] T064 [P] Remove dead `train_steerable` / `objective` call sites from `OpenYourBox/Source/`, `Backend/train_worker.py`, and `CloudService/` (keep only if needed for out-of-scope legacy—prefer delete per FR-015)
- [ ] T065 Verify copyright gate and non-blocking live audio still hold during generalized train in `OpenYourBox/Source/ui/CopyrightModal.cpp` / `PluginEditor.cpp` / train coordinator paths
- [ ] T066 [P] Update `CloudService/README.md` train package notes for `train_graph` if they still document objective modes

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Start immediately
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: After Foundational — MVP product-language + no-mode train shell
- **US2 (Phase 4)**: After Foundational; needed for real Start materials (US1 Independent Test fully green after US2)
- **US3 (Phase 5)**: After US2 (path/connect rules need Data Loader)
- **US4 (Phase 6)**: After US2–US3 (losses supervise data-loader path; configs drive stages)
- **US5 (Phase 7)**: After Foundational — independent of train stories
- **US6 (Phase 8)**: After Foundational — independent of train stories (avoid removing TCN before example graphs that might have referenced it; examples use Conv1D)
- **Polish (Phase 9)**: After desired stories complete

### User Story Dependencies

- **US1**: Foundational; full Independent Test expects US2+US4 pieces
- **US2**: Foundational; no dependency on US5/US6
- **US3**: US2
- **US4**: US2 + US3 (for on-path loss validation)
- **US5**: Foundational only
- **US6**: Foundational only; prefer after example templates authored (US4) if examples ever mentioned TCN

### Parallel Opportunities

- T001–T002; T004 with early type work; T008–T009–T011 after T007
- US5 and US6 can proceed in parallel with US1 once Foundational is done
- Within US4: T051/T052 parallel; T050 parallel with library work; T054 after worker schedule exists
- T062/T066 parallel in Polish

---

## Parallel Example: User Story 2

```bash
# After Data Loader model/bindings exist:
Task: "Worker reads data_loader_bindings in Backend/train_worker.py"
Task: "C++ tests for equal-count / active-loader in Tests/"
```

## Parallel Example: User Story 4

```bash
Task: "Author example graph templates under OpenYourBox/Resources/examples/training/"
Task: "Author example training configs under OpenYourBox/Resources/examples/training/"
Task: "Cloud runner pass-through loss_schedule in CloudService/worker/train_runner.py"
```

## Parallel Example: US5 + US6 (with US1)

```bash
Task: "Group-of-one in NodeGraph.cpp / NodeRenderer.cpp"
Task: "Remove TCN/Linear from FactoryPalette.h"
Task: "Remove objective combo in TrainPanel.cpp"
```

---

## Implementation Strategy

### MVP First (User Story 1 + Foundational)

1. Complete Phase 1–2
2. Complete Phase 3 (US1): no modes, `train_graph` shell, neutral naming
3. **STOP and VALIDATE**: Train UI has no architecture selectors; reconstruction/mapping gates gone
4. Immediately continue US2–US4 for a trainable MVP (Start still needs loader + loss)

### Recommended first shippable increment

Foundational + **US1 + US2 + US3 + US4** (all P1) — required for a real train Run under the new model.

### Incremental Delivery

1. Setup + Foundational → types and `train_graph` skeleton  
2. US1 → architecture-agnostic Train shell  
3. US2 → Data Loader materials  
4. US3 → path/arm/cables/live isolation  
5. US4 → losses, stages, configs, examples  
6. US5 → group-of-one  
7. US6 → palette cleanup  
8. Polish → quickstart + contract sync  

### Parallel Team Strategy

- After Foundational: Dev A on US1→US2→US3→US4 train spine; Dev B on US5; Dev C on US6 (after examples if shared)

---

## Notes

- [P] = different files, no unfinished dependencies
- Legacy project migration explicitly out of scope (FR-015)—do not add TCN/Linear/objective migrators
- Discriminators stay worker-only (005/019 research)
- Commit after each task or logical group
- Stop at checkpoints to validate each story’s Independent Test criteria
