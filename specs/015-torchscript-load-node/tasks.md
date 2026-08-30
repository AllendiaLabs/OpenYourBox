---
description: "Task list for TorchScript Checkpoint Loader Node"
---

# Tasks: TorchScript Checkpoint Loader Node

**Input**: Design documents from `specs/015-torchscript-load-node/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Spec does not require TDD. Targeted live-engine tests are included where `plan.md` lists `Tests/` coverage; no test-first gate. Manual UI flows validated via `quickstart.md`.

**Organization**: Tasks grouped by user story (US1–US4) for independent validation checkpoints.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- Graph: `OpenYourBox/Source/graph/`
- DSP: `OpenYourBox/Source/dsp/`
- Processor/Editor: `OpenYourBox/Source/PluginProcessor.*`, `OpenYourBox/Source/PluginEditor.*`
- Tests: `Tests/`
- Specs: `specs/015-torchscript-load-node/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design artifacts and inventory BlackBox / registry / palette touch-points before code changes.

- [x] T001 Verify design artifact cross-links (spec clarifications → contracts → plan → data-model) in `specs/015-torchscript-load-node/plan.md`
- [x] T002 [P] Inventory `BlackBoxOrigin`, `NodeType::blackBox`, fidelity / compactness fields on `GraphNode` in `OpenYourBox/Source/graph/GraphTypes.h`
- [x] T003 [P] Inventory `makeNode(blackBox)`, `absorbArmedChain`, `setWeightsPath`, `copyCompactnessFromArtifact`, Unfreeze gates in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T004 [P] Inventory Factory palette categories and BlackBox / weights / fidelity property rows in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [x] T005 [P] Inventory `TorchScriptBlackBoxFactory::load`, encode/decode / conditioning probe in `OpenYourBox/Source/dsp/TorchScriptBlackBox.cpp`
- [x] T006 [P] Inventory `prepareFrozenArtifact` / `prepareTrainedArtifact` / `resolveFrozenBlackBox` / `publishedFrozenArtifacts` in `OpenYourBox/Source/PluginProcessor.cpp`
- [x] T007 [P] Inventory `handleBrowseWeights` and restore/`applyPatchSnapshot` path checks in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Origin enum, empty external-load node factory, prepare/registry API, and safe live empty/error stubs required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T008 Add `BlackBoxOrigin::externalLoad` (+ persist string `external_load`) and document load-status / channel-override fields on `GraphNode` in `OpenYourBox/Source/graph/GraphTypes.h`
- [x] T009 Serialize/deserialize `blackBoxOrigin == external_load` and stub override/inferred channel properties in `OpenYourBox/Source/graph/NodeGraph.cpp` (`nodeToTree` / `nodeFromTree`)
- [x] T010 Add Factory palette entry **TorchScript Load** per `specs/015-torchscript-load-node/contracts/factory-palette-torchscript-load-contract.md` that creates `NodeType::blackBox` with `externalLoad`, empty path, audio in/out only in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T011 Implement `makeExternalTorchScriptLoadNode` (or `makeNode` branch) defaults: Gold presentation, no Control/latent pins, empty status, choose-file prompt properties in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T012 Add `prepareExternalArtifact` API (off-audio-thread `TorchScriptBlackBoxFactory::load`, silence-preservation off, publish to `publishedFrozenArtifacts`) in `OpenYourBox/Source/PluginProcessor.h` and `OpenYourBox/Source/PluginProcessor.cpp`
- [x] T013 Wire LiveGraphEngine empty `externalLoad` path: dry passthrough when status empty/never loaded; silence when error with no prior factory; keep prior factory during loading in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T014 Disable Unfreeze for `BlackBoxOrigin::externalLoad` with user-visible explanation in `OpenYourBox/Source/graph/NodeGraph.cpp` and/or `OpenYourBox/Source/graph/NodeRenderer.cpp`

**Checkpoint**: Foundation ready — palette places an empty TorchScript Load Gold node; prepare API exists; live engine dry-passthroughs empty nodes; Unfreeze blocked for external origin.

---

## Phase 3: User Story 1 — Load and Hear an External Checkpoint (Priority: P1) 🎯 MVP

**Goal**: User browses a valid TorchScript checkpoint, loads it off the audio thread, swaps into the live graph, and hears processed audio; can swap to another file without restarting the plugin.

**Independent Test**: `specs/015-torchscript-load-node/quickstart.md` Scenario A — Audio In → TorchScript Load → Audio Out; empty dry passthrough; load valid `.pt`; hear processed output; swap file.

### Implementation for User Story 1

- [x] T015 [US1] Property panel: path display, Browse, Clear, and load-status/error text for `externalLoad` nodes in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [x] T016 [US1] Wire Browse to `FileChooser` (`*.pt;*.pth`) → `prepareExternalArtifact` (not WeightLoader-only) from `OpenYourBox/Source/PluginEditor.cpp` / `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [x] T017 [US1] On prepare success: set `artifactPath`/`weightsPath`, status `ready`, copy compactness if present, rebuild runtime atomically in `OpenYourBox/Source/PluginProcessor.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T018 [US1] On prepare success without encode/decode or conditioning: keep audio-only pins; run via existing BlackBox `forward` path in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T019 [US1] Support selecting a different valid checkpoint (retain prior model until new factory ready) per `specs/015-torchscript-load-node/contracts/external-checkpoint-load-contract.md` in `OpenYourBox/Source/PluginProcessor.cpp`
- [x] T020 [US1] Implement Clear → empty status, clear paths, dry passthrough, drop optional pins in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T021 [P] [US1] Add LiveGraphEngine coverage for empty passthrough and ready forward with a fixture factory in `Tests/LiveGraphEngineTests.cpp`

**Checkpoint**: US1 complete — place, browse, hear, swap, clear work for forward-only checkpoints.

---

## Phase 4: User Story 2 — Clear Failure and Shape Feedback (Priority: P1)

**Goal**: Missing/invalid files and illegal shapes produce recoverable errors; silence vs passthrough policies hold; channel overrides drive shape checks.

**Independent Test**: `quickstart.md` Scenario B — missing/invalid path; override mismatch refused; empty passthrough vs error silence.

### Implementation for User Story 2

- [x] T022 [US2] Missing/unreadable/invalid checkpoint → recoverable error message; silence if no prior model; keep prior model if present in `OpenYourBox/Source/PluginProcessor.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [x] T023 [US2] Reject directory selection and non-accepted extensions with clear messages in `OpenYourBox/Source/PluginEditor.cpp` / `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [x] T024 [US2] Auto-infer input/output (and latent when applicable) channel counts on successful load; prefill editable override fields per `specs/015-torchscript-load-node/contracts/pin-surface-and-channels-contract.md` in `OpenYourBox/Source/PluginProcessor.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T025 [US2] Property UI for inferred + override channel fields and Reset-to-inferred in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [x] T026 [US2] Shape Integrity: effective channels (`override ?? inferred`) refuse illegal cables; revalidate on override change in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T027 [US2] Incomplete-shape state when inference fails and overrides missing (connections not legal) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T028 [P] [US2] Add tests for error silence, prior-model retain, and override shape refuse in `Tests/LiveGraphEngineTests.cpp` / graph tests as appropriate

**Checkpoint**: US2 complete — failures are safe and shapes are gateable via inferred/override channels.

---

## Phase 5: User Story 3 — Persist Path Across Session and Preset (Priority: P2)

**Goal**: Path and channel overrides survive graph/preset save/load; registry re-prepares on restore; missing files restore path + error without crash.

**Independent Test**: `quickstart.md` Scenario C — save/reload with path + override; missing file after move.

### Implementation for User Story 3

- [x] T029 [US3] Persist `artifactPath`/`weightsPath`, `external_load` origin, override/inferred channel fields with graph and presets in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T030 [US3] On `applyPatchSnapshot` / session restore: re-call `prepareExternalArtifact` for each `externalLoad` with existing file per `specs/015-torchscript-load-node/contracts/persist-and-restore-contract.md` in `OpenYourBox/Source/PluginProcessor.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T031 [US3] On restore with missing file: keep path string, status error, silence, no crash in `OpenYourBox/Source/PluginProcessor.cpp`
- [x] T032 [US3] Registry sharing: multiple nodes same path; clearing one must not break others (refcount or leave entry until unused) in `OpenYourBox/Source/PluginProcessor.cpp`
- [x] T033 [P] [US3] Add serialize/restore coverage for `external_load` path + overrides in graph/preset tests under `Tests/`

**Checkpoint**: US3 complete — relaunch restores ready models or clear missing-file errors.

---

## Phase 6: User Story 4 — Opaque Frozen Processing + RAVE Surface (Priority: P3)

**Goal**: External loads behave as Gold (no Unfreeze); encode/decode latent pins and fidelity match trained RAVE when advertised; Control pin only when conditioning advertised.

**Independent Test**: `quickstart.md` Scenarios D–F — encode/decode pins + fidelity; Control optional; Unfreeze disabled; Clear returns to passthrough.

### Implementation for User Story 4

- [x] T034 [US4] After load with `hasEncodeDecode`: morph latent in/out pins (trained RAVE labels/routing, decode-from-latent) in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T035 [US4] After load with conditioning: add Control pin only then; unconnected = unconditioned forward in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [x] T036 [US4] Remove latent/Control pins when clearing or loading a file lacking those capabilities; revalidate cables in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [x] T037 [US4] Fidelity control when encode/decode + `compactnessReady`; hide/disable with explanation otherwise; wire `copyCompactnessFromArtifact` for external loads in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [x] T038 [US4] Confirm Gold chrome / non-randomizable / Unfreeze disabled for `externalLoad` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [x] T039 [P] [US4] Add encode/decode + optional conditioning fixture coverage in `Tests/LiveGraphEngineTests.cpp`

**Checkpoint**: US4 complete — pretrained RAVE-style surface parity with trained Gold boxes for capable checkpoints.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and cleanup across stories.

- [x] T040 [P] Run and record `specs/015-torchscript-load-node/quickstart.md` Scenarios A–F against a Debug/Release build
- [x] T041 [P] Mark TODO item for loading external RAVE checkpoints done or link to this spec in `TODO.md`
- [x] T042 Code cleanup: ensure no audio-thread `jit::load` / file I/O; document `prepareExternalArtifact` with doxygen in `OpenYourBox/Source/PluginProcessor.h`
- [x] T043 Verify sample-rate mismatch best-effort warning path (if detectable) does not hard-fail in `OpenYourBox/Source/graph/NodeRenderer.cpp` / `OpenYourBox/Source/PluginProcessor.cpp`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: After Foundational — MVP
- **US2 (Phase 4)**: After US1 load path exists (needs prepare + status); can overlap UI once T017 works
- **US3 (Phase 5)**: After US1 path fields + US2 overrides exist
- **US4 (Phase 6)**: After US1 ready path; pin morph can follow US2 shape helpers
- **Polish (Phase 7)**: After desired stories complete

### User Story Dependencies

- **US1 (P1)**: Foundation only — MVP hear path
- **US2 (P1)**: Builds on US1 prepare/status; independently testable for errors/shapes
- **US3 (P2)**: Needs persisted path/overrides from US1/US2
- **US4 (P3)**: Needs ready load; independently testable with encode/decode fixtures

### Parallel Opportunities

- Phase 1 inventory tasks T002–T007 are [P]
- After Foundation: US1 implementation sequential within story; T021 tests [P] with late US1 work
- US2 T028, US3 T033, US4 T039 tests [P] within their stories
- Polish T040–T041 [P]

### Parallel Example: After Foundation

```bash
# Inventory already done; foundation complete, then:
# Primary track — US1 load/hear
Task: "T015 property panel Browse/Clear in NodeRenderer.cpp"
Task: "T016 FileChooser → prepareExternalArtifact"

# After T017 success path:
Task: "T021 LiveGraphEngineTests empty/ready coverage"  # [P]
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 Setup
2. Complete Phase 2 Foundational
3. Complete Phase 3 US1 (place → browse → hear → swap → clear)
4. **STOP and VALIDATE** via quickstart Scenario A
5. Demo MVP before US2–US4

### Incremental Delivery

1. Setup + Foundational → empty Gold loader + prepare API
2. US1 → hear external `.pt`
3. US2 → safe errors + channel overrides
4. US3 → persist/restore + registry rehydrate
5. US4 → RAVE encode/decode / Control / fidelity parity
6. Polish → full quickstart A–F

### Suggested MVP Scope

**US1 only** (plus Foundational): Factory TorchScript Load, browse valid checkpoint, hear forward processing, clear/swap — enough to prove external pretrained models in-graph.

---

## Notes

- [P] = different files, no dependency on incomplete sibling tasks
- Reuse `TorchScriptBlackBoxFactory` / published registry; do not add Python worker types for v1
- Prefer `BlackBoxOrigin::externalLoad` over a new `NodeType`
- Commit after each task or logical group; stop at checkpoints to validate independently
