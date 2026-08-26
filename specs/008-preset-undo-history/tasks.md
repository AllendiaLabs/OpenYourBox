---
description: "Task list for Preset Management & Undo/Redo History"
---

# Tasks: Preset Management & Undo/Redo History

**Input**: Design documents from `specs/008-preset-undo-history/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Plan lists C++ tests under `Tests/`; include focused unit/integration tasks per story (not full TDD gate). Validate end-to-end via `quickstart.md`.

**Organization**: Tasks grouped by user story for independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`US1`–`US5`) on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- Graph: `OpenYourBox/Source/graph/`
- Library: `OpenYourBox/Source/library/`
- State: `OpenYourBox/Source/state/`
- UI: `OpenYourBox/Source/ui/`
- Tests: `Tests/`
- Specs: `specs/008-preset-undo-history/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design anchors and wire new sources into the build before implementation.

- [ ] T001 Verify design artifact cross-links (`plan.md`, `data-model.md`, contracts, `quickstart.md`) and path conventions in `specs/008-preset-undo-history/plan.md`
- [ ] T002 [P] Add `presetsDirectory()` next to `boxesDirectory()` in `OpenYourBox/Source/library/UserDataPaths.h` (`userDataRoot()/UserPresets`)
- [ ] T003 [P] Add `state/` and preset library/panel sources plus new test targets to `CMakeLists.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared `PatchSnapshot` serialize/apply path, `EditHistory` scaffold, and `UserPresetLibrary` skeleton required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T004 Create `PatchSnapshot` types and capture/apply API stubs in `OpenYourBox/Source/state/PatchSnapshot.h` / `OpenYourBox/Source/state/PatchSnapshot.cpp` per `specs/008-preset-undo-history/contracts/patch-snapshot-contract.md`
- [ ] T005 Implement snapshot capture from APVTS + `GraphDocument` + weights/metadata (parity with current host state fields) in `OpenYourBox/Source/state/PatchSnapshot.cpp`
- [ ] T006 Implement snapshot apply on GUI thread with restore/`suppressHistory` flag and atomic runtime publish in `OpenYourBox/Source/state/PatchSnapshot.cpp` and `OpenYourBox/Source/PluginProcessor.cpp`
- [ ] T007 Refactor `OpenYourBoxAudioProcessor::getStateInformation` / `setStateInformation` to use `PatchSnapshot` in `OpenYourBox/Source/PluginProcessor.cpp` without changing DAW round-trip behavior
- [ ] T008 [P] Scaffold `EditHistory` (`UndoManager` façade, depth cap ≥ 50, suppress-on-apply) in `OpenYourBox/Source/state/EditHistory.h` / `OpenYourBox/Source/state/EditHistory.cpp` per `specs/008-preset-undo-history/contracts/edit-history-contract.md`
- [ ] T009 [P] Scaffold `UserPresetLibrary` (index load/save stubs, entry DTOs) in `OpenYourBox/Source/library/UserPresetLibrary.h` / `OpenYourBox/Source/library/UserPresetLibrary.cpp`
- [ ] T010 [P] Scaffold empty `UserPresetPanel` ImGui shell in `OpenYourBox/Source/ui/UserPresetPanel.h` / `OpenYourBox/Source/ui/UserPresetPanel.cpp`
- [ ] T011 Confirm `NodeGraph::toValueTree` / restore path is sufficient for full patch recall (groups, layout, Gold paths) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T012 [P] Add `PatchSnapshotTests.cpp` round-trip harness (graph + params; weights when present) in `Tests/PatchSnapshotTests.cpp` and register in `CMakeLists.txt`

**Checkpoint**: Shared snapshot path, history/library skeletons, and host-state refactor ready — story work can begin.

---

## Phase 3: User Story 1 - Save and Recall Named Presets (Priority: P1) 🎯 MVP

**Goal**: Save As / Save / load full sonic presets; track current preset name + dirty; persist catalog under `UserPresets`; load without stopping audio.

**Independent Test**: Quickstart §1 — Save As `PatchA`, edit → dirty, Save, Save As `PatchB`, load `PatchA`; structure/sonic match; chrome shows current/dirty correctly.

### Tests for User Story 1

- [ ] T013 [P] [US1] Add `UserPresetLibrary` CRUD/save-load tests in `Tests/UserPresetLibraryTests.cpp` and register in `CMakeLists.txt`

### Implementation for User Story 1

- [ ] T014 [US1] Implement catalog index (`index.json`) atomic load/save and entry folder layout in `OpenYourBox/Source/library/UserPresetLibrary.cpp`
- [ ] T015 [US1] Implement Save As (unique name → new entry; colliding name → overwrite confirm callback) writing `PatchSnapshot` + artifacts in `OpenYourBox/Source/library/UserPresetLibrary.cpp`
- [ ] T016 [US1] Implement Save (overwrite current entry payload) and load-by-id/name returning `PatchSnapshot` in `OpenYourBox/Source/library/UserPresetLibrary.cpp`
- [ ] T017 [US1] Embed weights/Gold artifacts into preset payload (or artifacts/ with rewritten relative paths) for full sonic recall in `OpenYourBox/Source/library/UserPresetLibrary.cpp` / `OpenYourBox/Source/state/PatchSnapshot.cpp`
- [ ] T018 [US1] Add CurrentPreset session state (id, name, dirty, baseline) on editor/processor in `OpenYourBox/Source/PluginEditor.h` / `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T019 [US1] Wire Save / Save As / Load actions: capture→library→apply snapshot; set current name; clear/set dirty in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/PluginProcessor.cpp`
- [ ] T020 [US1] Mark dirty on undoable patch-affecting edits; clear dirty on successful Save/Save As; refuse empty names with message in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T021 [US1] Implement ImGui Presets panel list + Save / Save As / Load controls and current/dirty chrome in `OpenYourBox/Source/ui/UserPresetPanel.cpp`
- [ ] T022 [US1] Mount Presets panel in the plugin UI chrome in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T023 [US1] Refuse corrupt/unrestorable preset load with clear message and unchanged live patch in `OpenYourBox/Source/library/UserPresetLibrary.cpp` / `OpenYourBox/Source/PluginEditor.cpp`

**Checkpoint**: Named Save/Save As/load works with current+dirty; catalog survives relaunch; DAW transport uninterrupted.

---

## Phase 4: User Story 2 - Browse, Rename, and Delete Presets (Priority: P1)

**Goal**: Browse catalog, rename unique names, delete with confirmation, empty-state UX.

**Independent Test**: Quickstart §2 — two presets, rename one, cancel then confirm delete; relaunch shows updated catalog.

### Implementation for User Story 2

- [ ] T024 [P] [US2] Implement rename (unique; conflict policy) and delete (remove index row + payload folder) in `OpenYourBox/Source/library/UserPresetLibrary.cpp`
- [ ] T025 [US2] Add rename/delete confirm dialogs and empty-state prompt in `OpenYourBox/Source/ui/UserPresetPanel.cpp`
- [ ] T026 [US2] If deleted/renamed entry is current, update CurrentPreset chrome accordingly in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T027 [US2] Extend `Tests/UserPresetLibraryTests.cpp` for rename, delete, and reload-from-disk catalog consistency

**Checkpoint**: Catalog management complete for MVP preset library alongside US1.

---

## Phase 5: User Story 3 - Undo and Redo Graph and Parameter Edits (Priority: P1)

**Goal**: Session undo/redo for patch-affecting edits; coalesce gestures; randomize as one step; exclude pan/zoom/selection; shortcuts + UI.

**Independent Test**: Quickstart §4–7 — ≥5 edits undo/redo; coalesced knob; randomize undo; pan/zoom do not create steps.

### Tests for User Story 3

- [ ] T028 [P] [US3] Add `EditHistory` depth/coalesce/redo-clear/suppress tests in `Tests/EditHistoryTests.cpp` and register in `CMakeLists.txt`

### Implementation for User Story 3

- [ ] T029 [US3] Implement before/after `PatchSnapshot` (+ CurrentPreset) history steps and push/undo/redo/apply in `OpenYourBox/Source/state/EditHistory.cpp`
- [ ] T030 [US3] Enforce max depth ≥ 50 (drop oldest) and clear redo on new edit in `OpenYourBox/Source/state/EditHistory.cpp`
- [ ] T031 [US3] Add beginGesture/endGesture coalescing for continuous controls and node layout drags in `OpenYourBox/Source/state/EditHistory.cpp` and call sites in `OpenYourBox/Source/PluginEditor.cpp` / `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [ ] T032 [US3] Record discrete graph edits (add/remove/connect/group/ungroup/layout commit) as single steps in `OpenYourBox/Source/graph/NodeGraph.cpp` / `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T033 [US3] Record weight randomization (and similar one-shot sonic actions) as one history step in `OpenYourBox/Source/PluginEditor.cpp` / randomization call path
- [ ] T034 [US3] Ensure pan/zoom/selection never push history or set dirty in `OpenYourBox/Source/graph/NodeRenderer.cpp` / `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T035 [US3] Wire Undo/Redo UI buttons + platform shortcuts to the same `EditHistory` API; disable when empty in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T036 [US3] Own `EditHistory` on processor/editor lifecycle (session-scoped; destroyed with instance) in `OpenYourBox/Source/PluginProcessor.h` / `OpenYourBox/Source/PluginEditor.cpp`

**Checkpoint**: Undo/redo usable for creative edits; view-only navigation ignored; audio continues.

---

## Phase 6: User Story 4 - Export and Import Presets for Sharing (Priority: P2)

**Goal**: Portable single-file export/import with full sonic payload; overwrite confirm on name collision; refuse invalid packages.

**Independent Test**: Quickstart §9 — export, delete catalog entry, import, load; sonic match; garbage import refused.

### Implementation for User Story 4

- [ ] T037 [P] [US4] Define portable package manifest + pack/unpack (patch + artifacts) in `OpenYourBox/Source/library/UserPresetLibrary.cpp` per `specs/008-preset-undo-history/contracts/preset-catalog-contract.md`
- [ ] T038 [US4] Implement Export selected entry to user-chosen file in `OpenYourBox/Source/library/UserPresetLibrary.cpp` and `OpenYourBox/Source/ui/UserPresetPanel.cpp`
- [ ] T039 [US4] Implement Import with name prompt; overwrite confirm on collision; refuse invalid packages without catalog mutation in `OpenYourBox/Source/library/UserPresetLibrary.cpp` / `OpenYourBox/Source/ui/UserPresetPanel.cpp`
- [ ] T040 [US4] Extend `Tests/UserPresetLibraryTests.cpp` for export→import round-trip and invalid-package refusal

**Checkpoint**: Presets shareable across machines/catalogs with overwrite safety.

---

## Phase 7: User Story 5 - Preset Load Interacts Predictably with Undo History (Priority: P2)

**Goal**: Successful preset load is one undoable step restoring prior patch **and** prior CurrentPreset (name + dirty); redo restores loaded state.

**Independent Test**: Quickstart §8 — edit, load preset, undo once → pre-load patch + prior current/dirty; redo → loaded preset.

### Implementation for User Story 5

- [ ] T041 [US5] On successful preset load, push one `EditHistory` step capturing before/after snapshots and CurrentPreset in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T042 [US5] Ensure failed load pushes no history step and leaves live patch/current unchanged in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T043 [US5] On undo/redo of load step, restore CurrentPreset name+dirty with the patch in `OpenYourBox/Source/state/EditHistory.cpp` / `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T044 [US5] Extend `Tests/EditHistoryTests.cpp` for preset-load single-step and CurrentPreset restore

**Checkpoint**: Preset load and edit history compose without surprising state loss.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Regression, performance, and documentation polish across stories.

- [ ] T045 [P] Add/extend processor host-state regression covering `PatchSnapshot`-based `getStateInformation` / `setStateInformation` in `Tests/ProcessorIntegrationTests.cpp`
- [ ] T046 Verify completed train/freeze patch mutations push a single history step once applied (not mid-job) in `OpenYourBox/Source/PluginEditor.cpp` / train-freeze completion hooks
- [ ] T047 [P] Run through `specs/008-preset-undo-history/quickstart.md` scenarios and note any gaps in `specs/008-preset-undo-history/quickstart.md`
- [ ] T048 Confirm UI shows disabled Undo/Redo affordances and current/dirty chrome under empty/clean states in `OpenYourBox/Source/PluginEditor.cpp` / `OpenYourBox/Source/ui/UserPresetPanel.cpp`
- [ ] T049 [P] Update `TODO.md` to mark preset management / undo-redo items done or link to this feature when validated

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **blocks all user stories**
- **US1 (Phase 3)**: Depends on Foundational — MVP preset save/load
- **US2 (Phase 4)**: Depends on US1 catalog/panel basics (rename/delete/browse)
- **US3 (Phase 5)**: Depends on Foundational (`EditHistory` + `PatchSnapshot`); can proceed in parallel with US2 after US1 if Save/Load already land dirty hooks carefully — prefer after US1 dirty wiring
- **US4 (Phase 6)**: Depends on US1 library payload format
- **US5 (Phase 7)**: Depends on US1 load path + US3 `EditHistory`
- **Polish (Phase 8)**: Depends on stories intended for the release cut

### User Story Dependencies

- **US1 (P1)**: After Foundational — no story dependency — **MVP**
- **US2 (P1)**: After US1 save/load/catalog
- **US3 (P1)**: After Foundational; integrates dirty hooks from US1
- **US4 (P2)**: After US1 payload format
- **US5 (P2)**: After US1 + US3

### Within Each User Story

- Library/state models before UI wiring
- Core actions before confirm/empty/error polish
- Story tests alongside or immediately after implementation
- Checkpoint before moving to the next priority

### Parallel Opportunities

- T002 / T003 in Setup
- T008 / T009 / T010 / T012 in Foundational (after T004 stubs exist)
- T013 with early US1 library work
- T024 parallelizable with panel work once library rename/delete APIs exist
- T028 parallel with US3 implementation start
- T037 packing format can start once US1 payload layout is stable
- T045 / T047 / T049 in Polish

---

## Parallel Example: User Story 1

```bash
# After Foundational checkpoint:
Task: "T013 UserPresetLibrary tests harness in Tests/UserPresetLibraryTests.cpp"
Task: "T014 index.json load/save in OpenYourBox/Source/library/UserPresetLibrary.cpp"

# After library Save As/Load APIs:
Task: "T021 Presets panel controls in OpenYourBox/Source/ui/UserPresetPanel.cpp"
Task: "T018 CurrentPreset state in OpenYourBox/Source/PluginEditor.cpp"
```

---

## Parallel Example: User Story 3

```bash
Task: "T028 EditHistory tests in Tests/EditHistoryTests.cpp"
Task: "T029 before/after snapshot steps in OpenYourBox/Source/state/EditHistory.cpp"
# Then wire call sites:
Task: "T031 gesture coalesce call sites in PluginEditor/NodeRenderer"
Task: "T034 exclude pan/zoom/selection in NodeRenderer/PluginEditor"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: US1 Save/Save As/Load + current/dirty
4. **STOP and VALIDATE** via Quickstart §1
5. Demo named preset recall

### Incremental Delivery

1. Setup + Foundational → shared snapshot path live for DAW state
2. US1 → named presets MVP
3. US2 → catalog hygiene
4. US3 → undo/redo safety net
5. US4 → share/backup
6. US5 → load↔history coherence
7. Polish → quickstart + host regression

### Parallel Team Strategy

1. Team completes Setup + Foundational together
2. Then:
   - Dev A: US1 → US2 → US4
   - Dev B: US3 → US5
3. Integrate at US5 (load as one undo step)

---

## Notes

- [P] = different files, no incomplete-task dependencies
- [USn] maps to spec user stories for traceability
- Presets ≠ boxes: never reuse `UserBoxLibrary` storage
- Apply snapshots only on GUI thread; never allocate on the audio thread
- Commit after each task or logical group; stop at checkpoints to validate independently
