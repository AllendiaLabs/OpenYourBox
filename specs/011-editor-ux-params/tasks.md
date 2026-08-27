---
description: "Task list for Editor UX & Parameter Flexibility"
---

# Tasks: Editor UX & Parameter Flexibility

**Input**: Design documents from `specs/011-editor-ux-params/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Spec does not request TDD. Optional targeted unit tests noted where they strongly reduce risk; no test-first gate.

**Organization**: Tasks grouped by user story (US1–US6) for independent validation checkpoints.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- Graph: `OpenYourBox/Source/graph/`
- UI: `OpenYourBox/Source/ui/`
- Library: `OpenYourBox/Source/library/`
- DSP: `OpenYourBox/Source/dsp/`
- Backend: `Backend/`
- Tests: `Tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design artifacts and touch-points before code changes.

- [X] T001 Verify design artifact cross-links and story order in `specs/011-editor-ux-params/plan.md`
- [X] T002 [P] Inventory current copy-list parse/commit and breadcrumb/palette width call sites in `OpenYourBox/Source/graph/GraphTypes.h`, `OpenYourBox/Source/graph/NodeGraph.cpp`, and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T003 [P] Inventory LeakyReLU hardcoded `0.01` sites in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`, `OpenYourBox/Source/dsp/TCNModel.cpp`, `Backend/train_worker.py`, and `Backend/freeze_worker.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared helpers for nest copy vectors and dividing-set math used by US1/US2 (and later invalidation).

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Add ancestor copy-count vector + dividing-set helpers (outer→inner, `D(C)` per `specs/011-editor-ux-params/contracts/copy-list-tiling-contract.md`) with Doxygen in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T005 Add `copyListInvalid` (or equivalent) flag and authored-length-preserving resize policy notes on `NodeProperty` in `OpenYourBox/Source/graph/GraphTypes.h` (stop silent pad/truncate as the only path)
- [X] T006 Wire `effectiveCopyCount` / ancestor walk to expose ordered copy vector for tiling in `OpenYourBox/Source/graph/NodeGraph.cpp`

**Checkpoint**: Foundation ready — dividing-set API available; property can represent authored L ≠ P without forced mute resize.

---

## Phase 3: User Story 1 — Nested-group parameter list tiling (Priority: P1) 🎯 MVP

**Goal**: Accept L ∈ dividing set; store authored L; expand to P under the hood; editable short field + read-only P preview; re-tile or flag invalid on ancestor copy-count change.

**Independent Test**: Nest O×M×N with P≥8; commit lengths 1, N, M×N, P; refuse illegal length; change outer N → re-tile or invalid per clarifications (`specs/011-editor-ux-params/quickstart.md` §1).

### Implementation for User Story 1

- [X] T007 [US1] Extend `parsePropertyCopyList` to accept L ∈ D(C) and expand by tiling in `OpenYourBox/Source/graph/GraphTypes.h` per `specs/011-editor-ux-params/contracts/copy-list-tiling-contract.md`
- [X] T008 [US1] Update `setPropertyCopyValues` / `setFloatPropertyCopyValues` to store authored L and derive expanded P for runtime in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T009 [US1] Stop `ensurePropertyCopyCount` from silently rewriting authored L; integrate invalid flag path in `OpenYourBox/Source/graph/GraphTypes.h` / `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T010 [US1] On `setGroupCopies`, re-tile when L still valid else mark `copyListInvalid` with user-visible message in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T011 [US1] Persist authored vectors without forcing size==P on save/load in `OpenYourBox/Source/graph/NodeGraph.cpp` (property ValueTree CSV path)
- [X] T012 [US1] Property UI: editable authored CSV + read-only expanded P preview when P>1 in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T013 [US1] Show clear refuse/invalid messages for bad lengths and post-nest invalid state in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T014 [P] [US1] Add dividing-set / tiling unit coverage in `Tests/` (new or existing graph property test file registered in `CMakeLists.txt`)

**Checkpoint**: US1 complete — short lists authorable; preview shows P; nest changes re-tile or flag.

---

## Phase 4: User Story 2 — Preserve-input keyword `in` (Priority: P1)

**Goal**: Shape fields accept `in` (and all-`in` lists that tile); resolve to paired input dims; refuse mix/unsupported fields.

**Independent Test**: Set `features`/`channels` to `in`; change upstream/copy counts → outputs follow; `in, 32` refused (`quickstart.md` §2).

### Implementation for User Story 2

- [X] T015 [US2] Extend property parse path to accept reserved token `in` (case-sensitive) for bindable integer fields in `OpenYourBox/Source/graph/GraphTypes.h` per `specs/011-editor-ux-params/contracts/preserve-in-keyword-contract.md`
- [X] T016 [US2] Persist `in` binding intent (not only resolved ints) on `NodeProperty` serialize/deserialize in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T017 [US2] Resolve `in` → paired input channels/features during shape refresh / `setProperty` in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T018 [US2] Re-resolve bindings when ancestor copies or upstream shapes change; treat unresolved as illegal in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T019 [US2] Property editor accepts `in` / list-of-`in`, refuses mixes and non-bindable fields with messages in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T020 [P] [US2] Add `in` parse/resolve unit coverage in `Tests/` (same or adjacent test target as T014)

**Checkpoint**: US2 complete — `in` keeps shapes legal across copy-count changes.

---

## Phase 5: User Story 3 — Project structure & library subpart insert (Priority: P2)

**Goal**: Collapsible Project structure under Library; expand library group entries; insert root or nested subpart; save nested targets.

**Independent Test**: Navigate/save from Project structure; expand library entry; insert root + nested child without invented external cables (`quickstart.md` §3).

### Implementation for User Story 3

- [X] T021 [US3] Add collapsible **Project structure** section under Library in left palette in `OpenYourBox/Source/graph/NodeRenderer.cpp` / `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp` per `specs/011-editor-ux-params/contracts/project-structure-library-tree-contract.md`
- [X] T022 [US3] Build live hierarchy tree (groups/elements) from `NodeGraph` and refresh on graph edits in `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp` (or dedicated helper beside it)
- [X] T023 [US3] Navigate: activating a group in Project structure calls `setCanvasFocus` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T024 [US3] Enable save-to-library from Project structure targeted nested box (reuse `exportBox`) in `OpenYourBox/Source/graph/NodeRenderer.cpp` / `OpenYourBox/Source/library/UserBoxLibrary.cpp`
- [X] T025 [US3] Expandable member tree for group library entries from snapshot payload in `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp`
- [X] T026 [US3] Insert subpart API: clone nested subtree only (no external cables) in `OpenYourBox/Source/library/UserBoxLibrary.cpp` / `OpenYourBox/Source/graph/NodeGraph.cpp` (`importBox` variant or path arg)
- [X] T027 [US3] DnD/place payload supports `(entryId, optional nestedPathOrId)` in `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp` and drop handling in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T028 [US3] Sort user library entries by display name (case-insensitive) within each folder when rendering in `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp` (and sort nested member rows by label when expanding an entry); keep order correct after rename/save per FR-004a

**Checkpoint**: US3 complete — live tree + library subtree insert/save + name-ordered lists.

---

## Phase 6: User Story 4 — Hierarchy sticky trail (Priority: P2)

**Goal**: When navigating up, previously opened children stay visible/clickable until branching elsewhere.

**Independent Test**: Parent→Child→Grandchild, up to Parent, one-click return; sibling branch clears old spine (`quickstart.md` §4).

### Implementation for User Story 4

- [X] T029 [US4] Add `stickySpine` (ordered GroupIds) on viewport/session state beside `focusedGroupId` in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T030 [US4] Update `setCanvasFocus` / open-child paths to extend or reset spine per `specs/011-editor-ux-params/contracts/hierarchy-sticky-trail-contract.md` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T031 [US4] Render sticky descendants in `renderScopeBreadcrumb` as clickable entries in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T032 [US4] Prune spine on delete/ungroup of sticky ids in `OpenYourBox/Source/graph/NodeGraph.cpp` / `OpenYourBox/Source/graph/NodeRenderer.cpp`

**Checkpoint**: US4 complete — sticky trail matches FR-003.

---

## Phase 7: User Story 5 — Resizable side menus (Priority: P3)

**Goal**: Drag-resize left palette and right Info column with min/max and min canvas width.

**Independent Test**: Drag both edges; clamps apply; widths stable for session (`quickstart.md` §5).

### Implementation for User Story 5

- [X] T033 [US5] Replace fixed left `200.f` palette width with splitter + clamps in `OpenYourBox/Source/graph/NodeRenderer.cpp` per `specs/011-editor-ux-params/contracts/resizable-side-menus-contract.md`
- [X] T034 [US5] Replace fixed right/`-340` graph inset with splitter + clamps in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T035 [US5] Store `leftWidthPx` / `rightWidthPx` for session (and prefs if UI prefs already exist) in `OpenYourBox/Source/PluginEditor.cpp` / `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T036 [US5] Enforce `minCanvasWidth` while dragging so canvas stays usable in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`

**Checkpoint**: US5 complete — both side menus resizable.

---

## Phase 8: User Story 6 — LeakyReLU negative slope (Priority: P3)

**Goal**: Editable `negative_slope` default 0.01, range [0,1], refuse OOR; live + train + freeze honor value.

**Independent Test**: Default 0.01 → set 0.2 persists; `-0.1` refused (`quickstart.md` §6).

### Implementation for User Story 6

- [X] T037 [US6] Add `negative_slope` real property (default 0.01, range [0,1], refuse OOR) on activation factory in `OpenYourBox/Source/graph/NodeGraph.cpp` per `specs/011-editor-ux-params/contracts/leakyrelu-negative-slope-contract.md`
- [X] T038 [US6] Show/edit `negative_slope` only when activation is LeakyReLU in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T039 [US6] Replace hardcoded `0.01` with property in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T040 [P] [US6] Replace hardcoded `0.01` with property in `OpenYourBox/Source/dsp/TCNModel.cpp`
- [X] T041 [P] [US6] Pass element `negative_slope` into LeakyReLU construction in `Backend/train_worker.py`
- [X] T042 [P] [US6] Pass element `negative_slope` into LeakyReLU construction in `Backend/freeze_worker.py`

**Checkpoint**: US6 complete — slope editable end-to-end.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Validation and cleanup across stories.

- [X] T043 Run manual scenarios in `specs/011-editor-ux-params/quickstart.md` and record gaps
- [X] T044 [P] Mark completed TODO bullets for this feature in `TODO.md` (lines covering Project structure, menus, hierarchy trail, `in`, nested lists, LeakyReLU slope)
- [X] T045 [P] Doxygen pass on new public helpers in `OpenYourBox/Source/graph/GraphTypes.h` and related `.h` APIs touched by this feature
- [X] T046 Fix any Shape Integrity regressions from `in` + tiling (illegal cable messaging) in `OpenYourBox/Source/graph/NodeGraph.cpp`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Start immediately
- **Foundational (Phase 2)**: After Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: After Foundational — MVP
- **US2 (Phase 4)**: After Foundational; benefits from US1 tiling for all-`in` lists (implement after US1 or share T007 helpers)
- **US3 (Phase 5)**: After Foundational — independent of US1/US2
- **US4 (Phase 6)**: After Foundational — independent
- **US5 (Phase 7)**: After Foundational — independent
- **US6 (Phase 8)**: After Foundational — independent
- **Polish (Phase 9)**: After desired stories complete

### User Story Dependencies

- **US1 (P1)**: No story dependencies — MVP
- **US2 (P1)**: Soft dependency on US1 dividing-set/list parse for list-of-`in`
- **US3 (P2)**: Independent (library/UI)
- **US4 (P2)**: Independent (breadcrumb)
- **US5 (P3)**: Independent (chrome layout)
- **US6 (P3)**: Independent (activation property + DSP/Python)

### Parallel Opportunities

- After Phase 2: US3, US4, US5, US6 can proceed in parallel with US1/US2
- T039–T042 (US6) parallel across C++/Python files after T037
- T002/T003, T014/T020, T044/T045 marked [P]

---

## Parallel Example: After Foundation

```bash
# MVP path:
Task: "T007–T014 US1 copy-list tiling"

# Parallel track (different files):
Task: "T021–T028 US3 Project structure + library tree"
Task: "T029–T032 US4 sticky trail"
Task: "T033–T036 US5 resizable menus"
Task: "T037–T042 US6 LeakyReLU slope"
```

---

## Parallel Example: User Story 6

```bash
Task: "Replace hardcoded 0.01 in OpenYourBox/Source/dsp/TCNModel.cpp"
Task: "Pass negative_slope in Backend/train_worker.py"
Task: "Pass negative_slope in Backend/freeze_worker.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 Setup → Phase 2 Foundational
2. Phase 3 US1 (tiling + preview + invalid-on-nest-change)
3. **STOP** — validate `quickstart.md` §1
4. Then US2 (`in`) for the other P1

### Incremental Delivery

1. US1 → demo nested editing
2. US2 → shape binding
3. US3 → library/project reuse
4. US4 → navigation sticky trail
5. US5 → chrome resize
6. US6 → LeakyReLU slope
7. Polish + full `quickstart.md`

### Suggested MVP Scope

**US1 only** (T001–T014): nested copy-list tiling is the highest-leverage graph-editing fix and unblocks list-of-`in` in US2.

---

## Notes

- [P] = different files, no wait on incomplete sibling tasks
- Do not keep silent `ensurePropertyCopyCount` pad/truncate as the sole behavior after US1
- Library subpart insert must not invent external cables (006 + project-structure contract)
- LeakyReLU OOR: refuse, never clamp
- Commit after each task or logical group; stop at story checkpoints
