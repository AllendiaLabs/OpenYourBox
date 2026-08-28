---
description: "Task list for Box Property Panel UX"
---

# Tasks: Box Property Panel UX

**Input**: Design documents from `specs/013-box-property-panel-ux/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Spec does not require TDD. Targeted unit tests included where the plan lists `Tests/` coverage (disconnect/reparent/undo); no test-first gate. UI flows validated via `quickstart.md`.

**Organization**: Tasks grouped by user story (US1–US5) for independent validation checkpoints.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- Graph: `OpenYourBox/Source/graph/`
- UI: `OpenYourBox/Source/ui/`
- Library: `OpenYourBox/Source/library/`
- Tests: `Tests/`
- Specs: `specs/013-box-property-panel-ux/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design artifacts and inventory UI/graph touch-points before code changes.

- [X] T001 Verify design artifact cross-links (spec clarifications → contracts → plan) in `specs/013-box-property-panel-ux/plan.md`
- [X] T002 [P] Inventory `renderNode` / `renderGroup` inline property, randomize, and size-measure paths in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/graph/NodeRenderer.h`
- [X] T003 [P] Inventory `SideTabs` / selection sync / drag tracking in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T004 [P] Inventory Project structure click/`setCanvasFocus`, palette/library DnD payloads, and `addToGroup`/`importBox` in `OpenYourBox/Source/graph/NodeRenderer.cpp`, `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp`, and `OpenYourBox/Source/graph/NodeGraph.cpp`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared selection context, Parameters tab shell, and graph disconnect/reparent APIs required by later stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Add session `SelectionContext` (None / Live / LibraryInspect / Multi) types and accessors per `specs/013-box-property-panel-ux/data-model.md` in `OpenYourBox/Source/graph/GraphTypes.h` (or `OpenYourBox/Source/graph/NodeRenderer.h` if kept UI-local)
- [X] T006 Wire canvas primary selection into `SelectionContext::Live` via `synchronizeSelection` / `getPrimarySelectedNodeId` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T007 Add empty **Parameters** tab to `SideTabs` and a one-shot force-tab-to-Parameters API per `specs/013-box-property-panel-ux/contracts/parameters-panel-contract.md` in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/PluginEditor.h`
- [X] T008 Extract shared property-row / randomize rendering helpers (callable from Parameters tab) from box chrome into methods on `NodeRenderer` (or a small helper) in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/graph/NodeRenderer.h`
- [X] T009 Implement `disconnectAllLinksForBox` (incident-link cleanup without deleting the box) in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/graph/NodeGraph.h`
- [X] T010 Implement `reparentBoxLikeInsert` (cycle reject → disconnect → detach → attach → new-item placement) per `specs/013-box-property-panel-ux/contracts/structure-hierarchy-dnd-contract.md` in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/graph/NodeGraph.h`
- [X] T011 Ensure reparent/disconnect participate in existing undo/patch gestures via `documentChanged` / history hooks in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/PluginEditor.cpp`
- [X] T012 [P] Add disconnect-all, reparent into group/root, and cycle-reject coverage in `Tests/GraphGroupTests.cpp`
- [X] T013 [P] Add undo/redo coverage for reparent gesture in `Tests/EditHistoryTests.cpp` (or extend `Tests/GraphGroupTests.cpp` if history fixtures live there)

**Checkpoint**: Foundation ready — SelectionContext exists; Parameters tab shell present; graph can disconnect+reparent with tests green.

---

## Phase 3: User Story 1 — Simplified boxes with property panel editing (Priority: P1) 🎯 MVP

**Goal**: Canvas boxes show name + pins only; all parameters and randomize live in the Parameters tab; selecting a box opens Parameters and always switches to that tab.

**Independent Test**: `specs/013-box-property-panel-ux/quickstart.md` Scenario A — slim chrome; click → Parameters; edit without box resize; randomize from panel only.

### Implementation for User Story 1

- [X] T014 [US1] Render live Parameters panel for `SelectionContext::Live` (node + group properties) using extracted helpers in `OpenYourBox/Source/graph/NodeRenderer.cpp` and invoke it from the Parameters tab in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T015 [US1] Force Parameters tab on canvas select/click (even when another SideTab was active) in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T016 [US1] Strip inline parameter fields from `renderNode` (name + pins only) per `specs/013-box-property-panel-ux/contracts/slim-box-chrome-contract.md` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T017 [US1] Strip inline parameter fields, repeats editor, and randomize from `renderGroup`; keep non-stealing name display in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T018 [US1] Move randomize/reset actions into Parameters panel only (element + group) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T019 [US1] Move group name / copy-count (repeats) editors into Parameters so canvas body does not steal focus in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T020 [US1] Multi-select: show concise Parameters state without full edit until single selection in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/PluginEditor.cpp`
- [X] T021 [US1] Confirm property edits use existing change callbacks / undo gestures and do not resize box chrome in `OpenYourBox/Source/graph/NodeRenderer.cpp`

**Checkpoint**: US1 complete — boxes are slim; Parameters is the only edit surface for relocated controls; select always opens Parameters.

---

## Phase 4: User Story 2 — One-click select and drag boxes (Priority: P1)

**Goal**: Press-drag on a box body selects (if needed) and moves in one gesture; pins still own wiring.

**Independent Test**: `quickstart.md` Scenario B — first press-drag moves unselected box; pin drag wires.

### Implementation for User Story 2

- [X] T022 [US2] Verify / adjust node drag tracking so unselected hovered box (no pin) starts drag without a prior activation click in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T023 [US2] Ensure pin/link hover still suppresses box drag (FR-006) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T024 [US2] Confirm `ed::SelectButtonIndex` / `DragButtonIndex` and `syncEditorTransforms` do not require a second click after chrome removal in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T025 [US2] Manually validate overlapping-box topmost drag behavior and document any residual hit-region fix in `OpenYourBox/Source/graph/NodeRenderer.cpp`

**Checkpoint**: US2 complete — one continuous press-drag moves boxes; wiring unchanged.

---

## Phase 5: User Story 3 — Project structure tree integration (Priority: P2)

**Goal**: Structure single-click opens Parameters without changing canvas; group double-click opens the inner canvas; leaf double-click camera-navigates; DnD reparent disconnects then adds like new with highlight; post-drop select + Parameters + destination focus.

**Independent Test**: `quickstart.md` Scenarios C–D — click/dblclick rules; wired element reparent clears cables; cycle reject; undo.

### Implementation for User Story 3

- [X] T026 [US3] Change Project structure single-click to select live box and force Parameters **without** opening group canvases per `specs/013-box-property-panel-ux/contracts/project-structure-navigation-contract.md` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T027 [US3] Implement Project structure double-click: leaf → focus containing canvas + center on box; group → open inner canvas (`setCanvasFocus(group)`) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T028 [US3] Add structure drag source for live rows and eligible drop targets (root + non-cyclic groups) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T029 [US3] Draw drop-target highlight rectangle around target label + nested rows during structure DnD in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T030 [US3] On valid structure drop of a live row, call `reparentBoxLikeInsert`, then select result, force Parameters, and focus destination canvas if needed in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T031 [US3] On invalid structure drop, cancel with no hierarchy/cable mutation and clear highlight in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T032 [US3] Refresh Project structure after undo/redo so stale ids cannot open Parameters in `OpenYourBox/Source/graph/NodeRenderer.cpp`

**Checkpoint**: US3 complete — structure navigates and reparents per contracts; cables cleared on move.

---

## Phase 6: User Story 4 — Library and element-list insert via canvas or Project structure (Priority: P2)

**Goal**: Library click is read-only Parameters inspect; library and element-list drag to canvas or Project structure inserts like today’s add-to-project rules.

**Independent Test**: `quickstart.md` Scenario E — read-only catalog inspect; canvas + structure drops from library and palette.

### Implementation for User Story 4

- [X] T033 [US4] On user-library entry/subpart single-click, set `SelectionContext::LibraryInspect` and force Parameters in `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp` and `OpenYourBox/Source/PluginEditor.cpp`
- [X] T034 [US4] Render read-only Parameters for library snapshot binding (no catalog writes) per `specs/013-box-property-panel-ux/contracts/parameters-panel-contract.md` in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/library/UserBoxLibrary.cpp` / `.h` as needed
- [X] T035 [US4] Accept `OPENYOURBOX_BOX_LIBRARY_ID` drops on Project structure targets → `insertBox`/`importBox` into highlighted scope with new-item placement in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T036 [US4] Accept `OPENYOURBOX_NODE_TYPE` (element list) drops on Project structure targets → `addNode` into highlighted scope with new-item placement in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T037 [US4] After structure insert success: select new instance, force Parameters, focus destination canvas when needed in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T038 [US4] Keep canvas drops for library and palette at pointer (existing semantics); cancel when release is outside canvas and outside valid structure targets in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T039 [P] [US4] Extend insert-into-parent / nested-subpart coverage if needed in `Tests/UserBoxLibraryTests.cpp`

**Checkpoint**: US4 complete — inspect is read-only; structure and canvas are dual insert surfaces.

---

## Phase 7: User Story 5 — Group box navigation and stable group chrome (Priority: P3)

**Goal**: Canvas double-click opens groups reliably; group box size stays stable during drag and Parameters edits/randomize.

**Independent Test**: `quickstart.md` Scenario F (+ Scenario C step 6) — open via canvas double-click; ≥20 drag cycles without size change.

### Implementation for User Story 5

- [X] T040 [US5] Restore reliable canvas double-click → `setCanvasFocus(group)` after chrome removal; verify context-menu Open still works in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T041 [US5] Stop content-measure feedback from removed widgets so `group.size` / `node.size` stay stable during drag per slim-box contract in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T042 [US5] Guard against Parameters/randomize edits causing on-canvas group resize in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T043 [US5] Manually validate nested-group open/drag stability against Scenario F in `specs/013-box-property-panel-ux/quickstart.md`

**Checkpoint**: US5 complete — group open works; no spontaneous size glitches.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and cleanup across stories.

- [X] T044 Run full manual matrix in `specs/013-box-property-panel-ux/quickstart.md` Scenarios A–F and note gaps
- [X] T045 [P] Mark completed TODO items for this feature in `TODO.md` (lines covering property panel / structure / library drag / box drag / group glitch)
- [X] T046 [P] Doxygen for new public APIs (`disconnectAllLinksForBox`, `reparentBoxLikeInsert`, SelectionContext, Parameters force-tab) in `OpenYourBox/Source/graph/NodeGraph.h`, `OpenYourBox/Source/graph/GraphTypes.h` / `NodeRenderer.h`, and `OpenYourBox/Source/PluginEditor.h`
- [X] T047 Code cleanup: remove dead on-box UI helpers left unused after relocation in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T048 Confirm Info / Capture / Train / Presets tabs and freeze flows unchanged in `OpenYourBox/Source/PluginEditor.cpp`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: Depends on Foundational — MVP
- **US2 (Phase 4)**: Depends on Foundational; practically benefits from US1 slim chrome (T016–T017) — prefer after US1
- **US3 (Phase 5)**: Depends on Foundational (SelectionContext + reparent APIs); uses Parameters force from US1 — prefer after US1
- **US4 (Phase 6)**: Depends on US3 structure drop targets + US1 Parameters modes
- **US5 (Phase 7)**: Depends on US1 slim group chrome; can overlap late US2/US3 polish
- **Polish (Phase 8)**: After desired stories complete

### User Story Dependencies

- **US1 (P1)**: After Foundational — no other story dependency — **MVP**
- **US2 (P1)**: After Foundational; strongest after US1 chrome strip
- **US3 (P2)**: After Foundational + US1 Parameters force-tab
- **US4 (P2)**: After US3 drop-target infrastructure
- **US5 (P3)**: After US1 group slim-down

### Parallel Opportunities

- T002–T004 (inventory) in parallel during Setup
- T012–T013 (tests) in parallel after T009–T011
- T033–T034 (library inspect) can overlap late US3 once Parameters read-only path exists
- T045–T046 in Polish in parallel

---

## Parallel Example: Foundational

```bash
# After T009–T011 land, launch graph tests together:
Task: "Add disconnect/reparent/cycle coverage in Tests/GraphGroupTests.cpp"
Task: "Add undo coverage for reparent in Tests/EditHistoryTests.cpp"
```

## Parallel Example: User Story 1

```bash
# After T014–T015 (panel + force tab), strip chrome in sequence on the same file:
Task: "Strip renderNode inline params"
Task: "Strip renderGroup inline params/randomize"
# Randomize + group name/repeats to panel can proceed once helpers exist (T008)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 Setup
2. Complete Phase 2 Foundational
3. Complete Phase 3 US1 (Parameters + slim boxes)
4. **STOP and VALIDATE** via `quickstart.md` Scenario A
5. Demo denser graphs with panel editing

### Incremental Delivery

1. Setup + Foundational → APIs + tab shell ready
2. US1 → MVP Parameters + slim boxes
3. US2 → one-click drag polish
4. US3 → structure navigate + reparent
5. US4 → library inspect + dual drop surfaces
6. US5 → group open + size stability
7. Polish → full quickstart matrix

### Suggested MVP Scope

**US1 only** (T014–T021 after Foundational): delivers the core UX win (parameters off boxes, Parameters tab). US2 often falls out of US1 chrome removal with light drag fixes.

---

## Notes

- [P] = different files / no incomplete-task dependency
- [USn] maps to spec user stories 1–5
- Prefer sequential work on `NodeRenderer.cpp` within a story to avoid merge thrash
- Commit after each task or logical group
- Stop at checkpoints to validate independently
- Constitution: GUI-thread only for panel/DnD/reparent; no audio-thread work
