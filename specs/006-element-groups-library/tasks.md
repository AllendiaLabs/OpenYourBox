---
description: "Task list for Element Groups & User Box Library"
---

# Tasks: Element Groups & User Box Library

**Input**: Design documents from `specs/006-element-groups-library/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Spec does not require TDD. Validate via `quickstart.md`; optional C++ round-trip helpers noted in Polish.

**Organization**: Tasks grouped by user story for independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`US1`–`US5`) on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- Graph: `OpenYourBox/Source/graph/`
- Library: `OpenYourBox/Source/library/`
- UI: `OpenYourBox/Source/ui/`
- Freeze: `OpenYourBox/Source/freeze/`
- DSP publish: `OpenYourBox/Source/dsp/`
- Tests: `Tests/`
- Specs: `specs/006-element-groups-library/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design anchors and CMake wiring before group/library code.

- [X] T001 Verify design artifact cross-links (`plan.md`, `data-model.md`, contracts) and path conventions in `specs/006-element-groups-library/plan.md`
- [X] T002 [P] Add `UserBoxLibrary` / `UserBoxLibraryPanel` source entries to the plug-in target in `CMakeLists.txt`
- [X] T003 [P] Confirm imgui-node-editor `Group` / `BeginGroupHint` APIs available from the FetchContent pin used by `OpenYourBox` (document pin in `specs/006-element-groups-library/research.md` if version note needed)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared group types, membership APIs, serialization hooks, user-data path, and freeze selection expansion stubs required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Add `GraphGroup` (id, name, parentGroupId, memberIds, collapsed, copies N, position/size) and `parentGroupId` on nodes in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T005 [P] Add Audio I/O exclusion helper for group/library eligibility (reuse/extend fixed I/O checks) in `OpenYourBox/Source/graph/GraphTypes.h` or `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T006 Implement group container storage + ID allocation in `OpenYourBox/Source/graph/NodeGraph.h` / `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T007 Persist/restore `Groups` collection and node `parentGroupId` in `GraphDocument` ValueTree in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T008 [P] Add `boxesDirectory()` beside samples/weights helpers in `OpenYourBox/Source/library/UserDataPaths.h`
- [X] T009 Scaffold empty `UserBoxLibrary` class (load/save index stubs) in `OpenYourBox/Source/library/UserBoxLibrary.h` / `OpenYourBox/Source/library/UserBoxLibrary.cpp`
- [X] T010 [P] Scaffold empty `UserBoxLibraryPanel` ImGui shell in `OpenYourBox/Source/ui/UserBoxLibraryPanel.h` / `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp`
- [X] T011 Add selection→leaf freezable member expansion helper (groups included) stub in `OpenYourBox/Source/graph/NodeGraph.cpp` per `specs/006-element-groups-library/contracts/freeze-per-member-contract.md`
- [X] T012 Confirm `LiveGraphPublisher` ignores collapse and always publishes full membership topology in `OpenYourBox/Source/dsp/LiveGraphPublisher.cpp`

**Checkpoint**: Types, persistence hooks, paths, and skeletons ready — story work can begin.

---

## Phase 3: User Story 1 - Group Elements and Nested Subgroups (Priority: P1) 🎯 MVP

**Goal**: Create named groups and nested subgroups; add/remove/rename/ungroup; refuse Audio I/O and illegal nests; persist hierarchy; freeze freezable members individually without replacing the group with one Gold box; expanded chrome via imgui-node-editor `Group`.

**Independent Test**: Quickstart §1–2 and §4: group ≥3 nodes, nest subgroup, save/reload; refuse Audio I/O; freeze group → N Gold members, group remains.

### Implementation for User Story 1

- [X] T013 [US1] Implement `createGroup` from selection (≥2 allowed members, default name, set `parentGroupId`) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T014 [US1] Implement nested subgroup create, ungroup (lift members), rename, and membership add/remove with cycle checks in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T015 [US1] Refuse Audio I/O in group create/move with user-visible reason in `OpenYourBox/Source/graph/NodeGraph.cpp` and surface message from `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T016 [US1] Enforce nesting depth (≥5 supported; refuse over hard cap if any) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T017 [US1] Draw expanded groups with `ed::Group` / group hints and keep members visually framed in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T018 [US1] Add context-menu actions Group / Ungroup / Rename Group in `OpenYourBox/Source/graph/NodeRenderer.cpp` per `specs/006-element-groups-library/contracts/group-editor-ui-contract.md`
- [X] T019 [US1] Support dragging members into/out of groups with illegal-move refusal in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T020 [US1] Wire per-member freeze: expand selection to freezable leaves; enqueue one freeze outcome per member; keep group container in `OpenYourBox/Source/PluginEditor.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp` (coordinate with `OpenYourBox/Source/freeze/FreezeCoordinator.cpp`)
- [X] T021 [US1] Skip already-Gold / non-freezable members with summary feedback on freeze in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T022 [US1] Verify project save/load restores group hierarchy via existing `persistGraph` path in `OpenYourBox/Source/PluginEditor.cpp` / `OpenYourBox/Source/PluginProcessor.cpp`

**Checkpoint**: Nested groups work, persist, and freeze per member; Audio I/O cannot be grouped.

---

## Phase 4: User Story 5 - Stack Group Copies (N Blocks) (Priority: P1)

**Goal**: Per-group copies parameter N; materialize N independent serial copies when I/O shapes allow; refuse illegal N; clone-on-grow; persist N + copies; include in library snapshots.

**Independent Test**: Quickstart §5: N 1→3 on chainable group; independent weights; illegal N refused; save/reload.

### Implementation for User Story 5

- [X] T023 [US5] Persist `copies` and copy-instance mapping in ValueTree with groups in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T024 [US5] Implement shape-legal serial-chain validation for a group’s external I/O in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T025 [US5] Implement `setGroupCopies(N)` materialize/remove independent copies (clone last on increase) in `OpenYourBox/Source/graph/NodeGraph.cpp` per `specs/006-element-groups-library/contracts/group-copies-contract.md`
- [X] T026 [US5] Re-assert serial wiring between copies and external first/last attach on successful N change in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T027 [US5] Refuse/clamp illegal N with user-visible message and no orphan nodes in `OpenYourBox/Source/graph/NodeGraph.cpp` / `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T028 [US5] Expose copies N ImGui property on group (expanded/collapsed) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T029 [US5] Ensure freeze selection expansion includes freezable members of all copies in `OpenYourBox/Source/graph/NodeGraph.cpp` / `OpenYourBox/Source/PluginEditor.cpp`
- [X] T030 [US5] Include N and all materialized copies in box-library snapshot builder in `OpenYourBox/Source/library/UserBoxLibrary.cpp`

**Checkpoint**: N-blocks stacking works for chainable groups; illegal N safe.

---

## Phase 5: User Story 2 - Expand and Collapse Groups in the UI (Priority: P1)

**Goal**: Collapse/expand groups independently (including nested); collapsed compact box with derived external ports; presentation-only (audio unchanged); persist collapse state.

**Independent Test**: Quickstart §3: collapse/expand outer and inner; cables work; audio unchanged; collapse state survives reload.

### Implementation for User Story 2

- [X] T031 [US2] Implement `setGroupCollapsed` / toggle APIs and persist `collapsed` in ValueTree in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T032 [US2] When collapsed, skip drawing interior members/links and draw compact group node in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T033 [US2] Derive and render external ports from boundary-crossing links on collapsed groups in `OpenYourBox/Source/graph/NodeRenderer.cpp` / `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T034 [US2] Map new links on collapsed group pins to underlying member pins with existing shape validation in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T035 [US2] Add Collapse / Expand context-menu (and optional header click) with independent nested states in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T036 [US2] Ensure freeze on collapsed group still expands to members (US1 helper) without requiring expand first in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T037 [US2] Confirm DSP path unchanged when toggling collapse (no republish topology change beyond layout) in `OpenYourBox/Source/dsp/LiveGraphPublisher.cpp`

**Checkpoint**: Collapse/expand usable and persisted; audio unaffected.

---

## Phase 6: User Story 3 - Save Boxes to a User Library (Priority: P1)

**Goal**: Per-box save (element or group, Gold or live) with parameters + artifacts; no multi-select save; no Audio I/O; overwrite/rename/delete with confirm; persist under user data.

**Independent Test**: Quickstart §6–7 (save half): save element + group; refuse multi-select and Audio I/O; rename/overwrite/delete; library survives relaunch.

### Implementation for User Story 3

- [X] T038 [US3] Implement box snapshot builder (single node or group tree + internal links + properties/seeds + artifact refs + copies N) in `OpenYourBox/Source/library/UserBoxLibrary.cpp`
- [X] T039 [US3] Copy weight / BlackBox artifacts into entry `artifacts/` on save in `OpenYourBox/Source/library/UserBoxLibrary.cpp`
- [X] T040 [US3] Implement `index.json` CRUD: save (name prompt path), overwrite confirm, rename, delete confirm in `OpenYourBox/Source/library/UserBoxLibrary.cpp`
- [X] T041 [US3] Add **Save to Box Library** context action for exactly one node or one group target in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T042 [US3] Refuse multi-select save and Audio I/O save with clear messages in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T043 [US3] Wire `UserBoxLibrary` instance lifecycle (load on editor init) in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T044 [US3] Show minimal ImGui name/overwrite/delete confirms for save flows in `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp` or inline dialogs in `OpenYourBox/Source/graph/NodeRenderer.cpp`

**Checkpoint**: Boxes can be saved and managed on disk; placement UI can come in US4.

---

## Phase 7: User Story 4 - Browse and Place From the User Library (Priority: P2)

**Goal**: Browse Box Library (name + kind); place/insert clones with new IDs; groups insert fully collapsed; empty state; distinct from Training Library.

**Independent Test**: Quickstart §6 place half: list `E1`/`G1`, place both; group collapsed; library originals unchanged.

### Implementation for User Story 4

- [X] T045 [US4] Render Box Library list (name, Element|Group kind, empty state) in `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp` distinct from Training Library tab in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T046 [US4] Implement insert/clone from snapshot (new IDs, rewrite internal links, restore artifacts, restore N/copies) in `OpenYourBox/Source/library/UserBoxLibrary.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T047 [US4] Force root + nested groups `collapsed=true` on insert in `OpenYourBox/Source/library/UserBoxLibrary.cpp` / `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T048 [US4] Add DnD payload `OPENYOURBOX_BOX_LIBRARY_ID` (or Place button) into canvas in `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T049 [US4] Fail insert safely on unknown element type / missing artifact with toast; leave graph unchanged in `OpenYourBox/Source/library/UserBoxLibrary.cpp` and `OpenYourBox/Source/PluginEditor.cpp`
- [X] T050 [US4] Label UI **Boxes** / Box Library so it cannot be confused with Training **Library** in `OpenYourBox/Source/PluginEditor.cpp`

**Checkpoint**: Full save → browse → place loop works; inserted groups start collapsed.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and small hardening across stories.

- [X] T051 Run full manual scenarios in `specs/006-element-groups-library/quickstart.md` and note gaps
- [X] T052 [P] Add C++ ValueTree round-trip coverage for groups + copies in `Tests/` (extend an existing graph test target if present)
- [X] T053 [P] Add library index save/load smoke test helpers in `Tests/` or a focused unit file under `OpenYourBox` test suite
- [X] T054 Review UX copy for Group / Copies N / Collapse / Save to Box Library / Freeze M elements in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T055 Confirm Train absorb single-BlackBox path unchanged and Freeze menu copy does not imply selection→one box in `OpenYourBox/Source/PluginEditor.cpp`
- [ ] T056 [P] Update `specs/006-element-groups-library/plan.md` Structure Decision notes if final file names differ

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Start immediately
- **Foundational (Phase 2)**: Depends on Setup — **blocks all user stories**
- **US1 (Phase 3)**: After Foundational — MVP groups
- **US5 (Phase 4)**: After US1 — needs group model before copies
- **US2 (Phase 5)**: After Foundational; practically after US1 group drawing exists
- **US3 (Phase 6)**: After Foundational; group save with N needs US5 for full fidelity (element-only save can start earlier)
- **US4 (Phase 7)**: Depends on US3 catalog/payload format
- **Polish (Phase 8)**: After desired stories complete

### User Story Dependencies

- **US1**: No story dependency — MVP
- **US5**: Depends on US1 group entities
- **US2**: Uses US1 group entities/UI; independently testable once groups exist
- **US3**: Can save single elements with foundation; group+N save needs US1/US5
- **US4**: Requires US3 persistence format

### Parallel Opportunities

- T002–T003 after T001
- T005, T008, T010 in parallel during Foundational once T004 started/settled
- After US1: US5 and US2 can proceed in parallel if staffed (different concerns; watch `NodeRenderer` conflicts)
- US3 library core vs menus after API shapes agreed
- T052–T053 and T056 in Polish

---

## Parallel Example: User Story 1

```bash
# After Foundational:
Task: "T013–T016 NodeGraph group operations in OpenYourBox/Source/graph/NodeGraph.cpp"
Task: "T017–T019 NodeRenderer group chrome/menus in OpenYourBox/Source/graph/NodeRenderer.cpp"
Task: "T020–T021 per-member freeze in PluginEditor.cpp / NodeGraph.cpp"
```

---

## Parallel Example: User Story 5

```bash
Task: "T024–T027 NodeGraph setCopies + validation in OpenYourBox/Source/graph/NodeGraph.cpp"
Task: "T028 NodeRenderer copies property in OpenYourBox/Source/graph/NodeRenderer.cpp"
```

---

## Parallel Example: User Story 3 + 4 prep

```bash
Task: "T038–T040 UserBoxLibrary persistence in OpenYourBox/Source/library/UserBoxLibrary.cpp"
Task: "T041–T042 Save menu gating in OpenYourBox/Source/graph/NodeRenderer.cpp"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 Setup  
2. Phase 2 Foundational  
3. Phase 3 US1 (groups + nest + persist + per-member freeze)  
4. **STOP** — validate quickstart §1–2, §4  

### Incremental Delivery

1. US1 → nested groups + freeze  
2. US5 → copies N stacking  
3. US2 → collapse/expand  
4. US3 → save to box library  
5. US4 → browse/place  
6. Polish → quickstart full pass  

### Suggested MVP Scope

**US1** first, then **US5** for N-blocks: durable groups + independent serial copies.

---

## Notes

- Prefer imgui-node-editor `Group` for expanded frames; collapse is app-level (research Decision 2)
- Copies N = independent materialized serial stack (research Decision 8)
- Box Library ≠ Training Library (samples)
- Freeze ≠ Train absorb
- Commit after each task or logical group
- Every task above uses checklist format: `- [ ] Txxx ...` with file paths

## Phase 9: Group editor UX (TODO 30–37)

Product follow-ups applied on top of US1/US2/US5.

- [X] T057 [US2] Double-click a group to expand/collapse in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T058 [US1] Add members only when dragging an element (or palette drop) onto a group; highlight the drop target; do not add when dragging a group onto an element in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T059 [US1] Move a group together with all members, including clipped members, in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T060 [US5] Keep copies invisible in the UI; unroll serial copies at compile/train time; store independent `copySlots` on members in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/dsp/LiveGraphPublisher.cpp`
- [X] T061 [US1] Resizable group frames via imgui-node-editor `Group` (resize cursor) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T062 [US1] Context-menu **Remove from Group** in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T063 [US1] Drag a member outside group bounds to remove it in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T064 [US1] Fit group to members on create; allow clip when resized down; draw a map on each expanded group and subgroup in `OpenYourBox/Source/graph/NodeGraph.cpp` / `OpenYourBox/Source/graph/NodeRenderer.cpp`

