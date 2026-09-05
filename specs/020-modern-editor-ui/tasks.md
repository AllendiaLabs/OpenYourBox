---
description: "Task list for Modern Editor UI"
---

# Tasks: Modern Editor UI

**Input**: Design documents from `specs/020-modern-editor-ui/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Plan requires CTest token distinctness in `Tests/VisualLanguageTests.cpp`. Beauty/QA is Standalone stills per `quickstart.md` and `contracts/visual-qa-screenshots-contract.md` (not pixel-diff CI).

**Organization**: Tasks grouped by user story for independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`US1`–`US5`) on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Chrome/tokens: `OpenYourBox/Source/ui/VisualLanguage.*`, `InstrumentWidgets.*`, `ImGuiHost.cpp`
- Canvas: `OpenYourBox/Source/graph/GraphTypes.h`, `NodeRenderer.cpp`
- Editor shell: `OpenYourBox/Source/PluginEditor.cpp`
- Panels/modals: `OpenYourBox/Source/ui/*Panel.cpp`, `CopyrightModal.cpp`, `ErrorModal.cpp`
- Fonts: `OpenYourBox/Resources/fonts/`, `CMakeLists.txt`, `NOTICE`
- Tests: `Tests/VisualLanguageTests.cpp`
- Stills: `.ignore/visual-refs/` (gitignored)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Directories, resource hooks, and a before-still so restyle has a baseline. No product look change yet except capturing the current UI.

- [X] T001 Create `OpenYourBox/Resources/fonts/` (license text placeholders) and `.ignore/visual-refs/` per `specs/020-modern-editor-ui/contracts/visual-qa-screenshots-contract.md`
- [X] T002 [P] Note Standalone + `ctest -R VisualLanguage` build commands in `specs/020-modern-editor-ui/quickstart.md` Prerequisites if missing
- [X] T003 Build and launch OpenYourBox Standalone; capture window-only **before** stills at plugin size and large window into `.ignore/visual-refs/before-plugin.png` and `.ignore/visual-refs/before-large.png`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Tokens, fonts, icons, tests, and `applyStyle()` replacing `StyleColorsDark`. Blocks all user stories.

**⚠️ CRITICAL**: No user story work begins until this phase is complete

- [X] T004 Add colour/type/icon token table (`surface.*`, `text.*`, `accent`, `live`, `frozen`, families, `danger`) in `OpenYourBox/Source/ui/VisualLanguage.h` per `specs/020-modern-editor-ui/data-model.md` and `contracts/visual-language-tokens-contract.md`
- [X] T005 Implement `VisualLanguage::applyStyle()` (rounding, padding, `ImGuiCol_*` mapped to tokens; no default toolkit blue fills) in `OpenYourBox/Source/ui/VisualLanguage.cpp`
- [X] T006 Remap `liveBlueColour`, `frozenGoldColour`, family colours, and `chromeColourForType` to tokens in `OpenYourBox/Source/graph/GraphTypes.h` (keep `NodeState::liveBlue` / `frozenGold` names)
- [X] T007 [P] Add pairwise distinctness tests (Live ≠ Frozen ≠ accent ≠ danger; dark surfaces) in `Tests/VisualLanguageTests.cpp` and register `OpenYourBoxVisualLanguageTests` in `CMakeLists.txt`
- [X] T008 Bundle Inter Regular + SemiBold (SIL OFL) plus `OFL.txt` under `OpenYourBox/Resources/fonts/` and add them to `juce_add_binary_data` in `CMakeLists.txt`
- [X] T009 [P] Bundle Phosphor Regular (or Lucide TTF subset) plus MIT/ISC license text under `OpenYourBox/Resources/fonts/` and add to `juce_add_binary_data` in `CMakeLists.txt`
- [X] T010 Load Inter (Regular + SemiBold) and merge the icon font; call `VisualLanguage::applyStyle()` instead of `ImGui::StyleColorsDark()` in `OpenYourBox/Source/ui/ImGuiHost.cpp`
- [X] T011 [P] Add Inter OFL and icon-font MIT/ISC attribution blocks in `NOTICE`
- [X] T012 Add `VisualLanguage.cpp` (and later `InstrumentWidgets.cpp`) to `target_sources(OpenYourBox …)` in `CMakeLists.txt`

**Checkpoint**: Standalone boots dark with Inter; CTest token tests pass; `StyleColorsDark` is not the effective theme

---

## Phase 3: User Story 1 - First glance is OpenYourBox, not a toolkit (Priority: P1) 🎯 MVP

**Goal**: Idle editor at plugin size reads as a dark modern graph instrument: graph dominant, professional type, token chrome — not default ImGui and not a VS Code clone.

**Independent Test**: Launch Standalone at ~900×600; capture idle; side-by-side with `.ignore/visual-refs/before-plugin.png` — reviewers pick before as “generic toolkit.” Graph is the majority of the window.

### Implementation for User Story 1

- [X] T013 [US1] Apply token page/canvas/panel colours to the host clear and main window in `OpenYourBox/Source/ui/ImGuiHost.cpp` and `OpenYourBox/Source/PluginEditor.cpp`
- [X] T014 [US1] Restyle top session chrome (RF/params, freeze/train status, error/warning) with tokens and SemiBold/Regular ramp in `OpenYourBox/Source/PluginEditor.cpp` (no IDE status-bar clone)
- [X] T015 [US1] Restyle right inspector frame and `SideTabs` using token surfaces/borders in `OpenYourBox/Source/PluginEditor.cpp` per `contracts/editor-chrome-layout-contract.md`
- [X] T016 [US1] Apply imgui-node-editor `Style` (bg, grid, node, pin, select) from tokens in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T017 [US1] Confirm library overlay stays on the graph (no new activity bar or permanent left dock) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T018 [US1] Build Standalone; capture window-only idle stills `.ignore/visual-refs/idle-plugin.png` and `.ignore/visual-refs/idle-large.png`

**Checkpoint**: Idle view is dark, graph-first, Inter-set, distinguishable from the before-still without copying VS Code layout

---

## Phase 4: User Story 2 - Public visual references stay in the loop (Priority: P1)

**Goal**: A private reference board exists (VS Code materials + at least ImHex and Tracy) and implementers compare OpenYourBox stills to it — without cloning those layouts.

**Independent Test**: `.ignore/visual-refs/` contains public inspiration stills/URLs plus OpenYourBox idle stills; a reviewer can name shared *materials* but not a copied activity bar/terminal.

### Implementation for User Story 2

- [X] T019 [P] [US2] Save public inspiration stills (VS Code dark workbench materials, ImHex chrome, Tracy typography) plus a short `urls.txt` in `.ignore/visual-refs/` per `specs/020-modern-editor-ui/spec.md` US2
- [X] T020 [US2] After T018, compare `idle-plugin.png` to the board and record “materials vs layout-clone” notes in `.ignore/visual-refs/review-notes.md` (do not commit)

**Checkpoint**: Reference board exists; idle still is judged against it as inspiration, not a template

---

## Phase 5: User Story 3 - Existing workflows remain discoverable (Priority: P1)

**Goal**: Place/wire/Parameters/freeze/train/capture/presets still work in the same places (graph-first). Live vs Frozen remain obvious; illegal cables stay red; modals keep blocking behaviour.

**Independent Test**: First-session path (place box, wire In→process→Out, open Parameters, Dry/Wet) and freeze/unfreeze without a tutorial; all former inspector tabs reachable.

### Implementation for User Story 3

- [X] T021 [US3] Keep Parameters-on-select (`consumeForceParametersTab` / `pendingSidePanelTab`) and the Info/Parameters/Library/Capture/Train/Presets tab set in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T022 [P] [US3] Restyle copyright modal materials only (same blocking/ack) in `OpenYourBox/Source/ui/CopyrightModal.cpp`
- [X] T023 [P] [US3] Restyle error modal materials only in `OpenYourBox/Source/ui/ErrorModal.cpp`
- [X] T024 [US3] Ensure illegal-link colour uses `danger` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T025 [US3] Verify Live vs Frozen + lock remain arm’s-length distinct on canvas via `chromeColourForType` in `OpenYourBox/Source/graph/GraphTypes.h` / `NodeRenderer.cpp`
- [X] T026 [US3] Capture `.ignore/visual-refs/mixed-live-frozen.png` from Standalone

**Checkpoint**: Workflows unchanged; mixed Live/Frozen still captured; modals restyled

---

## Phase 6: User Story 4 - Every control is an instrument part (Priority: P2)

**Goal**: Tabs, trees, buttons, fields, Dry/Wet, knobs, XY, and boxes share custom instrument chrome — not stock ImGui widgets.

**Independent Test**: Capture Library tree, inspector tabs, Parameters/Dry/Wet, Knob Input, XY, populated canvas; all share instrument language.

### Implementation for User Story 4

- [X] T027 [US4] Add instrument tab, tree-row, button, field, and checkbox helpers in `OpenYourBox/Source/ui/InstrumentWidgets.h` and `OpenYourBox/Source/ui/InstrumentWidgets.cpp` per `contracts/instrument-controls-contract.md`
- [X] T028 [US4] Add instrument Dry/Wet (thin track, round thumb) in `OpenYourBox/Source/ui/InstrumentWidgets.cpp` and use it from `OpenYourBox/Source/PluginEditor.cpp` / `OpenYourBox/Source/ui/InfoPanel.cpp`
- [X] T029 [US4] Add circular knob + XY well widgets in `OpenYourBox/Source/ui/InstrumentWidgets.cpp` and wire them from `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T030 [US4] Replace stock inspector `BeginTabBar` look with instrument tabs in `OpenYourBox/Source/PluginEditor.cpp`
- [X] T031 [US4] Restyle Factory/User Library/Project structure tree rows in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/ui/UserBoxLibraryPanel.cpp`
- [X] T032 [P] [US4] Restyle Train/Capture/Presets/Cloud controls to instrument buttons/fields in `OpenYourBox/Source/ui/TrainPanel.cpp`, `OpenYourBox/Source/ui/CaptureSamplesPanel.cpp`, `OpenYourBox/Source/ui/UserPresetPanel.cpp`, `OpenYourBox/Source/ui/CloudSettingsPanel.cpp`
- [X] T033 [P] [US4] Restyle analysis plot well on Info in `OpenYourBox/Source/ui/InfoPanel.cpp`
- [X] T034 [US4] Restyle slim graph boxes (name + pins) and cables to token fills in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T035 [US4] Capture instrument-control stills (tabs, tree, Dry/Wet, knob, XY) into `.ignore/visual-refs/` from Standalone

**Checkpoint**: No unstyled stock toolkit tabs/sliders as the visible default; knobs/XY are tactile instrument shapes

---

## Phase 7: User Story 5 - Windowed host screenshots at every visual step (Priority: P2)

**Goal**: Full window-only still set from Standalone at plugin size (and large where required). Stills are gitignored.

**Independent Test**: Folder contains before, idle, mixed Live/Frozen, each inspector tab, analysis, and one modal; all window-only.

### Implementation for User Story 5

- [X] T036 [US5] Capture each inspector tab (`tab-info` … `tab-presets`) window-only into `.ignore/visual-refs/` from Standalone per `contracts/visual-qa-screenshots-contract.md`
- [X] T037 [P] [US5] Capture analysis well still `.ignore/visual-refs/analysis.png` from Standalone
- [X] T038 [P] [US5] Capture one modal still `.ignore/visual-refs/modal.png` from Standalone
- [X] T039 [US5] Confirm stills are window-only (no desktop/Dock) and are **not** staged for git (`.ignore/` already gitignored)

**Checkpoint**: Screenshot contract required set is complete

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Quickstart, licenses, DAW parity, docs.

- [X] T040 Walk `specs/020-modern-editor-ui/quickstart.md` steps 3–9 on Standalone (and DAW plugin check in step 9)
- [X] T041 [P] Doxygen comments on public APIs in `OpenYourBox/Source/ui/VisualLanguage.h` and `OpenYourBox/Source/ui/InstrumentWidgets.h`
- [X] T042 [P] Confirm `NOTICE` lists Inter OFL and the icon font; license files exist under `OpenYourBox/Resources/fonts/`
- [X] T043 Run `ctest --output-on-failure -R VisualLanguage` and fix failures in `Tests/VisualLanguageTests.cpp`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Start immediately (T003 before-stills before restyle)
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **US1 (Phase 3)**: After Foundational — MVP
- **US2 (Phase 4)**: After US1 idle stills (needs something to compare)
- **US3 (Phase 5)**: After Foundational; typically after US1 so chrome exists
- **US4 (Phase 6)**: After US1 (widgets replace remaining stock controls)
- **US5 (Phase 7)**: After US3/US4 surfaces exist
- **Polish (Phase 8)**: After desired stories

### User Story Dependencies

- **US1 (P1)**: After Phase 2 — no other story
- **US2 (P1)**: After US1 idle still (T018)
- **US3 (P1)**: After Phase 2; best after US1
- **US4 (P2)**: After US1
- **US5 (P2)**: After US3/US4 for full still set; stills also happen inside earlier stories

### Parallel Opportunities

- T002 with T001; T007 with T006; T009 with T008; T011 with T010
- T022 and T023 (modals) in parallel
- T032 and T033 (panels) in parallel
- T037 and T038 (stills) in parallel
- T041 and T042 in polish

---

## Parallel Example: Foundational tokens

```bash
Task: "Remap chromeColourForType in OpenYourBox/Source/graph/GraphTypes.h"
Task: "Add VisualLanguageTests.cpp and CMakeLists.txt test target"
Task: "Bundle Phosphor under OpenYourBox/Resources/fonts/"
```

---

## Parallel Example: User Story 4 panels

```bash
Task: "Restyle Train/Capture/Presets/Cloud in OpenYourBox/Source/ui/TrainPanel.cpp (and siblings)"
Task: "Restyle analysis well in OpenYourBox/Source/ui/InfoPanel.cpp"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 Setup (including **before** stills)
2. Phase 2 Foundational (tokens, Inter, `applyStyle`, tests)
3. Phase 3 US1 (graph-first idle restyle + idle stills)
4. **STOP and VALIDATE**: before vs idle side-by-side

### Incremental Delivery

1. US2 reference board (while implementing)
2. US3 workflow preservation + mixed Live/Frozen still
3. US4 instrument widgets
4. US5 complete still set
5. Polish / quickstart / DAW

### Parallel Team Strategy

After Phase 2: one person on canvas/`NodeRenderer`, one on `PluginEditor` inspector/top chrome; US4 widgets after US1.

---

## Notes

- [P] = different files, no wait on an incomplete sibling
- Do not add an activity bar, command palette, or light/day theme
- Do not commit `.ignore/visual-refs/`
- Standalone is QA-only; VST remains the product UI
- Next: `/speckit-implement`
