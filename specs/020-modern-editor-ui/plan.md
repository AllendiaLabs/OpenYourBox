# Implementation Plan: Modern Editor UI

**Branch**: `020-modern-editor-ui` | **Date**: 2026-09-05 | **Spec**: `specs/020-modern-editor-ui/spec.md`

**Input**: Feature specification from `specs/020-modern-editor-ui/spec.md`

**Status**: Design complete. Ready for `/speckit-tasks`.

Next: `/speckit-tasks` (then `/speckit-implement`).

Restyle OpenYourBox as a **dark modern graph instrument editor**. Keep today’s graph-first layout (library overlay on the canvas, right inspector, top session chrome). Replace `ImGui::StyleColorsDark()` and stock widgets with a tokenized palette, bundled Inter (regular + semibold), Phosphor-class outline icons, and custom instrument chrome for tabs, trees, fields, Dry/Wet, knobs, and XY. Remap Live / Frozen / family hues into one dark family (Live cooler, Frozen warmer, accent not equal to Live). VS Code is materials inspiration only — not a layout to copy. Build the existing JUCE **Standalone** format as a visual-QA host and capture window-only stills under `.ignore/visual-refs/` during implementation.

## Technical Context

**Language/Version**: C++17 (JUCE plug-in editor / ImGui host)

**Primary Dependencies**: JUCE 8, Dear ImGui, imgui-node-editor, OpenGL via existing `ImGuiHost`; bundled Inter (SIL OFL) + one MIT/ISC outline icon font; no new runtime services

**Storage**: No new user persistence. Visual tokens live in code. Private stills and the public reference board live under gitignored `.ignore/visual-refs/`. Font and icon files ship via `juce_add_binary_data`. Attribution in `NOTICE`

**Testing**: CTest unit tests for token uniqueness / arm’s-length distinctness (Live vs Frozen vs accent vs illegal red) and style application without a GPU window; visual QA via Standalone stills in `quickstart.md`

**Target Platform**: Desktop AU/VST3, macOS first; same editor in the existing Standalone target used only for screenshots

**Project Type**: Single desktop audio plug-in (visual restyle of the existing editor)

**Performance Goals**: Constitution 60 FPS UI while audio runs; custom draw-list widgets MUST stay cheap (no per-frame texture atlases beyond the ImGui font atlas); zero audio-thread allocations

**Constraints**: VST remains the product UI; Standalone is QA-only (already in `CMakeLists.txt`); dark mode only; no activity bar / command palette / file tabs; graph keeps majority of typical plugin window; Parameters-on-select unchanged; illegal cables stay unambiguously red; `NodeState::liveBlue` / `frozenGold` enum names stay (RGB remaps)

**Scale/Scope**: One editor chrome (all current panels + canvas boxes + knobs/XY + modals); ~10 named colour tokens; two font weights; one icon set; screenshot set of idle, mixed Live/Frozen, each inspector tab, analysis, and one modal — at plugin size and large window

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Single Interface, Decoupled Compute`: **PASS (justified Standalone)**. Product UI remains the VST. The existing JUCE `Standalone` format is used only as a windowed visual-QA host (spec FR-012/FR-015). It is not a second customer application and does not replace Train/Freeze in the plug-in. See Complexity Tracking.
- `Dual-Engine Execution Model`: **PASS**. Live vs Frozen remain roles; lock affordance stays. Hues remap into the modern family (Live cooler, Frozen warmer) without changing engine behaviour.
- `Manual Granular Freeze Policy`: **PASS**. Freeze/unfreeze and compile status remain visible in restyled top chrome (not a cloned IDE status bar).
- `Shape Integrity & Legal Constraints`: **PASS**. Illegal cables stay warning-red. Copyright / error modals restyle materials only.
- `Zero Audio Allocations / Non-Blocking Audio Thread`: **PASS**. All restyle work is GUI/OpenGL. No audio-thread draws or allocations.
- `UI Responsiveness 60 FPS`: **PASS**. Tokens + custom widgets on the existing ImGui frame; no extra windows or retained scene graph.
- `Local vs cloud access`: **PASS**. Unchanged.

**Post-Design Re-Check**: **PASS**. Contracts keep layout graph-first, engines untouched, and Standalone QA-only. No unjustified constitution violations.

## Project Structure

### Documentation (this feature)

```text
specs/020-modern-editor-ui/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── visual-language-tokens-contract.md
│   ├── editor-chrome-layout-contract.md
│   ├── instrument-controls-contract.md
│   └── visual-qa-screenshots-contract.md
├── checklists/
│   └── requirements.md
└── tasks.md             # /speckit-tasks (NOT this command)
```

### Source Code (repository root)

```text
OpenYourBox/Source/ui/
├── VisualLanguage.h / .cpp     # tokens, applyStyle(), font/icon load
├── InstrumentWidgets.h / .cpp  # tabs, tree rows, buttons, fields, Dry/Wet, knob, XY
├── ImGuiHost.cpp               # load fonts; apply VisualLanguage instead of StyleColorsDark
├── InfoPanel.cpp / TrainPanel.cpp / CaptureSamplesPanel.cpp / …
└── CopyrightModal.cpp / ErrorModal.cpp
OpenYourBox/Source/graph/
├── GraphTypes.h                # remap chromeColourForType / family RGB (keep enum names)
└── NodeRenderer.cpp            # box chrome, library overlay, imgui-node-editor style
OpenYourBox/Source/PluginEditor.cpp  # top chrome + right inspector using instrument widgets
OpenYourBox/Resources/fonts/    # Inter Regular + SemiBold; icon font; OFL/MIT texts
CMakeLists.txt                  # juce_add_binary_data fonts; tests
NOTICE                          # Inter OFL + icon-set attribution
Tests/VisualLanguageTests.cpp   # token distinctness
.ignore/visual-refs/            # gitignored stills + reference board (not committed)
```

**Structure Decision**: Extend the existing OpenYourBox ImGui editor. Do not add a second UI toolkit, a customer Standalone app, or a VS Code-like activity-bar shell.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Constitution “no standalone application” vs building the JUCE Standalone format | Spec requires windowed screenshots without a DAW (FR-012). The Standalone *format* already exists in `CMakeLists.txt` and is the practical QA host. | DAW-only screenshots are unreliable (host chrome, scaling). A new dedicated screenshot tool would be a second executable and a worse constitution fit. |

## Phase 0: Research — Complete

Resolved in `research.md`: layout stays graph-first; Inter + Phosphor (or Lucide TTF subset); OYB-owned dark tokens; custom ImGui draw-list widgets; Standalone QA + window-only stills; NOTICE.

## Phase 1: Design — Complete

Delivered: `data-model.md`, four contracts, `quickstart.md`.

## Execution Notes

Suggested implementation order (for `/speckit-tasks`):

1. Capture a **before** Standalone still into `.ignore/visual-refs/`
2. Tokens + `applyStyle()`; remap `chromeColourForType`; unit tests
3. Bundle Inter + icon font; load in `ImGuiHost`; NOTICE
4. Instrument widgets; restyle inspector tabs, library overlay, top chrome, modals
5. Knob / XY / Dry/Wet / analysis wells
6. imgui-node-editor + box chrome
7. Screenshot pass at plugin size and large window after each visual slice

**Artifact map**

| Artifact | Path |
|----------|------|
| Research | `specs/020-modern-editor-ui/research.md` |
| Data model | `specs/020-modern-editor-ui/data-model.md` |
| Quickstart | `specs/020-modern-editor-ui/quickstart.md` |
| Contracts | `specs/020-modern-editor-ui/contracts/` |
