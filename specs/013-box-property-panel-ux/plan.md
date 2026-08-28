# Implementation Plan: Box Property Panel UX

**Branch**: `013-box-property-panel-ux` | **Date**: 2026-08-28 | **Spec**: `specs/013-box-property-panel-ux/spec.md`

**Input**: Feature specification from `specs/013-box-property-panel-ux/spec.md`

## Summary

Move editable parameters and randomize off canvas boxes into a new right-menu **Parameters** tab driven by selection (canvas, Project structure, or read-only library inspect). Slim boxes to **name + pins** so one press-drag selects and moves without widget steal, and group boxes stop content-measuring size glitches. Extend Project structure with click→Parameters, double-click→center (groups stay closed), and DnD reparent that **disconnects cables then places like a new insert**. Allow **element list** and **user library** drops onto Project structure (as well as canvas), with post-drop select + Parameters + destination canvas focus.

## Technical Context

**Language/Version**: C++17 (VST / graph / UI)

**Primary Dependencies**: JUCE, Dear ImGui, imgui-node-editor, existing `NodeGraph` / `NodeRenderer` / `UserBoxLibrary` / `UserBoxLibraryPanel` / `PluginEditor` tab bar, `EditHistory`

**Storage**: No new persistence schema for parameters (same `GraphNode` / `GraphGroup` properties). Session-only: selection context kind (live vs library inspect), active Parameters tab force, Project structure drop-target highlight id. Hierarchy membership already in graph document / patch snapshots.

**Testing**: C++ unit tests under `Tests/` for disconnect-all-links, reparent-into-group/root, cycle reject; extend `GraphGroupTests` / `UserBoxLibraryTests` / `EditHistoryTests`. Manual UI scenarios in `quickstart.md` (ImGui interactions not unit-tested today).

**Target Platform**: macOS VST3/AU (primary); Windows/Linux secondary if build allows

**Project Type**: Desktop audio plugin with embedded ImGui node editor

**Performance Goals**: 60 FPS UI with slim nodes and structure DnD highlight; structure reparent / insert interactive (&lt; 100 ms typical); no audio-thread work for panel, DnD, or hierarchy edits

**Constraints**: VST-only UI; zero audio-thread allocations; Shape Integrity after reparent (no illegal leftover cables — cleared by design); reuse existing insert/placement and undo gesture patterns; extend 011 Project structure / library contracts rather than replace trees

**Scale/Scope**: Right-menu Parameters tab; slim element + group chrome; Project structure click/dblclick/DnD; library + palette → canvas or structure; graphs with nested groups depth ≥ 3 and ~100 boxes for navigate timing

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Single Interface, Decoupled Compute**: Pass. All UX lives in the VST; no new worker or standalone app.
- **Dual-Engine Execution Model**: Pass. Blue/Gold and randomize semantics unchanged; randomize only relocates to Parameters panel.
- **Manual Granular Freeze Policy**: Pass. Freeze/unfreeze and analysis flows untouched; Info/Capture/Train/Presets tabs remain.
- **Shape Integrity & Legal Constraints**: Pass. Reparent clears incident cables before move; new inserts use existing validation; copyright/Train untouched.
- **Zero Audio-Thread Allocation Rule**: Pass. Panel edits, DnD, disconnect, reparent, and canvas focus run on GUI thread; audio sees committed graph via existing publisher/history path.
- **Complexity Justification**: Pass. Parameters tab + structure DnD are required by clarified spec; disconnect-then-add-like-new matches user intent and avoids fragile cross-scope wiring.

**Post–Phase 1 re-check**: Still Pass. Contracts keep selection context and Parameters rendering on the GUI path; reparent API is a graph mutation with undo via existing patch gestures; no backend changes.

## Project Structure

### Documentation (this feature)

```text
specs/013-box-property-panel-ux/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── parameters-panel-contract.md
│   ├── slim-box-chrome-contract.md
│   ├── project-structure-navigation-contract.md
│   └── structure-hierarchy-dnd-contract.md
└── tasks.md                 # /speckit-tasks (not created here)
```

### Source Code (repository root)

```text
OpenYourBox/
├── Source/
│   ├── PluginEditor.cpp / .h          # Add Parameters tab; force tab on selection; wire selection context
│   ├── graph/
│   │   ├── NodeGraph.cpp / .h         # disconnectAllLinks(boxId); reparentBoxLikeInsert(boxId, targetParentOrRoot)
│   │   ├── NodeRenderer.cpp / .h      # Slim renderNode/renderGroup; Parameters panel render; structure click/dblclick/DnD; select+drag
│   │   └── GraphTypes.h               # Optional SelectionContext / InspectTarget enums if not kept local to renderer/editor
│   ├── library/
│   │   └── UserBoxLibrary.*           # Unchanged insert semantics; may expose snapshot property read for inspect
│   └── ui/
│       └── UserBoxLibraryPanel.*      # Click → inspect callback; keep catalog drag payloads
Tests/
├── GraphGroupTests.cpp                # Disconnect + reparent; cycle reject; undo
├── UserBoxLibraryTests.cpp            # Insert still OK; structure-target insert if exercised via graph API
└── EditHistoryTests.cpp               # Reparent / disconnect as undoable gesture
```

**Structure Decision**: Extend existing `OpenYourBox` tree only. Relocate property-row UI from `renderNode`/`renderGroup` into a shared Parameters renderer invoked from the new right tab. Reuse palette/library DnD payloads and `addToGroup` / `importBox` / `addNode` placement; add an explicit disconnect step before reparent.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Selection context (live vs library-read-only) | Spec requires inspect without catalog mutation | Always-editable panel would corrupt saved library entries |
| Disconnect-all before reparent | Spec: add like new item; cross-scope cables become illegal/ambiguous | Keeping cables via `addToGroup` alone violates clarified UX and Shape Integrity across canvases |
| Dual drop targets (canvas + Project structure) | Spec requires hierarchy placement without canvas-only detour | Canvas-only insert cannot target a non-focused group cleanly |
