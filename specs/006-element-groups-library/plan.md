# Implementation Plan: Element Groups & User Box Library

**Branch**: `006-element-groups-library` | **Date**: 2026-08-26 | **Spec**: `specs/006-element-groups-library/spec.md`

**Input**: Feature specification from `specs/006-element-groups-library/spec.md`

## Summary

Add durable hierarchical **groups** (with nested subgroups) to the embedded imgui-node-editor graph, **expand/collapse** as a presentation-only UI affordance that exposes external ports when collapsed, a per-group **copies (N)** parameter that **materializes N independent serial copies** when I/O shapes allow, and a **user box library** for saving/inserting a single element or group box (with parameters and weight/artifact fidelity) from local user data. Prefer imgui-node-editor’s native `Group` / group-hint APIs for expanded frames; implement collapse and library catalog with standard Dear ImGui widgets mirroring the Training Library pattern. Freeze on a group/multi-selection freezes **each freezable member individually** (no single BlackBox for the whole selection). Audio I/O cannot be grouped or saved.

## Technical Context

**Language/Version**: C++17 (VST / graph / UI); Python worker unchanged for this feature’s core path

**Primary Dependencies**: JUCE, Dear ImGui, imgui-node-editor (`ax::NodeEditor::Group`, group hints, selection/context menus), existing `NodeGraph` / `NodeRenderer`, LibTorch artifacts for weight/BlackBox fidelity

**Storage**: Project groups in `GraphDocument` ValueTree; user box library under `UserDataPaths` (new boxes directory + per-entry `index` + payload files); weight/`.pt` copies beside entries when needed

**Testing**: C++ unit/integration under `Tests/` (group membership, cycle checks, serialize round-trip, library CRUD/insert); manual plugin scenarios in `quickstart.md`

**Target Platform**: macOS VST3/AU (primary); Windows/Linux as secondary if build allows

**Project Type**: Desktop audio plugin with embedded ImGui node editor

**Performance Goals**: 60 FPS UI with nested groups; collapse/expand &lt; 2 s interaction; no audio-path change on collapse; library insert of moderate groups (&lt; 30 nodes) feels interactive

**Constraints**: VST-only UI; zero audio-thread allocations; collapse presentation-only; no Audio I/O in groups/library; save-to-library is per-box not multi-select; prefer imgui-node-editor/ImGui over custom non-ImGui surfaces; freeze = per freezable member

**Scale/Scope**: ≥5 nesting levels; group copies N up to at least 8 in guided tests; graphs of 50+ nodes with several groups; local library catalogs of dozens of entries in v1

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Single Interface, Decoupled Compute**: Pass. Groups, collapse, and box library live entirely in the plugin UI; no standalone app; training worker unused for grouping/library CRUD.
- **Dual-Engine Execution Model**: Pass. Groups do not invent a third engine; Blue/Gold semantics unchanged; collapse does not alter live vs frozen execution.
- **Manual Granular Freeze Policy**: Pass with clarified product rule. Freeze remains explicit user action; for this feature, freeze applies **per freezable member** of the selection (not “selection → one BlackBox”). Train absorb → single BlackBox remains a separate path.
- **Shape Integrity & Legal Constraints**: Pass. Connection validation unchanged; library insert reuses shape rules; copyright/Train gates untouched.
- **Zero Audio-Thread Allocation Rule**: Pass. Group membership, collapse toggles, library I/O, and freeze prep stay on GUI/background threads; audio continues to see full topology (collapse is UI-only).
- **Complexity Justification**: Pass. Hierarchy + library are required by the spec; collapse needs app-level behavior because imgui-node-editor has Group geometry but no collapse API (see `research.md`).

**Post–Phase 1 re-check**: Still Pass. Contracts keep audio topology independent of collapse state; library store mirrors Training Library under user data; freeze contract documents per-member behavior vs train absorb.

## Project Structure

### Documentation (this feature)

```text
specs/006-element-groups-library/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── group-editor-ui-contract.md
│   ├── group-copies-contract.md
│   ├── user-box-library-contract.md
│   └── freeze-per-member-contract.md
└── tasks.md                 # /speckit-tasks (not created here)
```

### Source Code (repository root)

```text
OpenYourBox/
├── Source/
│   ├── PluginEditor.cpp / .h          # Wire group + library actions; persist graph
│   ├── PluginProcessor.cpp / .h       # GraphDocument state (groups already via NodeGraph)
│   ├── graph/
│   │   ├── GraphTypes.h               # GroupId, membership, collapse flags; library snapshot DTOs
│   │   ├── NodeGraph.cpp / .h         # Group CRUD, nesting rules, serialize, freeze-per-member
│   │   └── NodeRenderer.cpp / .h      # imgui Group frames, collapse rendering, context menus, DnD
│   ├── library/
│   │   ├── UserDataPaths.h            # Add boxesDirectory()
│   │   ├── UserBoxLibrary.h / .cpp    # NEW: catalog CRUD + payload files
│   │   └── TrainingLibrary.*          # Pattern reference only (unchanged samples library)
│   ├── ui/
│   │   └── UserBoxLibraryPanel.h/.cpp # NEW: ImGui list + place; distinct from Training Library tab
│   ├── freeze/
│   │   └── FreezeCoordinator.*        # Invoke per-member freeze jobs when selection expands to members
│   └── dsp/
│       └── LiveGraphPublisher.*       # Ignore collapse; compile full membership topology
Tests/
└── (group serialize, library insert, freeze-per-member fixtures)
```

**Structure Decision**: Extend the existing single-plugin `OpenYourBox` tree. Groups live in `graph/`; box library parallels `TrainingLibrary` under `library/` + a dedicated ImGui panel; no new process or app.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| App-level collapse (beyond imgui `Group`) | Spec requires hide-internals + external ports | imgui-node-editor has no collapse/expand or group pins API |
| Explicit `parentGroupId` membership (not geometry-only) | Durable nest/save/reload and cycle checks | Spatial-only `GetGroupedNodes` does not survive serialize or nested collapse reliably |
