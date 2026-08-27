# Implementation Plan: Editor UX & Parameter Flexibility

**Branch**: `011-editor-ux-params` | **Date**: 2026-08-27 | **Spec**: `specs/011-editor-ux-params/spec.md`

**Input**: Feature specification from `specs/011-editor-ux-params/spec.md`

## Summary

Extend the OpenYourBox graph editor with (1) **nested copy-list tiling** — authored lengths in the dividing set `{1, N, M×N, …, P}` with under-the-hood expansion, editable short form + read-only P preview, and re-tile / flag-invalid on ancestor copy-count change; (2) shape binding keyword **`in`** on dim/channels/features fields; (3) left-menu **Project structure** under Library plus **expandable library entry trees** for whole-or-subpart insert, with **entries ordered by name**; (4) **sticky hierarchy trail** for visited children; (5) **resizable** left/right side menus; (6) LeakyReLU **negative slope** property (default 0.01, refuse out-of-range). All work stays inside the VST UI / graph / DSP / freeze-train workers — no new process.

## Technical Context

**Language/Version**: C++17 (VST / graph / UI / live DSP); Python 3.x (train/freeze workers for activation slope parity)

**Primary Dependencies**: JUCE, Dear ImGui, imgui-node-editor, existing `NodeGraph` / `NodeRenderer` / `UserBoxLibrary`, LibTorch live engine

**Storage**: Authored copy lists + `in` bindings in `GraphDocument` / `NodeProperty`; sticky trail & panel widths in session viewport/UI state (persist widths if layout prefs already exist); library payloads unchanged schema with optional nested-node insert path

**Testing**: C++ unit tests under `Tests/` (tiling math, dividing-set validation, `in` resolve, trail branch-clear); Python smoke for LeakyReLU slope in freeze/train; manual scenarios in `quickstart.md`

**Target Platform**: macOS VST3/AU (primary); Windows/Linux secondary if build allows

**Project Type**: Desktop audio plugin with embedded ImGui node editor

**Performance Goals**: 60 FPS UI while resizing panels and expanding library/project trees; copy-list re-tile and `in` shape refresh interactive (&lt; 100 ms for typical nests); no audio-thread work for UI/list parsing

**Constraints**: VST-only UI; zero audio-thread allocations; Shape Integrity (illegal cables refuse); `in` lowercase reserved; no mixing `in` with numbers in one field; LeakyReLU slope refuse (no clamp) outside [0, 1]; extend 006 library/group contracts rather than replace them

**Scale/Scope**: Nested groups depth ≥3 with product P ≥ 8; library entries with multi-level groups; left palette + right inspector resize; one new left-menu Project structure section

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Single Interface, Decoupled Compute**: Pass. UX, library tree, Project structure, and parameter editing live in the plugin; workers only receive slope/param values already on the architecture snapshot.
- **Dual-Engine Execution Model**: Pass. Blue/Gold unchanged; LeakyReLU slope applies in live activation and in freeze/train codegen the same way `gain` does.
- **Manual Granular Freeze Policy**: Pass. No change to freeze selection policy; slope is a normal element property included in freeze payload.
- **Shape Integrity & Legal Constraints**: Pass. `in` resolves to concrete dims before cable validation; invalid authored lengths and unresolved `in` refuse or mark illegal like other bad params; copyright/Train untouched.
- **Zero Audio-Thread Allocation Rule**: Pass. Parse/tile/`in` resolve/panel resize/trail updates on GUI thread; audio sees only committed expanded numeric params after publisher swap.
- **Complexity Justification**: Pass. Dividing-set tiling and Project structure are required by clarified spec; sticky trail and resizers are presentation-only.

**Post–Phase 1 re-check**: Still Pass. Contracts keep expansion/preview/authored storage on the GUI/document path; library subpart insert reuses `exportBox`/`importBox` without inventing external cables; workers only read a new float property.

## Project Structure

### Documentation (this feature)

```text
specs/011-editor-ux-params/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── copy-list-tiling-contract.md
│   ├── preserve-in-keyword-contract.md
│   ├── project-structure-library-tree-contract.md
│   ├── hierarchy-sticky-trail-contract.md
│   ├── resizable-side-menus-contract.md
│   └── leakyrelu-negative-slope-contract.md
└── tasks.md                 # /speckit-tasks (not created here)
```

### Source Code (repository root)

```text
OpenYourBox/
├── Source/
│   ├── PluginEditor.cpp / .h       # Right panel width splitter; persist widths if prefs exist
│   ├── graph/
│   │   ├── GraphTypes.h            # Authored copy-list length; in-binding; trail entries; parse/tile helpers
│   │   ├── NodeGraph.cpp / .h      # Dividing-set commit; re-tile/invalid on setGroupCopies; in resolve; export subpath
│   │   └── NodeRenderer.cpp / .h   # Authored field + read-only P preview; Project structure; sticky breadcrumb; left splitter
│   ├── library/
│   │   └── UserBoxLibrary.*        # Insert by nested node path/id within entry payload
│   ├── ui/
│   │   └── UserBoxLibraryPanel.*   # Expandable entry tree; Project structure section under Library
│   └── dsp/
│       ├── LiveGraphEngine.cpp     # LeakyReLU negative_slope from property (not hardcoded 0.01)
│       └── TCNModel.cpp            # Same for TCN activation path
Backend/
├── train_worker.py                 # _activation(LeakyReLU) uses element slope
└── freeze_worker.py                # Same
Tests/
└── (tiling, in-keyword, trail, library sub-insert fixtures)
```

**Structure Decision**: Extend existing `OpenYourBox` + `Backend` trees. No new app or process. Reuse `effectiveCopyCount`, `UserBoxLibrary`, and `renderScopeBreadcrumb` as anchors.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Authored L + derived P (dual representation) | Spec requires editable short form and read-only expanded preview | Always storing only P loses compact authoring and re-tile semantics |
| Project structure panel (second hierarchy UI) | Clarified as left menu under Library for navigate + save | Library-only expand cannot browse the live graph without opening groups |
