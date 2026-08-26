# Implementation Plan: Preset Management & Undo/Redo History

**Branch**: `008-preset-undo-history` | **Date**: 2026-08-26 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/008-preset-undo-history/spec.md` (clarified 2026-08-26)

## Summary

Add an in-plugin **named preset catalog** for full sonic patches (graph + APVTS parameters + weights/Gold artifacts), with browse, **Save / Save As**, load, rename, and delete—distinct from DAW project state and from the user box library. Track **current preset name + dirty** after load/edit. Add **session undo/redo** for patch-affecting edits only (exclude pan/zoom/selection), with gesture coalescing, one-shot sonic actions (e.g. randomize) as single steps, and preset load as one undoable step. Extract a shared **`PatchSnapshot`** path used by host state, presets, and history so recall fidelity stays consistent; apply on the GUI thread with atomic runtime publish so audio continues without a plugin restart.

## Technical Context

**Language/Version**: C++17 (VST shell, graph, UI); Python worker unchanged

**Primary Dependencies**: JUCE (`AudioProcessor` state XML, `UndoManager` / `UndoableAction`, `ValueTree`, file I/O), Dear ImGui (preset browser + undo/redo chrome), existing `NodeGraph` / `OpenYourBoxAudioProcessor` / `OpenYourBoxAudioProcessorEditor`, LibTorch weight archives already used in `getStateInformation` / `setStateInformation`

**Storage**: Named presets under `openyourbox::library::userDataRoot()/UserPresets` (`index.json` + per-entry payload); undo/redo in memory per plugin instance (depth ≥ 50); DAW project state continues via existing processor state APIs

**Testing**: C++ unit/integration under `Tests/` (preset CRUD, snapshot round-trip, undo stack/coalesce/redo clear, preset-load-as-one-step, dirty/current-preset); manual host scenarios in `quickstart.md`

**Target Platform**: macOS VST3/AU (primary); Windows/Linux secondary if build allows

**Project Type**: Desktop audio plugin with embedded ImGui node editor

**Performance Goals**: Preset load and undo/redo user-visible update &lt; 1 s for typical patches; 60 FPS UI; no host transport stop; zero audio-thread allocations on apply

**Constraints**: VST-only UI; presets = full sonic patches (not boxes); undo session-scoped; coalesce continuous gestures; exclude view-only history; factory/MIDI program banks out of scope; shared snapshot must not break existing DAW save/load

**Scale/Scope**: Dozens of user presets in v1; undo depth ≥ 50 steps; graphs with groups, Gold artifacts, and weights must round-trip

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Single Interface, Decoupled Compute**: Pass. Preset catalog and undo live entirely in the VST UI/processor; no standalone app; training worker unused for preset/history CRUD.
- **Dual-Engine Execution Model**: Pass. Snapshot restore may rebuild Blue and/or Gold runtime the same way session restore already does; no third engine.
- **Manual Granular Freeze Policy**: Pass. Freeze/unfreeze remain explicit user actions; history only records resulting patch changes after they land (no mid-compile scrubbing).
- **Shape Integrity & Legal Constraints**: Pass. Restored graphs re-enter existing connection/shape rules; copyright/Train gates untouched.
- **Zero Audio-Thread Allocation Rule**: Pass. Snapshot I/O, undo stack mutation, and model rebuild stay on GUI/background threads; audio thread only consumes atomically published runtime (same pattern as today’s `setStateInformation` / graph compile).
- **Complexity Justification**: Pass. Shared patch snapshot avoids divergent serialize paths; snapshot-based undo is simpler and safer than inverse-command stacks for a mutable node graph with weights.

**Post–Phase 1 re-check**: Still Pass. Contracts require GUI-thread apply + atomic publish; preset catalog mirrors box-library user-data pattern under a separate `UserPresets` folder; current-preset/dirty and undo exclusions documented without audio-thread work.

## Project Structure

### Documentation (this feature)

```text
specs/008-preset-undo-history/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── patch-snapshot-contract.md
│   ├── preset-catalog-contract.md
│   └── edit-history-contract.md
└── tasks.md                 # /speckit-tasks (not created here)
```

### Source Code (repository root)

```text
OpenYourBox/
├── Source/
│   ├── PluginProcessor.cpp / .h     # Extract/share PatchSnapshot serialize+apply; wire UndoManager; preset load
│   ├── PluginEditor.cpp / .h        # Preset panel, Undo/Redo UI + shortcuts, gesture begin/end, current/dirty chrome
│   ├── library/
│   │   ├── UserDataPaths.h          # Add presetsDirectory() → userDataRoot()/UserPresets
│   │   ├── UserPresetLibrary.h/.cpp # NEW: catalog CRUD, Save/Save As payloads
│   │   └── UserBoxLibrary.*         # Unchanged; keep boxes ≠ presets
│   ├── state/                       # NEW: shared patch + history helpers
│   │   ├── PatchSnapshot.h/.cpp     # Serialize/apply full patch (graph + APVTS + weights bytes)
│   │   └── EditHistory.h/.cpp       # UndoManager façade, coalesce transactions, depth cap, suppress-on-apply
│   ├── graph/
│   │   └── NodeGraph.*              # Ensure toValueTree/fromValueTree sufficient for full restore
│   └── ui/
│       └── UserPresetPanel.h/.cpp   # NEW: ImGui preset browser (Save/Save As/load/rename/delete)
Tests/
├── UserPresetLibraryTests.cpp       # NEW
├── PatchSnapshotTests.cpp           # NEW
└── EditHistoryTests.cpp             # NEW
```

**Structure Decision**: Extend the existing single-plugin `OpenYourBox` tree. Presets parallel `UserBoxLibrary` under `library/` + ImGui panel; shared `PatchSnapshot` is the single serialize/apply path used by DAW state, presets, and undo steps. No new process or app.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| In-memory full-patch undo snapshots (incl. weights when present) | Spec requires sonic/graph fidelity on undo (e.g. after randomize) and preset-load as one step | Command-inverse stack cannot reliably invert freeze/train/weight outcomes; structure-only undo fails SC sonic recall |
| Shared `PatchSnapshot` extracted from processor state APIs | Keep DAW session, presets, and history consistent | Separate ad-hoc serializers diverge and break SC-007 |
