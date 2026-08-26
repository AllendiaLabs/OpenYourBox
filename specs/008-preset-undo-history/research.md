# Phase 0 Research: Preset Management & Undo/Redo History

## Decision 1: Shared `PatchSnapshot` for DAW state, presets, and undo

**Decision**: Extract a single **patch snapshot** serialize/apply path from today’s `OpenYourBoxAudioProcessor::getStateInformation` / `setStateInformation` (APVTS XML + `GraphDocument` child + Base64 weight archive attributes + train-objective / randomization metadata). Host state, named presets, and undo steps all capture/apply the same logical document. Apply always runs off the audio thread and republishes runtime the same way session restore already does.

**Rationale**: Spec FR-001 / FR-001a, FR-014, SC-002, SC-007 — one fidelity bar for identical sonic recall. Avoids three divergent serializers.

**Alternatives considered**:
- Preset-only graph JSON without APVTS/weights: rejected — incomplete patch (clarified: full sonic recall).
- Keep host XML format private and invent a second preset format: rejected — drift risk.
- Diff-only undo without full snapshots: rejected for v1 — harder correctness for graph+weights.

## Decision 2: Named presets as a user-data catalog (not box library, not factory bank)

**Decision**: Add `UserPresetLibrary` mirroring `UserBoxLibrary`: `index.json` + per-entry folders under `openyourbox::library::presetsDirectory()` → `userDataRoot()/UserPresets`. Each entry stores display name, id, timestamps, schema version, and a payload file (XML/binary patch snapshot) plus copied weight/BlackBox artifacts when referenced by the graph. UI: dedicated ImGui **Presets** panel with **Save**, **Save As**, load, rename, delete-with-confirm, overwrite-with-confirm, export/import. Factory banks and MIDI program changes are out of scope.

**Rationale**: Spec Stories 1–2, FR-001–005, FR-015; `userDataRoot` already lives under the OS Audio Presets tree on macOS, so a `UserPresets` child avoids colliding with Weights/Samples/Boxes.

**Alternatives considered**:
- Store only inside DAW project: rejected — must survive across projects (FR-005).
- Reuse Box Library entries as “presets”: rejected — boxes are components; presets are full patches.
- Host-only program list (no in-plugin UI): rejected — product wants in-VST management.

## Decision 3: Current preset identity + dirty flag

**Decision**: After successful load or Save/Save As, the editor tracks `currentPresetId`/`currentPresetName` and `dirty`. Any **undoable patch-affecting edit** sets dirty. **Save** overwrites the current catalog entry without picking a new name (optional brief confirm). **Save As** prompts for a name; colliding names require overwrite confirm. Undoing back to the exact post-load/post-save snapshot clears dirty. View-only pan/zoom/selection never sets dirty.

**Rationale**: Clarification Session 2026-08-26 (Option B); FR-002 / FR-002a / FR-002b / FR-018.

**Alternatives considered**:
- No current-preset concept (always Save As): rejected — poor overwrite UX.
- Clear name on any edit: rejected — forces retyping; loses Save target.

## Decision 4: Export/import = portable single-file package

**Decision**: Export packs one catalog entry into a single portable file (zip-like or self-contained archive with manifest + patch XML + artifacts). Import unpacks into `UserPresets`, creates/updates a catalog row (name prompt; **overwrite confirm** on collision), and refuses corrupt/incompatible packages with a clear message and no catalog mutation.

**Rationale**: Spec Story 4, FR-012 / FR-012a, SC-008; clarification on import collision = overwrite prompt.

**Alternatives considered**:
- Raw XML only without artifacts: rejected — Gold/weight fidelity breaks.
- Auto-rename on import collision: rejected by clarification (prompt overwrite).

## Decision 5: Snapshot-based undo via JUCE `UndoManager`

**Decision**: Implement `EditHistory` wrapping `juce::UndoManager` with undoable actions that store **before/after `PatchSnapshot`** (or “replace document with snapshot X”). Default max depth **50** (FR-016; assumption ≥ 50). New edit after undo clears redo (UndoManager default). Session-scoped only. UI: Undo/Redo buttons + platform shortcuts. Disable affordances when `canUndo`/`canRedo` are false (FR-017).

**Rationale**: Spec Story 3, FR-007–010; JUCE is already the VST shell; full snapshots match preset fidelity and handle randomize/freeze outcomes once applied.

**Alternatives considered**:
- Per-node inverse commands: rejected — incomplete for weights/Gold/train absorb.
- Persist undo across plugin close: rejected by spec assumption.
- Unlimited depth: memory risk with weight blobs.

## Decision 6: What is / is not on the undo stack

**Decision**:
- **Included**: graph structure, membership/grouping, connections, **box layout positions**, parameter/control values, one-shot sonic actions (weight randomization), completed train/freeze patch mutations, preset **load**.
- **Excluded**: canvas **pan**, **zoom**, **selection**; in-flight train/freeze jobs; catalog-only rename/delete (only live patch load is undoable); host project load.

Continuous knobs/XY/node drags use begin/end gesture so one history step. Discrete clicks push one step each.

**Rationale**: Clarifications on randomize (A) and view-only exclusion (B); FR-007 / FR-007a / FR-007b / FR-008.

**Alternatives considered**:
- Include pan/zoom/selection: rejected — eats creative undo steps.
- Omit weights from undo: fails undo-after-randomize.

## Decision 7: Preset load is one undoable step (preserves current/dirty)

**Decision**: Before applying a loaded preset, capture current patch **and** current-preset association (name + dirty) as the undo “before” state; after successful apply, commit one history action. Failed load pushes nothing. Undo restores pre-load patch **and** prior current/dirty; redo restores loaded patch and clean current name.

**Rationale**: Spec Story 5, FR-011; acceptance scenarios include restoring current-preset name and dirty flag.

**Alternatives considered**:
- Clear history on load: rejected by Story 5.
- Push per-field diffs during load: unnecessary complexity.

## Decision 8: Audio-safe apply path

**Decision**: All snapshot applies reuse the existing restore pipeline: set restoring/`suppressHistory` flag, replace APVTS + graph ValueTree on GUI thread, rebuild/publish runtime, `requestGraphCompile()` as today. Never allocate or rebuild models on the audio thread.

**Rationale**: Constitution zero-allocation audio rule; Spec FR-006, SC-005.

**Alternatives considered**:
- Apply directly on audio thread: forbidden.
- Stop/restart plugin instance on load: rejected by SC-005 / FR-006.

## Decision 9: Memory bound for undo with weights

**Decision**: Cap depth at 50. Store weight archive bytes only when present (same as host state). If memory pressure becomes an issue in testing, drop oldest steps first and optionally spill large weight blobs to temp under user-data — implement spill only if tests show resident memory problems; plan assumes in-memory first.

**Rationale**: Spec finite depth; keep v1 simple.

**Alternatives considered**:
- Always spill to disk: extra I/O before proven need.
- Omit weights from undo: fails sonic undo fidelity.

## Decision 10: Scope exclusions remain firm

**Decision**: Out of scope: factory preset banks, MIDI program change maps, cloud sync/marketplace, merging Presets UI into Box Library or Training Library tabs, undoing an in-flight train/freeze job (only the completed patch mutation is a history step).

**Rationale**: Spec Assumptions; keeps Phase 1 contracts bounded.
