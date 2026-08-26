# Quickstart: Preset Management & Undo/Redo History

Manual / integration validation for `008-preset-undo-history`. Prefer a Debug/Release plugin build loaded in a host.

## Prerequisites

- Branch `008-preset-undo-history` built (`OpenYourBox` VST3/AU).
- Host session with one OpenYourBox instance.
- A non-trivial graph (several nodes, at least one parameter change, optionally Gold/weights).

## 1. Save, dirty, Save As, and load

1. Configure graph + parameters/weights so the sound is distinctive.
2. Open **Presets** → **Save As** as `PatchA` (no current name yet).
3. Confirm chrome shows `PatchA` and not dirty.
4. Change the graph or a parameter once.
5. **Expect**: dirty indicator on.
6. **Save** (overwrite current).
7. Change again, then **Save As** `PatchB`.
8. Load `PatchA`.
9. **Expect**: structure and sonic result match what was saved for `PatchA`; current name `PatchA` clean; audio continues without restarting the plugin (SC-001, SC-002, FR-006).

## 2. Catalog manage

1. With `PatchB` present, rename `PatchB` → `PatchB2`.
2. Delete `PatchA` with confirm; cancel once to verify no delete; then confirm delete.
3. Quit host; reopen plugin.
4. **Expect**: `PatchB2` listed; `PatchA` gone (FR-003–005).

## 3. Overwrite confirm

1. Save As current state as `PatchB2` again.
2. **Expect**: overwrite confirmation; cancel leaves previous payload; confirm replaces it.

## 4. Undo / redo edits

1. Perform ≥5 discrete edits (add node, connect, change a property, move a node, delete a link).
2. Undo three times; verify intermediate states.
3. Redo twice; verify recovery.
4. Make a new edit.
5. **Expect**: redo cleared; history matches FR-007–009 / SC-003.

## 5. Coalesced gesture

1. Drag a knob or XY control continuously, then release.
2. Undo once.
3. **Expect**: entire gesture reverts in one step (SC-004, FR-008).

## 6. Randomize as one step

1. Note current sound/weights; trigger weight randomization.
2. Undo once.
3. **Expect**: prior sonic state restored (SC-004a, FR-007a).

## 7. View-only excluded

1. Pan and zoom the canvas; change selection.
2. Invoke Undo.
3. **Expect**: history unchanged; dirty unchanged if it was clean (FR-007b).

## 8. Preset load as one undo step

1. Make a few edits (ensure undo available); note dirty/current if any.
2. Load `PatchB2`.
3. Undo once.
4. **Expect**: pre-load patch restored **including** prior current-preset name and dirty; Redo restores the loaded preset (Story 5, FR-011).

## 9. Host state still works

1. With a loaded/edited patch, save the DAW project; reload the project/session.
2. **Expect**: session patch restores correctly even after using the Presets catalog (SC-007, FR-013).

## 10. Empty / failure paths

1. Open Presets with empty catalog → empty-state message.
2. Attempt save with blank name → refused.

## Automated hooks (when implemented)

- Unit: `UserPresetLibrary` index CRUD, overwrite, delete payload cleanup — `Tests/UserPresetLibraryTests.cpp`
- Unit: `PatchSnapshot` round-trip equality for graph + parameters + weights when present — `Tests/PatchSnapshotTests.cpp`
- Unit: `EditHistory` depth cap, redo clear, gesture coalesce, randomize one step, preset-load single step, pan/zoom ignored — `Tests/EditHistoryTests.cpp`
- Integration: `Tests/ProcessorIntegrationTests.cpp` covers `PatchSnapshot`-based `getStateInformation` / `setStateInformation`

Manual host walkthrough (Save/Load chrome, shortcuts, freeze/train history, Ableton project recall) still required after installing the built VST3/AU.

## References

- [data-model.md](./data-model.md)
- [contracts/patch-snapshot-contract.md](./contracts/patch-snapshot-contract.md)
- [contracts/preset-catalog-contract.md](./contracts/preset-catalog-contract.md)
- [contracts/edit-history-contract.md](./contracts/edit-history-contract.md)
