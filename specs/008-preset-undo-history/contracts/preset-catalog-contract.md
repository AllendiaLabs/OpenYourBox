# Contract: Preset Catalog UI & Persistence

## Purpose

In-plugin **Presets** catalog for named full-patch recall, distinct from:

- Training Library (audio samples)
- Box Library (reusable elements/groups)
- DAW project save/load (host state)

## Placement

- Dedicated ImGui section/tab labeled **Presets** in the plugin window
- Primary actions: **Save**, **Save As**, Load selected, Rename, Delete
- Chrome shows **current preset name** and **dirty** indicator when associated (FR-017)
- Must not share storage or list rows with Boxes or Training Library

## Entry (UI-visible)

| Field | Notes |
|-------|--------|
| Name | Unique in catalog |
| Updated | Timestamp |
| (optional) hint | e.g. schema version — not required for v1 |

Internal: id, payload path, schema version.

## Current preset + dirty

| Event | Current name | Dirty |
|-------|--------------|-------|
| Successful load | Set to loaded name | false |
| Undoable patch edit | Unchanged | true |
| Successful Save (overwrite current) | Unchanged | false |
| Successful Save As | Set to new name | false |
| Undo back to post-load/save baseline | Unchanged | false |
| Pan / zoom / selection only | Unchanged | unchanged |
| Undo of preset load | Restored to prior association | prior dirty restored |

## Core actions (v1)

1. **Save** — when current preset is set and dirty (or user confirms), overwrite that catalog entry’s payload with current `PatchSnapshot`; refuse if no current name (direct user to Save As)
2. **Save As** — capture current `PatchSnapshot` under a user-provided name; refuse empty name; overwrite only after confirm when name exists and is not the current Save target’s own overwrite path
3. **Browse** — list entries by name
4. **Load** — apply snapshot to the live instance; record **one** undo step; set current name clean; refuse corrupt payloads with toast; leave live patch unchanged on failure
5. **Rename** — unique name; conflict → block or confirm per box-library UX consistency
6. **Delete** — confirm; remove index row + payload folder
7. **Empty state** — prompt user to save the current patch as a preset

## Persistence

- Root: `openyourbox::library::presetsDirectory()` → `userDataRoot()/UserPresets`
- `index.json` + per-entry folders (`patch` snapshot + `artifacts/` as needed)
- Atomic index writes (temp + rename), same pattern as `UserBoxLibrary`
- Survives plugin/DAW relaunch; independent of open project
- Not cloud-synced in v1

## Fidelity

Loading a preset must restore graph structure, parameters, and sound-relevant state (weights/Gold) equivalent to a successful full sonic recall of the same snapshot (see `patch-snapshot-contract.md`).

## Failure modes

| Case | Behavior |
|------|----------|
| Empty name | Refuse save |
| Corrupt/missing payload or unrestorable weights/Gold | Refuse load; live patch unchanged |
| Name collision (Save As) | Confirm overwrite or cancel |
| Audio playing during load | Allowed; seamless apply via shared snapshot path |

## Non-Goals

- Factory preset bank
- MIDI program change / host program list sync
- Marketplace publish
- Undoing catalog rename/delete (only live patch load is undoable)

## Implementation anchors

- NEW: `OpenYourBox/Source/library/UserPresetLibrary.h/.cpp`
- NEW: `OpenYourBox/Source/ui/UserPresetPanel.h/.cpp`
- `OpenYourBox/Source/library/UserDataPaths.h` (`presetsDirectory`)
- Pattern reference: `UserBoxLibrary` + `UserBoxLibraryPanel`
- Orchestration: `PluginEditor.cpp` + processor apply API
