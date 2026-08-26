# Data Model: Preset Management & Undo/Redo History

## Entity: PatchSnapshot

Logical full-plugin patch used by DAW state, presets, and undo.

| Field | Type | Notes |
|-------|------|--------|
| schemaVersion | int | Snapshot format version (start at 1) |
| parameterState | ValueTree / XML | APVTS copy (`parameters.copyState()`) |
| graphDocument | ValueTree | Full `GraphDocument` from `NodeGraph::toValueTree()` |
| weightsBlob | optional bytes | LibTorch archive when a published model exists (same role as today’s Base64 `weights` attribute) |
| architectureHash | optional string/int | Validates weight load against rebuilt model |
| randomizationCounter | uint64 | Matches processor counter for recall |
| lastTrainObjective | string | Existing train-objective attribute |

**Validation**:
- `graphDocument` must parse; unknown node types → apply fails closed (no partial graph)
- Empty/whitespace preset names rejected at catalog layer (not inside snapshot)
- Weight load failure → keep rebuilt seed/counter model (parity with current `setStateInformation`)

**Relationships**: Embedded in `PresetEntry` payload; stored inside each `HistoryStep`; produced/consumed by processor apply.

## Entity: PresetEntry

One named catalog row.

| Field | Type | Notes |
|-------|------|--------|
| id | UUID string | Stable catalog id / payload folder name |
| name | string | Unique display name in catalog |
| createdAt | ISO-8601 | Set on create |
| updatedAt | ISO-8601 | Set on save/rename/overwrite |
| schemaVersion | int | Entry metadata version |
| payloadRelativePath | string | Folder under `UserPresets` |
| patchFile | file | Serialized `PatchSnapshot` (XML or binary wrapper) |
| artifacts/ | optional files | Copied weight/BlackBox files referenced by graph; paths rewritten relative |

**Validation**:
- Name unique; overwrite only after confirm (Save As / import / colliding first save)
- Delete removes index row + payload folder
- Corrupt payload → load refused; catalog row may remain until user deletes

## Entity: PresetCatalog

| Field | Type | Notes |
|-------|------|--------|
| rootDirectory | path | `openyourbox::library::presetsDirectory()` → `userDataRoot()/UserPresets` |
| entries | list of PresetEntry | Insertion or name-sorted for UI |
| index.json | file | Durable index (atomic write, same pattern as boxes) |

**Relationships**: Owns `PresetEntry` rows; independent of open project and of Box Library.

## Entity: CurrentPreset

Session association on the live plugin instance (not persisted in catalog).

| Field | Type | Notes |
|-------|------|--------|
| entryId | optional UUID | Set after load/Save/Save As |
| name | optional string | Display name for chrome |
| dirty | bool | True after undoable patch-affecting edit while associated |
| baselineFingerprint | optional hash/id | Optional helper to clear dirty when undo returns to post-load/save snapshot |

**State transitions**:
1. Load success → name set, dirty=false, baseline captured
2. Undoable patch edit → dirty=true
3. Save (overwrite current) success → dirty=false, baseline refreshed
4. Save As success → name/id switch to new entry, dirty=false
5. Undo to baseline → dirty=false
6. View-only pan/zoom/selection → no change
7. Undo of preset load → restore prior CurrentPreset (including prior dirty)

## Entity: HistoryStep

One undoable unit.

| Field | Type | Notes |
|-------|------|--------|
| label | string | e.g. “Add node”, “Load preset X”, “Parameter edit”, “Randomize” |
| before | PatchSnapshot | State prior to the change |
| after | PatchSnapshot | State after the change |
| beforeCurrentPreset | CurrentPreset | Association before the change (for load/undo fidelity) |
| afterCurrentPreset | CurrentPreset | Association after the change |
| coalesced | bool | True when produced by gesture begin/end |

**Validation**:
- Failed operations must not push a step
- Preset load = exactly one step on success
- Pan/zoom/selection never create a step

## Entity: EditHistory

Session undo/redo controller (per plugin instance).

| Field | Type | Notes |
|-------|------|--------|
| maxDepth | int | Default 50 |
| undoStack / redoStack | via UndoManager | Cleared redo on new edit |
| openGesture | optional | Active coalesce transaction |
| suppressPush | bool | True while applying snapshot for undo/redo/host restore/preset apply mid-flight |

**State transitions**:
1. Idle → GestureOpen (`beginGesture`) → Idle+PushStep (`endGesture`)
2. Idle → PushStep (discrete action / randomize / successful preset load)
3. Undo: apply `before` + restore `beforeCurrentPreset`; enable redo
4. Redo: apply `after` + `afterCurrentPreset`
5. New PushStep after Undo: clear redo
6. Processor destroy: history discarded

## Entity: PortablePresetPackage

Export/import file.

| Field | Type | Notes |
|-------|------|--------|
| manifest | JSON | name hint, schemaVersion, artifact list |
| patch | PatchSnapshot file | Required |
| artifacts | files | Optional accompanying binaries |

**Validation**: Missing/invalid manifest or patch → import refused; catalog unchanged. Name collision → overwrite confirm.

## Relationship summary

```text
PresetCatalog 1──* PresetEntry 1──1 PatchSnapshot (+ artifacts)
PluginInstance 1──1 CurrentPreset
PluginInstance 1──1 EditHistory 1──* HistoryStep (before/after PatchSnapshot + CurrentPreset)
PluginInstance 1──1 live PatchSnapshot
DAW host state ── uses same PatchSnapshot serialize/apply
```

## Identity & lifecycle notes

- Loading a preset replaces the live patch and sets CurrentPreset; does not modify the catalog entry
- Saving a preset copies from live patch into catalog (artifacts duplicated into entry folder)
- Box Library remains a separate catalog of component snapshots (not modeled here)
- Undo does not write the preset catalog unless the undone/redone action was itself a catalog mutation (v1: catalog mutations are not on the undo stack — only patch edits and preset *load* into the instance)
- Host project save/load remains independent of the named catalog and of session undo
