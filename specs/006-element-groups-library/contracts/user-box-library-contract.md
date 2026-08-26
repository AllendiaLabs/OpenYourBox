# Contract: User Box Library

## Purpose

Defines a durable in-plugin **Box Library** for reusable single elements and groups (with parameters and artifacts), distinct from the Training Library (audio sample pairs).

## Placement

- Reachable from the plugin window near the element palette and/or a dedicated **Boxes** section/tab — must not be confused with the Training **Library** tab.
- Save entry point: per-box context action on one node or one group (not multi-select).
- Place entry point: list drag-and-drop or Place button onto the graph canvas (same gesture family as palette insert).

## Entry (UI-visible)

| Field | Notes |
|-------|--------|
| Display name | Unique in catalog; overwrite requires confirm |
| Kind | `Element` \| `Group` badge |
| Root hint | Element type name or “Group” |
| Updated | Timestamp |

Internal: entry id, payload path, schema version, artifact manifests.

## Core actions (v1)

1. **Save box** — from one focused element or group; name prompt; refuse Audio I/O and multi-select.
2. **Browse** — ImGui list of entries (name, kind).
3. **Place / insert** — clone into current graph at drop/insert position; new IDs; **all groups collapsed**.
4. **Rename** — edit display name (conflict → confirm or block).
5. **Overwrite** — save under existing name only after confirm.
6. **Delete** — confirm; removes index row + payload folder.
7. **Empty state** — message that user can save boxes from the graph.

## Persistence

- Root: `UserDataPaths::boxesDirectory()` (new helper beside `samplesDirectory` / `weightsDirectory`).
- `index.json` + per-entry folders (`box` snapshot + `artifacts/`).
- Survives plugin/DAW relaunch; independent of open project.
- Not cloud-synced in v1.

## Fidelity

Snapshot must restore:
- Structure (nested groups, internal links, **copies N and all materialized instances**)
- User-facing parameters / properties / seeds / conditioning
- Weight or BlackBox artifacts when files exist (copied into entry; re-copied or linked into project weights on insert)

## Failure modes

| Case | Behavior |
|------|----------|
| Unknown element type on insert | Error toast; graph unchanged |
| Corrupt/missing artifact | Error; no partial insert (transactional best-effort: add nothing or roll back new nodes) |
| Name collision on save | Confirm overwrite or cancel |
| Multi-select save | Action hidden/disabled with reason |

## DnD payload

- Suggest `OPENYOURBOX_BOX_LIBRARY_ID` (string entry id), parallel to existing `OPENYOURBOX_NODE_TYPE`.

## Non-Goals

- Marketplace publish / cloud sync
- Cross-machine export/import UI (deferred)
- Merging with Training Library storage
- Saving raw multi-selections without a group

## Implementation anchors

- NEW: `OpenYourBox/Source/library/UserBoxLibrary.h/.cpp`
- NEW: `OpenYourBox/Source/ui/UserBoxLibraryPanel.h/.cpp`
- `OpenYourBox/Source/library/UserDataPaths.h`
- Pattern reference: `TrainingLibrary` + `TrainingLibraryPanel`
- Orchestration: `PluginEditor.cpp`
