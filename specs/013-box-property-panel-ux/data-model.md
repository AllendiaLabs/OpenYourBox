# Data Model: Box Property Panel UX

**Feature**: `013-box-property-panel-ux` | **Date**: 2026-08-28

## Overview

No change to persisted parameter schemas. This feature adds **session selection/inspect state**, a **Parameters panel binding**, and a **reparent mutation** that clears links then updates hierarchy membership using existing graph entities.

## Entities

### GraphNode / GraphGroup (existing)

- **Identity**: `BoxId`
- **Relevant fields**: `label`, `properties`, pin lists, `parentGroupId` / membership, `position`, `size`
- **Change**: Canvas chrome no longer mirrors `properties` as inline widgets; `size` reflects slim name+pins layout only
- **Relationships**: unchanged parent/child group membership; links via pins

### GraphLink (existing)

- **Role in reparent**: All links incident to a reparented box’s pins are **deleted** before membership change
- **Validation**: After reparent, no dangling links remain by construction

### SelectionContext (session, new)

| Field | Type | Notes |
|-------|------|-------|
| `kind` | `None` \| `Live` \| `LibraryInspect` \| `Multi` | Drives Parameters editability |
| `liveBoxId` | optional `BoxId` | Primary selected node or group when `Live` |
| `libraryEntryId` | optional string/id | When `LibraryInspect` |
| `libraryNestedRootId` | optional id | Subpart within entry snapshot |
| `forceParametersTab` | bool / one-shot | Set when selection changes per FR-003/007/008 |

**Transitions**:
- Canvas select / structure single-click → `Live` + force Parameters
- Library single-click → `LibraryInspect` + force Parameters (read-only)
- Multi-select → `Multi` (simplified panel; no full param edit)
- Empty click / clear → `None`
- Successful structure drop → `Live` on new/moved id + force Parameters

### ProjectStructureDropTarget (session, transient)

| Field | Type | Notes |
|-------|------|-------|
| `targetParentId` | optional `BoxId` / root sentinel | Highlighted group or project root |
| `valid` | bool | False for cycles / illegal parents |
| `highlightBounds` | UI rect | Rectangle around label + nested rows |

Cleared on drag end or cancel.

### LibraryInspectBinding (session)

- Points at `UserBoxLibrary` entry snapshot (and optional nested member)
- **Read-only** projection of properties for Parameters tab
- Must not call save/rename/property-write APIs on the catalog from this binding

## Mutations

### disconnectAllLinksForBox(boxId)

1. Collect all links with either end on `boxId`’s pins
2. Remove each link (same bookkeeping as delete-node link cleanup)
3. Document dirty; shape refresh as today

### reparentBoxLikeInsert(boxId, targetParentOrRoot)

1. If target is descendant of `boxId` (when group) or otherwise illegal → **no-op**, return failure
2. `disconnectAllLinksForBox(boxId)`
3. Detach from current parent (`removeFromGroup` or equivalent)
4. Attach to target (`addToGroup`) or leave at root
5. Apply **new-item placement** position in destination canvas scope
6. Emit single undoable gesture

### insertFromPaletteOrLibrary(payload, target)

- **Canvas target**: existing `addNode` / `insertBox` at pointer (and optional drop-on-group)
- **Structure target**: insert into highlighted parent/root with new-item placement; then SelectionContext → Live on new id

## Validation Rules

- Cycle: group cannot become its own descendant (FR-014)
- Library inspect: property commits refused (FR-008)
- Live Parameters: existing property validation (ranges, expression/`in`/tiling from 011–012) unchanged
- Audio I/O and other insert refusals from 006/011 remain

## State Diagram (SelectionContext)

```text
        clear / empty click
    ┌──────────────────────────┐
    │                          v
  Live <── canvas/structure click ── None ── library click ──> LibraryInspect
    ^                                                          │
    │              live select                                 │
    └──────────────────────────────────────────────────────────┘
    │
    └── multi-select ──> Multi ── (single select) ──> Live
```
