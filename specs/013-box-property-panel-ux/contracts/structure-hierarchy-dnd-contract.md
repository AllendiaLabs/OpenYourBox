# Contract: Structure Hierarchy Drag-and-Drop

## Purpose

Project structure as drop surface for live reparent and for palette/library insert (FR-012–FR-015b, FR-013, FR-014).

## Live reparent (structure → structure)

1. Drag a live element or group row.
2. Highlight eligible targets (project root and groups that would not create a cycle) with a **rectangle around label + nested content rows**.
3. On valid drop onto target parent (or root):
   - Disconnect **all** cables incident to the moved box
   - Reparent using the same membership + **default placement** rules as inserting a new item into that scope
   - Select the moved box; force Parameters; focus destination canvas if different
4. On invalid drop (cycle, etc.): no hierarchy change, no cable change; clear highlight.

## Insert from element list / user library → structure

| Payload | On structure drop |
|---------|-------------------|
| `OPENYOURBOX_NODE_TYPE` | `addNode` (or equivalent) into highlighted parent/root with new-item placement |
| `OPENYOURBOX_BOX_LIBRARY_ID` | `insertBox` / `importBox` into that scope (whole root or nested subpart per payload) |

After success: select new instance; Parameters; focus destination canvas if needed.

## Insert → canvas (unchanged semantics)

- Library and palette drops on the canvas keep pointer-position insert and existing drop-on-group-box parenting.
- Cancel when release is outside canvas and outside a valid structure target.

## Undo

- Reparent and inserts participate in existing edit history / patch gestures as one user action each.

## Failure modes

| Case | Behavior |
|------|----------|
| Cycle / illegal parent | Reject; toast optional; no mutation |
| Missing library entry / nested path | Same refusals as canvas insert |
| Drop on self as parent | Reject |

## Implementation anchors

- `NodeRenderer::renderProjectStructureItem` drop targets + highlight
- `NodeGraph` disconnect-all + reparent helpers
- `UserBoxLibrary::insertBox`, palette `addNode`, `adoptNewBox` patterns
- Existing payloads in `UserBoxLibraryPanel` / `renderPalette`

## Non-goals

- Sibling-index “insert between rows” API (placement follows new-item defaults)
- Preserving cables across reparent
- Library folder DnD changes (catalog folder move stays as today)
