# Contract: Project Structure & Library Tree

## Purpose

Live **Project structure** browser and expandable **user box library** trees for nested save targeting and whole-or-subpart insert (FR-004, FR-005, FR-005a). Extends `specs/006-element-groups-library/contracts/user-box-library-contract.md`.

## Project structure (left menu)

- Placement: collapsible section **directly under** Library in the left element palette.
- Content: live graph hierarchy (root + nested groups/elements).
- Expand/collapse section independently of Library folder trees.
- **Navigate**: activating a group node focuses that canvas (`setCanvasFocus`); elements may select/focus parent scope as appropriate.
- **Save**: user can save the targeted nested box via existing single-box save rules (Audio I/O excluded; not multi-select).
- Stale ids (deleted/ungrouped) removed from the tree on refresh.

## Library entry tree

- Group entries are expandable to show nested members from the saved snapshot.
- Leaf element entries are non-expandable (or empty children).
- **Sort**: Within each folder, entries MUST appear in alphabetical order by display name (case-insensitive). After rename/save, the list MUST reflect the new order. Nested member rows inside an expanded entry SHOULD sort alphabetically by member label when available. Folder ordering remains alphabetical as today.
- **Insert root**: current behavior (`importBox` full entry, groups collapsed).
- **Insert subpart**: clone only the selected subtree; **do not** recreate cables that left that subtree in the original snapshot; new independent ids.
- DnD payload SHOULD identify `(entryId, optional nestedPathOrId)`; omit nested → root.

## Failure modes

| Case | Behavior |
|------|----------|
| Nested path missing in payload | Error toast; no insert |
| Subpart contains only illegal types for insert | Same refusals as root save/insert |
| Save Audio I/O via tree | Refuse |

## Implementation anchors

- `NodeRenderer::renderPalette` / `UserBoxLibraryPanel`
- `UserBoxLibrary::insertBox` / `NodeGraph::exportBox` / `importBox`
- Snapshot already includes nested structure (006 fidelity)

## Non-goals

- Multi-select save
- Training Library merger
- Restoring expand/collapse UI state from save-time into Project structure
