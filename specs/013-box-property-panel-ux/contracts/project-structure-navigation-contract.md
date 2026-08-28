# Contract: Project Structure Navigation

## Purpose

Update live **Project structure** interactions from 011: properties on single-click; group double-click opens the inner canvas; leaf double-click still camera-navigates (FR-007, FR-009, FR-010). Extends `specs/011-editor-ux-params/contracts/project-structure-library-tree-contract.md`.

## Single-click

| Row | Behavior |
|-----|----------|
| Element | Select live box; open Parameters (editable); do **not** require opening a nested canvas first if selection can be set by id |
| Group | Select group; open Parameters; do **not** enter the inner canvas |

## Double-click

| Row | Behavior |
|-----|----------|
| Leaf element | Focus containing canvas; center viewport on the element box (unchanged camera-focus behavior) |
| Group | Open the group’s inner canvas; fit and centre the camera on every inner box; do **not** jump to the parent canvas |

## Canvas vs tree

- Canvas double-click on a group box still opens the inner canvas (FR-011) and fits/centres the camera on its contents, independent of Project structure.
- Project structure **double-click** on a group opens the inner canvas and fits/centres the camera. Single-click stays on Parameters only. Non-group double-click remains camera-center only.

## Stale rows

- After undo/redo/delete, refresh tree; missing ids clear selection context.

## Non-goals

- Changing tree placement under Library
- Changing save-to-library context menu rules from 011
