# Contract: Project Structure Navigation

## Purpose

Update live **Project structure** interactions from 011: properties on click; camera navigate on double-click without opening groups from the tree (FR-007, FR-009, FR-010). Extends `specs/011-editor-ux-params/contracts/project-structure-library-tree-contract.md`.

## Single-click

| Row | Behavior |
|-----|----------|
| Element | Select live box; open Parameters (editable); do **not** require opening a nested canvas first if selection can be set by id |
| Group | Select group box; open Parameters; do **not** enter inner canvas |

Supersedes 011 “activating a group node focuses that canvas” for **single-click**.

## Double-click

| Row | Behavior |
|-----|----------|
| Leaf element | Focus containing canvas; center viewport on the element box |
| Group | Focus **parent** canvas (or root); center on the group’s **outer** box; do **not** call `setCanvasFocus(groupId)` |

## Canvas vs tree

- Opening a group’s inner canvas remains **canvas double-click** (and existing context Open), not Project structure double-click.

## Stale rows

- After undo/redo/delete, refresh tree; missing ids clear selection context.

## Non-goals

- Changing tree placement under Library
- Changing save-to-library context menu rules from 011
