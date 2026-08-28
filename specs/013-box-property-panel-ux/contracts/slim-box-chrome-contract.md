# Contract: Slim Box Chrome

## Purpose

Canvas boxes show **name + pins only**; stable size; one-click drag; group double-click opens (FR-001, FR-005, FR-006, FR-011, FR-016, FR-017).

## On-box chrome

**Allowed on element and group boxes:**
- Display name (non-stealing text; editing via Parameters)
- Input/output pins and pin captions
- Existing Blue/Gold / freeze visual affordances that are not interactive editors

**Removed from box body:**
- Parameter fields, combos, expression rows
- Randomize / Reset buttons
- Other auxiliary controls previously inlined (seed editors, etc. — relocate to Parameters when still required)

## Interaction

- Primary press-drag on box body (not pin): select if needed and move in one gesture (FR-005).
- Hovered pin / create-link takes precedence (FR-006).
- Double-click group box on canvas: `setCanvasFocus(group)` to open inner canvas (FR-011). Removing InputText/button steal is required for reliability.

## Size stability

- Box size MUST NOT grow/shrink because Parameters panel content changed (FR-017).
- Dragging a group MUST NOT change width/height except via an explicit resize affordance if one exists (FR-016).
- Stop content-driven measure feedback from removed widgets; persist slim layout size only.

## Implementation anchors

- `NodeRenderer::renderNode` / `renderGroup`
- Existing drag tracking (`draggingNodeId` / `draggingGroupId`)
- `ed::GetDoubleClickedNode` for groups

## Non-goals

- Redesigning pin layout
- New resize-handle product unless already present
