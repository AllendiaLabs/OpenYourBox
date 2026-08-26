# Contract: Group Editor UI

## Purpose

Defines in-plugin graph interactions for creating, nesting, editing, and opening groups using Dear ImGui + imgui-node-editor as far as practical.

## Placement

- All interactions inside the existing graph view (`NodeRenderer` / plugin editor window).
- No standalone grouping tool.

## Canvas model

- One editor canvas is visible at a time: the graph root, or the interior of exactly one focused group.
- Nested structure is shown as a breadcrumb above the canvas (`Graph > Parent > Subgroup`). Clicking an ancestor opens that canvas.
- Double-clicking a group box opens that group as the focused canvas and hides sibling/parent boxes.
- Groups on their parent's canvas are compact boxes with derived I/O pins; interiors are never drawn in place.
- The overview map shows only boxes on the focused canvas (members and nested group boxes).

## imgui-node-editor usage (required preference)

| Concern | Use |
|---------|-----|
| Group box | Compact `ed::BeginNode` with mediating I/O pins; **not** nested `Group()` viewports |
| Selection | Existing `GetSelectedNodes` / synchronizeSelection |
| Context menus | `ShowNodeContextMenu` / background menu for Group / Ungroup / Open / Save to Box Library |
| Library browser | Standard ImGui list/buttons (see user-box-library-contract) |

## Actions

| Action | Preconditions | Result |
|--------|---------------|--------|
| **Group** | ≥2 selected allowed boxes; no Audio I/O | New `GraphGroup`; members get `parentGroupId`; default name e.g. `Group` |
| **Ungroup** | One group targeted | Members lifted to parent/root; group removed; links kept |
| **Add to group** | Drag/drop or command onto a group box | Membership update if legal |
| **Remove from group** | Command | Member to parent/root |
| **Rename group** | Group targeted | Name edit (ImGui field / menu) |
| **Set copies (N)** | Group targeted | Update N; materialize/remove independent serial copies when legal (see `group-copies-contract.md`) |
| **Open** | Group box on the current canvas | Focus that group's interior; breadcrumb updates |
| **Back** | Breadcrumb ancestor | Restore that canvas (root or parent group) |
| **Save to Box Library** | Exactly one target box (node or group); not Audio I/O | Opens name dialog → library save (see library contract) |

## Group box external ports

- For each pin that is not wired internally inside the group, expose a pin on the group box matching the outside-facing endpoint’s type/shape.
- Drawing a new link to a group pin wires to the underlying member pin (same validation as direct connect).
- Internal links are drawn only on the group's own canvas.

## Persistence

- Group hierarchy, names, membership, bounds, and focused-canvas id persist in project/plugin state with the graph.
- Opening a group does not change DSP topology.

## Errors (user-visible)

- Audio I/O in selection/move
- Cycle / illegal nest
- Group with insufficient selection on create
- Depth cap exceeded (if enforced)
- Illegal copies N (serial chain not shape-legal)

## Non-Goals

- Auto-layout of entire graphs
- Opening a group affecting audio
- Geometry-only membership without durable IDs
- Nested in-place expanded frames / progressive clip of member chrome

## Implementation anchors

- `OpenYourBox/Source/graph/NodeRenderer.cpp`
- `OpenYourBox/Source/graph/NodeGraph.cpp`
- `OpenYourBox/Source/graph/GraphTypes.h`
- imgui-node-editor: selection/context APIs
