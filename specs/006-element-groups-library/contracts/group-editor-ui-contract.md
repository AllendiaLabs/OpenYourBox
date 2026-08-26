# Contract: Group Editor UI

## Purpose

Defines in-plugin graph interactions for creating, nesting, editing, expanding, and collapsing groups using Dear ImGui + imgui-node-editor as far as practical.

## Placement

- All interactions inside the existing graph view (`NodeRenderer` / plugin editor window).
- No standalone grouping tool.

## imgui-node-editor usage (required preference)

| Concern | Use |
|---------|-----|
| Expanded group chrome | `ed::BeginNode` + `ed::Group(size)` (+ group style colors); `BeginGroupHint` / `EndGroupHint` when helpful at distance |
| Selection | Existing `GetSelectedNodes` / synchronizeSelection |
| Context menus | `ShowNodeContextMenu` / background menu for Group / Ungroup / Collapse / Expand / Save to Box Library |
| Collapsed group | Compact `BeginNode` representing the group; **not** a third-party widget toolkit |
| Library browser | Standard ImGui list/buttons (see user-box-library-contract) |

## Actions

| Action | Preconditions | Result |
|--------|---------------|--------|
| **Group** | ≥2 selected allowed boxes; no Audio I/O | New `GraphGroup`; members get `parentGroupId`; default name e.g. `Group` |
| **Ungroup** | One group targeted | Members lifted to parent/root; group removed; links kept |
| **Add to group** | Drag/drop or command into group | Membership update if legal |
| **Remove from group** | Drag out or command | Member to parent/root |
| **Rename group** | Group targeted | Name edit (ImGui field / menu) |
| **Set copies (N)** | Group targeted | Update N; materialize/remove independent serial copies when legal (see `group-copies-contract.md`) |
| **Collapse** | Expanded group | `collapsed=true`; hide interior draw; show external pins |
| **Expand** | Collapsed group | `collapsed=false`; show members |
| **Save to Box Library** | Exactly one target box (node or group); not Audio I/O | Opens name dialog → library save (see library contract) |

## Collapsed external ports

- For each link crossing the collapsed group boundary, expose a pin on the compact group node matching the outside-facing endpoint’s type/shape.
- Drawing a new link to a collapsed group pin wires to the underlying member pin (same validation as direct connect).
- Internal links are not drawn while collapsed.

## Persistence

- Group hierarchy, names, membership, bounds, and `collapsed` persist in project/plugin state with the graph.
- Collapse does not change DSP topology.

## Errors (user-visible)

- Audio I/O in selection/move
- Cycle / illegal nest
- Group with insufficient selection on create
- Depth cap exceeded (if enforced)
- Illegal copies N (serial chain not shape-legal)

## Non-Goals

- Auto-layout of entire graphs
- Collapse affecting audio
- Geometry-only membership without durable IDs

## Implementation anchors

- `OpenYourBox/Source/graph/NodeRenderer.cpp`
- `OpenYourBox/Source/graph/NodeGraph.cpp`
- `OpenYourBox/Source/graph/GraphTypes.h`
- imgui-node-editor: `Group`, `SetGroupSize`, `BeginGroupHint`, selection/context APIs
