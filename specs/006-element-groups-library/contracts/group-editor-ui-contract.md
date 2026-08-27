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
- Groups on their parent's canvas are compact boxes exposing the ordered lanes declared by their internal Group Input and Group Output hubs; interiors are never drawn in place.
- The overview map shows only boxes on the focused canvas (members and nested group boxes).

## imgui-node-editor usage (required preference)

| Concern | Use |
|---------|-----|
| Group box | Compact `ed::BeginNode` with I/O pins declared by boundary hubs; **not** nested `Group()` viewports |
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
| **Set interface lanes** | Group Input or Group Output hub targeted | Resize its paired lanes (minimum one); refuse shrinking across a connected lane |
| **Set copies (N)** | Group targeted | Persist requested N; effective runtime copies is N when legal or 1 with actionable diagnostics when inactive (see `group-copies-contract.md`) |
| **Open** | Group box on the current canvas | Focus that group's interior; breadcrumb updates |
| **Back** | Breadcrumb ancestor | Restore that canvas (root or parent group) |
| **Save to Box Library** | Exactly one target box (node or group); not Audio I/O | Opens name dialog → library save (see library contract) |

## Group boundary hubs and external ports

- Every group interior contains exactly one **Group Input** hub and one **Group Output** hub.
- Hubs are editor-only structural nodes and cannot be deleted, removed from their group, nested independently, frozen, analyzed, or saved/inserted as standalone boxes. Ungrouping their owner splices each connected lane through and removes the hubs.
- Each hub has a variable positive lane count controlled by its Inputs/Outputs property. Each lane is a paired pass-through whose shape follows its incoming connection.
- The parent-canvas group box exposes Group Input lanes as its input pins and Group Output lanes as its output pins, preserving lane order and shape.
- Drawing a new link to a group pin wires to the corresponding hub lane (same validation as direct connect).
- Internal links are drawn only on the group's own canvas.
- Unconnected member pins do not create group ports; the interface changes only through explicit hub lane edits.

## Persistence

- Group hierarchy, names, membership, bounds, focused-canvas id, both boundary hubs, and their lane counts persist in project/plugin state with the graph.
- Loading a legacy group without hubs creates both hubs and converts every previously inferred interface pin into an explicit lane while preserving existing external links. Box-library import performs the same migration.
- Opening a group does not change DSP topology.

## Errors (user-visible)

- Audio I/O in selection/move
- Cycle / illegal nest
- Group with insufficient selection on create
- Depth cap exceeded (if enforced)
- Boundary lane shrink would remove a connected lane
- Inactive requested copies N: mismatched lane counts, missing Group Input-to-Group Output path, or incompatible paired shapes
- Shape-blocked copies identify the lane shapes and highlight candidate shape-driving properties with repair guidance; diagnostics never change model parameters automatically

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
