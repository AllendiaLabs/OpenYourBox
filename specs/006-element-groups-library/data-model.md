# Data Model: Element Groups & User Box Library

## Entities

### GraphGroup

Named hierarchical container on the canvas.

| Field | Type / notes |
|-------|----------------|
| `id` | Stable string/int ID unique within the graph document |
| `name` | User-visible label (editable) |
| `parentGroupId` | Nullable; null = root-level group |
| `memberIds` | Ordered list of child IDs: each is a `GraphNode.id` or nested `GraphGroup.id` |
| `collapsed` | Bool; presentation only |
| `copies` | Integer N ≥ 1 (default 1); number of independent materialized block instances |
| `copyInstanceIds` | When N &gt; 1, ordered IDs of materialized copy roots/subtrees managed by this repeat (implementation may nest each copy as a child group or tagged member set) |
| `position` / `size` | Layout for expanded imgui `Group` frame and collapsed compact node |
| `createdAt` / `updatedAt` | Optional; useful for debugging |

**Rules**:
- ≥2 members to **create** from selection; afterward membership may temporarily drop to 1 until user adds more (or ungroup)—product may allow 1 member after edits; creation gate stays ≥2.
- No Audio Input / Audio Output in `memberIds` (transitively when moving).
- No cycles: `parentGroupId` chain must not include `id`.
- Nesting depth ≥5 supported; optional hard cap with clear refusal.
- Collapse does not remove members from the model.
- `copies` N default 1. N &gt; 1 requires shape-legal **serial** chaining of the block’s external I/O; otherwise refuse/clamp.
- Materialized copies are independent (own parameters/weights). New copies clone the last. Nested groups each have their own `copies`.

### GroupCopyInstance

Logical identity for one of N independent replicas when `copies` &gt; 1.

| Field | Notes |
|-------|--------|
| `index` | 0 .. N-1 in serial order |
| `rootMemberIds` | Nodes/subgroups belonging to this copy |
| Owned weights/params | Independent of sibling copies |

### GraphNode (extensions)

Existing node entity gains:

| Field | Notes |
|-------|--------|
| `parentGroupId` | Nullable; which group owns this node (redundant with group.memberIds but convenient for queries—keep consistent) |

Unchanged: type, state (liveBlue / frozenGold), pins, properties, seeds, artifact/weights paths, etc.

### GraphLink

Unchanged structurally. **Boundary links**: any link with one endpoint inside a group subtree and one outside is an **external** link; used to derive collapsed-group ports.

### UserBoxLibrary

Per-user catalog (not part of project `GraphDocument`).

| Field | Notes |
|-------|--------|
| `rootPath` | `UserDataPaths::boxesDirectory()` |
| `entries` | List of `UserBoxLibraryEntry` from `index.json` |

### UserBoxLibraryEntry

| Field | Notes |
|-------|--------|
| `id` | Stable UUID for the catalog row |
| `name` | Unique display name within library (overwrite policy by name) |
| `kind` | `element` \| `group` |
| `createdAt` / `updatedAt` | Timestamps |
| `payloadRelativePath` | Folder under boxes root containing snapshot + artifacts |
| `rootTypeHint` | Optional: element `NodeType` or `"group"` for list UI |
| `schemaVersion` | Integer for forward compatibility |

**Payload (inside entry folder)**:
- `box.json` (or `.xml`): recursive snapshot of one element or one group tree (nodes, nested groups, internal links, properties, seeds, relative artifact filenames).
- `artifacts/`: copied weight / `.pt` / BlackBox files referenced by the snapshot.
- Expand/collapse flags in payload are **ignored on insert**; placer sets all groups collapsed.

## Relationships

```text
UserBoxLibrary 1──* UserBoxLibraryEntry
GraphDocument 1──* GraphNode
GraphDocument 1──* GraphGroup
GraphDocument 1──* GraphLink
GraphGroup *──* (members) GraphNode | GraphGroup   # via memberIds / parentGroupId
```

Library entries are **independent** of any open `GraphDocument`. Insert **clones** into the current document with new IDs.

## Validation Rules

| Rule | On failure |
|------|------------|
| Create group from selection with &lt;2 allowed members | Disable / refuse |
| Selection or move includes Audio I/O | Refuse with message |
| Move group into descendant | Refuse (cycle) |
| Set copies N &gt; 1 when serial I/O chain illegal | Refuse / clamp with message |
| Save to library without single box target | Refuse / disable |
| Save target is Audio I/O | Refuse |
| Insert unknown `NodeType` / schema | Refuse insert; graph unchanged |
| Overwrite existing library name | Require confirm |
| Delete library entry | Require confirm; delete folder |

## State Transitions

### Group presentation

```text
[Expanded] --user collapse--> [Collapsed]
[Collapsed] --user expand--> [Expanded]
[Library insert] --> [Collapsed] (root + nested)
```

Audio topology: unchanged across transitions.

### Group copies (N)

```text
[N=1 single block] --set N=K legal--> [K independent serial copies]
[K copies] --set N=K+1--> [clone last → append]
[K copies] --set N=J<K--> [drop trailing copies]
[any] --set N illegal--> [unchanged + error]
```

Audio/DSP sees the full materialized topology (all copies), same as any other nodes.

### Freeze (selection including groups)

```text
For each freezable leaf member in selection:
  [liveBlue] --freeze success--> [frozenGold] (own artifact as per freeze pipeline)
Non-freezable / already Gold: skipped
Group entity: unchanged membership
```

Train absorb (existing): still may replace an armed chain with one `blackBox` — out of scope to redesign here; do not use absorb semantics for Freeze menu.

### Library entry lifecycle

```text
[Absent] --save box--> [Active entry]
[Active] --rename--> [Active]
[Active] --overwrite confirm--> [Active] (replaced payload)
[Active] --delete confirm--> [Absent]
[Active] --insert--> clones into GraphDocument (entry unchanged)
```

## Serialization

### Project (`GraphDocument` ValueTree)

Add `Groups` child collection; each `Group` with id, name, parent, collapsed, **copies**, bounds, member id list, and copy-instance mapping. Nodes store `parentGroupId`. Version bump `GraphDocument` version field if present.

### Library (`index.json` + folders)

Human-debuggable JSON index; binary/model files only under `artifacts/`. Not written into host preset blob except by reference after insert (project then owns copied artifact paths like today).
