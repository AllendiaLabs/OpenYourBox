# Data Model: Editor UX & Parameter Flexibility

**Feature**: `011-editor-ux-params` | **Date**: 2026-08-27  
**Extends**: `specs/006-element-groups-library/data-model.md`

## Entities

### AuthoredCopyList (extends `NodeProperty`)

| Field | Type | Notes |
|-------|------|--------|
| `copyIntValues` / `copyFloatValues` | vector | **Authored** length L (may be &lt; P) |
| `value` / `realValue` | scalar | Scalar when L=1 or non-copy mode |
| Derived expanded | vector length P | Not separately persisted; computed for runtime + preview |
| `copyListInvalid` | bool | Set when L ∉ dividing set after nest change |

**Relationships**: Owned by `Node` under nested `GraphGroup` chain. P = `effectiveCopyCount(node)`.

**Validation**:
- On commit: L ∈ dividing set(ancestor copies) or refuse.
- On `GraphGroup.copies` change: if L still valid → keep authored, recompute expanded; else `copyListInvalid=true`, runtime treats as unresolved.
- Dividing set for copy vector `[c0,c1,…,ck-1]` (outer→inner): `{1} ∪ {Π_{j=i}^{k-1} c_j for i=0..k-1}` (includes P).

### PreserveInBinding

| Field | Type | Notes |
|-------|------|--------|
| Token | `in` | Case-sensitive lowercase |
| Scope | per property slot or per authored list entry | All entries must be `in` if list form |
| Resolved value | int (channels/features) | From paired input pin shape for that copy slot |

**Relationships**: Applies to shape-driving integer properties (`features`, `channels`, and other bindable dims documented in contract). Cleared when user types a number.

**Validation**: Refuse mix of `in` and numbers; refuse on non-bindable fields; unresolved input → illegal like other shape failures.

### ProjectStructureNode (UI projection)

| Field | Type | Notes |
|-------|------|--------|
| `id` | NodeId or GroupId | Live graph identity |
| `kind` | Element \| Group | |
| `children` | list | Group members / nested groups |
| `expanded` | bool | UI tree state (session) |

**Relationships**: Projection of current `NodeGraph` (root canvas + groups). Not a separate persistence store.

### LibraryTreeNode (UI + insert target)

| Field | Type | Notes |
|-------|------|--------|
| `entryId` | string | Catalog entry |
| `payloadNodeId` / path | id or path in snapshot | Root or nested member |
| `kind` | Element \| Group | From snapshot |
| `children` | list | Nested members in saved group |

**Relationships**: Derived from `UserBoxLibraryEntry` payload. Insert of non-root clones that subtree only.

### HierarchyTrailEntry

| Field | Type | Notes |
|-------|------|--------|
| `groupId` | GroupId | Visited descendant under current focus |
| `order` | int | Spine order parent→…→deepest |

**Relationships**: Stored on `GraphViewport` (or renderer session state) alongside `focusedGroupId`. Cleared when user opens a sibling branch not on the spine; pruned when group deleted/ungrouped.

### SideMenuLayout

| Field | Type | Notes |
|-------|------|--------|
| `leftWidthPx` | float | Default 200; min/max clamped |
| `rightWidthPx` | float | Default ~332; min/max clamped |
| `minCanvasWidthPx` | float | Enforce when dragging |

**Relationships**: UI session / optional prefs; independent of graph document.

### LeakyReluNegativeSlope

| Field | Type | Notes |
|-------|------|--------|
| Property key | `negative_slope` (or agreed name) | `PropertyKind::real` |
| Default | `0.01` | |
| Range | `[0, 1]` | Refuse outside; no clamp |
| Applicability | Activation node when choice = LeakyReLU | |

**Relationships**: On activation `Node`; consumed by live DSP, TCN path, freeze/train workers.

## State transitions

### Copy list

```text
[empty/default] --commit valid L--> [authored L, expanded P, valid]
[authored L] --ancestor N change, L still ok--> [same L, new P, re-tiled]
[authored L] --ancestor N change, L invalid--> [authored L preserved, invalid]
[invalid] --user commits new valid L--> [valid]
[invalid] --user commits bad L--> refuse, stay invalid
```

### Hierarchy trail

```text
[focus F, spine S] --open child C--> [focus C, spine S′ extended]
[focus C, spine S′] --navigate to ancestor A--> [focus A, spine keeps descendants under A]
[focus A, sticky children] --open sibling B not in spine--> [focus B, spine reset under A→B]
[any] --delete/ungroup sticky id--> prune from spine
```

### `in` binding

```text
[numeric] --set "in"--> [bound, resolve on shape refresh]
[bound] --input shape change--> [re-resolve]
[bound] --set number--> [numeric]
[bound] --unsupported field--> refuse
```

## Serialization notes

- Persist authored copy vectors (current CSV fields) **without** forcing resize to P on load; after load, validate L vs current nest and set invalid or expand.
- Persist `in` as token in property value representation (string form or dedicated flag + sentinel); never persist only resolved ints if binding must survive input changes.
- Trail + panel widths: session (and prefs if available); not required in graph file for v1.
- Library: no mandatory index.json schema bump if nested insert addresses nodes inside existing box snapshot by stable saved ids/paths.
