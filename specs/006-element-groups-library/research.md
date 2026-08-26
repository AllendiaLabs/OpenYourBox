# Phase 0 Research: Element Groups & User Box Library

## Decision 1: Durable hierarchy in `NodeGraph`, imgui `Group` for expanded chrome

**Decision**: Model groups as first-class graph entities with stable IDs, `name`, `parentGroupId` (nullable), `collapsed`, and ordered `memberIds` (nodes and/or nested group IDs). Persist in `GraphDocument` ValueTree. For **expanded** groups, use imgui-node-editor’s `ed::Group` / `BeginGroupHint` / group style colors so the frame moves members together and matches the “use as much ImGui as possible” clarification. Membership for product rules (nesting, save, freeze) comes from the durable IDs, not from hit-testing alone; after layout edits, optionally reconcile positions so members stay inside the group frame.

**Rationale**: Spec FR-001–004, FR-015; imgui provides spatial group chrome but not durable nesting or save semantics. Explore confirmed no project usage of `Group` yet and no collapse API.

**Alternatives considered**:
- Geometry-only grouping via `GetGroupedNodes`: rejected — fails save/reload and nested collapse.
- Fully custom non-ImGui group widgets: rejected — contradicts clarification preference for imgui-node-editor.
- Subgraph replacement nodes only (always collapsed): rejected — blocks expanded editing of interiors.

## Decision 2: Collapse is presentation-only with derived external ports

**Decision**: When `collapsed == true`, `NodeRenderer` skips drawing member nodes/links that are fully inside the collapsed subtree and draws a compact group node (still via node-editor node APIs) showing the group name and **derived external ports**: one pin per link that crosses the group boundary (same type/shape as the underlying member pin). Expanding restores member drawing and prior layout. `LiveGraphPublisher` / audio always compile the **full** flat topology of members regardless of collapse. Project save persists each group’s `collapsed` flag (FR-009). Library **insert** forces root + all nested groups to `collapsed = true` (FR-012a); does not restore save-time expand state into the library payload as authoritative.

**Rationale**: Spec FR-005–008, FR-012a; SC-003. imgui-node-editor cannot hide internals with external pins natively.

**Alternatives considered**:
- Literally remove members from the graph while collapsed: rejected — breaks audio and identity.
- Keep drawing members off-canvas: rejected — still costly and confuses selection/freeze.
- Store expand state in library entries and restore on insert: rejected — clarify Option B.

## Decision 3: Allowed members and I/O gate

**Decision**: Reuse/extend `isFixedIoType` (or equivalent): **Audio Input** and **Audio Output** cannot join a group, be dragged into a group, or be the target of save-to-library. Knob, XY, live processing, Gold frozen nodes, BlackBox nodes, and nested groups are allowed. Attempting group/save with Audio I/O yields a clear ImGui error/toast and no mutation.

**Rationale**: Spec FR-001a and clarifications.

**Alternatives considered**:
- Allow I/O inside groups: rejected by clarify.
- Exclude Gold until unfrozen: rejected by clarify (“any except audio I/O”).

## Decision 4: Freeze = per freezable member (groups keep structure)

**Decision**: When Freeze is invoked on a selection that includes groups and/or multiple nodes, expand the selection to **leaf freezable members** (live weighted/process nodes that today’s freeze accepts). Freeze **each** member with its own freeze job / Gold outcome **in place**. Do **not** replace the selection or group with one Gold BlackBox. Skip already-Gold / non-freezable members; report skips. Group containers remain. This diverges from a literal reading of constitution “selection → one BlackBox” and from train’s `absorbArmedChain` (still one BlackBox). It aligns with current code’s “nodes stay, turn Gold” spirit more than with train absorb, but may require **per-node** artifacts instead of today’s shared chain `artifactPath` when multiple members freeze together—implement per-member freeze requests (see `freeze-per-member-contract.md`).

**Rationale**: Spec FR-001b, SC-008; clarifications.

**Alternatives considered**:
- Keep chain partition → one artifact for whole selection: rejected by clarify.
- Freeze group → one BlackBox inside group: rejected by clarify.
- Disable freeze on groups: rejected — weakens workflow.

## Decision 5: User box library mirrors Training Library storage, separate catalog

**Decision**: Add `UserBoxLibrary` under `OpenYourBox/Source/library/` with `UserDataPaths::boxesDirectory()`. Persist `index.json` plus per-entry folders containing a **box snapshot** (JSON or ValueTree XML fragment: structure, properties, nested groups, link stubs) and copied weight/BlackBox artifacts referenced by relative paths. UI: `UserBoxLibraryPanel` (ImGui list+actions) distinct from the Training **Library** tab (samples)—label clearly e.g. **Boxes** / **Box Library**. Save is a **single-box** context action (element or group). Insert via palette-like DnD or “Place” using a new payload (e.g. `OPENYOURBOX_BOX_LIBRARY_ID`), cloning IDs anew. Overwrite/rename/delete with confirm. Missing element types → failed insert, no partial graph corruption.

**Rationale**: Spec FR-010–014, FR-016; existing `TrainingLibrary` + `TrainingLibraryPanel` patterns; constitution single UI.

**Alternatives considered**:
- Store boxes only inside project presets: rejected — must survive across projects/sessions.
- Embed large tensors in plugin state XML: rejected — bloat/host limits; use files under user data.
- Merge with Training Library panel: rejected — different domain; naming collision risk.

## Decision 6: Snapshot fidelity for parameters and weights

**Decision**: Library snapshot includes all user-facing `properties`, conditioning fields, seeds, freeze/BlackBox metadata needed to restore type and Gold/live state, plus **file copies** of `weightsPath` / `artifactPath` / related TorchScript when present. On insert, allocate new node/group IDs, rewrite internal links, copy artifacts into project/user weights area as needed, and set groups collapsed. If weights were seed-only with no file, store seed + type so re-insert can re-init then match seed policy already used by the live engine.

**Rationale**: Spec assumptions on parameters including weights; explore gap that tensors are not in `GraphDocument`.

**Alternatives considered**:
- Hyperparameters only (no weights): rejected — would not “sound like” saved boxes.
- Always re-randomize on insert: rejected — breaks fidelity.

## Decision 7: Nesting depth and illegal moves

**Decision**: Support at least 5 nesting levels. Refuse moves that create cycles (group into descendant) or that pull Audio I/O into a group. Ungroup lifts members to parent (or root) preserving links. Deleting a group without “delete members” ungroups first.

**Rationale**: Spec edge cases / FR-015.

**Alternatives considered**:
- Unlimited depth with no guard: risk UI/stack issues; soft cap with message is enough.

## Decision 8: Group copies N = independent materialized serial stack

**Decision**: Each `GraphGroup` has integer `copies` N (default 1). For N > 1, **materialize** N independent canvas copies of the group’s block (separate weights/parameters—not shared). Wire copies in a **serial chain** when external I/O shapes allow (outputs of copy i → matching inputs of copy i+1); attach external cables to the first copy’s inputs and last copy’s outputs. If chaining is illegal, refuse/clamp N with a clear message and do not leave orphan nodes. Increasing N clones the last copy into a new trailing instance; decreasing N deletes trailing instances. Nested groups each have their own N. Persist N and all instances in project state and box-library snapshots. On successful N update, re-assert serial wiring between copies.

**Rationale**: Spec FR-017–FR-017e; clarifications (independent copies, canvas materialize, serial-only).

**Alternatives considered**:
- Shared weights across repeats: rejected by clarify.
- Runtime-only unroll with single UI group: rejected (materialize on canvas).
- Parallel or residual stack topologies in v1: deferred.
