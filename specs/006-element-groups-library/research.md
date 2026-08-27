# Phase 0 Research: Element Groups & User Box Library

## Decision 1: Durable hierarchy in `NodeGraph`, imgui `Group` for expanded chrome

**Decision**: Model groups as first-class graph entities with stable IDs, `name`, `parentGroupId` (nullable), `collapsed`, and ordered `memberIds` (nodes and/or nested group IDs). Persist in `GraphDocument` ValueTree. For **expanded** groups, use imgui-node-editor’s `ed::Group` / `BeginGroupHint` / group style colors so the frame moves members together and matches the “use as much ImGui as possible” clarification. Membership for product rules (nesting, save, freeze) comes from the durable IDs, not from hit-testing alone; after layout edits, optionally reconcile positions so members stay inside the group frame.

**Rationale**: Spec FR-001–004, FR-015; imgui provides spatial group chrome but not durable nesting or save semantics. Explore confirmed no project usage of `Group` yet and no collapse API.

**Alternatives considered**:
- Geometry-only grouping via `GetGroupedNodes`: rejected — fails save/reload and nested collapse.
- Fully custom non-ImGui group widgets: rejected — contradicts clarification preference for imgui-node-editor.
- Subgraph replacement nodes only (always collapsed): rejected — blocks expanded editing of interiors.

## Decision 2: Collapse is presentation-only with explicit external ports

**Decision**: Each group contains one mandatory editor-only **Group Input** hub and one mandatory editor-only **Group Output** hub. Their positive, variable lane counts explicitly declare the ordered external interface; ports are no longer inferred from dangling member pins. The hubs cannot be deleted, moved out of their owner, frozen, analyzed, or saved/inserted independently. Shrinking a hub is refused if it would remove a connected lane. When `collapsed == true`, `NodeRenderer` skips drawing member nodes/links and draws a compact group node exposing these declared lanes. `LiveGraphPublisher` / audio compiles the full topology after boundary hubs are flattened into direct connections. Project save persists each group’s `collapsed` flag (FR-009). Library **insert** forces root + all nested groups to `collapsed = true` (FR-012a).

**Rationale**: Explicit hubs make the interface intentional and stable when internal wiring changes, while retaining the one-canvas-at-a-time editor model. Spec FR-004a–004c, FR-005–008, FR-012a; SC-003.

**Alternatives considered**:
- Derive group ports continuously from unconnected/dangling member pins: rejected — accidental wiring edits silently redefine the public interface and make copy-chain validity unstable.
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

## Decision 8: Persist requested copies; derive safe runtime copies

**Decision**: Each `GraphGroup::copies` stores the user-authored/requested N (default 1). `groupCopyStatus` derives an effective runtime count: requested N only becomes effective when Group Input and Group Output declare the same nonzero lane count, a directed path exists from the input hub to the output hub, and each serial copy can feed the next after applying that copy’s properties (per-copy channel/feature/stride lists and `in` bindings). First-copy output need not match first-copy input, so RAVE-style encoder/decoder stacks stay active. Residual joins that cannot combine at a copy keep N inactive. Otherwise effective runtime copies is 1 while requested N remains unchanged and persisted. The live engine unrolls the effective count into independent serial copies (separate weights/parameters—not shared); the editor does not draw clones, and each member retains per-copy values/artifacts. Increasing requested N clones the last slot and decreasing it drops trailing slots. Nested effective counts multiply. Because validity is recomputed from current graph state, a stored request reactivates automatically as soon as the user makes the interface, path, and shapes legal.

**Rationale**: Preserving intent avoids losing an architecture setting during temporary edits, while clamping only runtime expansion prevents invalid graph publication. Spec FR-017–FR-017h plus the product requirement that copies not appear as extra canvas boxes. Independent weights remain required for training.

**Alternatives considered**:
- Reject the N edit or overwrite N with 1 when currently invalid: rejected — loses user intent and prevents automatic reactivation after repair.
- Shared weights across repeats: rejected by clarify.
- Materialize copies as extra visible nodes: rejected — clutters the graph and duplicates UI.
- Parallel or residual stack topologies in v1: deferred.

## Decision 9: Actionable copy diagnostics, never automatic model edits

**Decision**: An inactive request reports requested N, effective runtime count 1, and the specific interface/path/shape failure. Shape failures include paired lane labels and shape values. Reverse traversal from the affected Group Output lane identifies candidate shape-driving properties (for example `channels`, `features`, `latent_size`, `stride`, and `n_band`); the property UI highlights them and suggests `in` preservation where supported. Validation and diagnostics never rewrite model properties, links, or lane counts. The user chooses and performs every repair.

**Rationale**: A safe clamp without repair guidance is difficult to resolve, but automatic parameter tuning could change model architecture or sound without consent. Explicit hints preserve user control and allow the already stored N to reactivate naturally.

**Alternatives considered**:
- Automatically force output dimensions/rates to match the group input: rejected — ambiguous in branched graphs and violates authored model intent.
- Show only a generic “cannot chain” warning: rejected — does not identify the lane or properties the user can inspect.

## Decision 10: Migrate legacy inferred interfaces on load/import

**Decision**: Graph document version 3 persists the boundary hub node types. During project restore and box-library import, process groups deepest-first. Any group without boundary hubs receives both hubs; each prior inferred interface pin becomes a corresponding lane, external links are redirected to the hub’s outside pin, and an internal link joins the hub to the former member pin. New groups seed hubs from cables crossing the newly created boundary. Every hub has at least one lane even when no interface was previously inferred.

**Rationale**: Existing projects and library entries must retain connectivity while moving from incidental dangling-pin inference to explicit interfaces. Deepest-first conversion keeps nested boundaries coherent.

**Alternatives considered**:
- Require users to redraw legacy group interfaces: rejected — destructive migration and poor project compatibility.
- Keep a permanent mixed inferred/explicit mode: rejected — creates two interface semantics and inconsistent copy validation.
