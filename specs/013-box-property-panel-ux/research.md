# Research: Box Property Panel UX

**Feature**: `013-box-property-panel-ux` | **Date**: 2026-08-28

## R1 — Where do parameters live after the move?

**Decision**: Add a **Parameters** tab to the existing `PluginEditor` right `SideTabs` bar (alongside Info / Library / Capture / Train / Presets). Render parameter controls by relocating the existing ImGui property-row logic from `NodeRenderer::renderNode` / `renderGroup` into a shared panel function. On any live selection (canvas or Project structure), force the active tab to Parameters.

**Rationale**: Spec requires a dedicated Parameters tab and always-switch-on-select. The right menu already hosts inspector-like tabs; Info is analysis-centric and must not absorb all property editors. Reusing `beginPropertyRow` / `propertyDragSteps` / combo handlers preserves edit semantics and undo gestures.

**Alternatives considered**:
- Put parameters only under Info — conflates analysis with editing; conflicts with Analyze-driven Info content.
- Floating inspector on the canvas — more chrome, duplicates side-menu resize work from 011.
- Keep a collapsed “advanced” strip on boxes — violates name+pins-only and reintroduces drag steal / size churn.

## R2 — Why one-click drag fails today; how to fix it

**Decision**: Remove interactive ImGui widgets from the node body (parameters, randomize, group name field if it steals double-click — move name edit to Parameters or a non-stealing display). Keep pins as the only interactive hit targets besides the node drag region. Rely on existing `DragButtonIndex = 0` / `SelectButtonIndex = 0` plus custom drag tracking that already starts on hovered node without hovered pin.

**Rationale**: Exploration showed dense widgets inside `BeginNode` capture the primary button before imgui-node-editor treats the gesture as node drag. Slimming chrome is the structural fix required by FR-001/FR-005.

**Alternatives considered**:
- Invisible full-body drag button under widgets — fragile z-order; still fights InputText.
- Separate “move tool” mode — worse UX than one-click.
- Raise drag threshold only — does not fix first-click activation.

## R3 — Group size glitches

**Decision**: Stop letting parameter/randomize chrome drive `ed::GetNodeSize` → `group.size` / `node.size` growth. Slim group body to name label + pins (+ minimal non-interactive hint if needed). Persist size only for layout of the slim chrome; do not grow box when Parameters panel content changes.

**Rationale**: Size is content-measured each frame after `EndNode`. Randomize button and property rows changing during drag caused FR-016 failures. Moving controls off-box removes the feedback loop.

**Alternatives considered**:
- Freeze `SetNodeSize` during drag — band-aid; edits still resize later.
- Manual fixed sizes per type — brittle across labels/pin counts; pins still need vertical room.

## R4 — Project structure click vs navigate (changes 011)

**Decision**: **Single-click** live row → select that box and open Parameters (editable); do **not** change the focused canvas (groups stay closed). **Double-click** group → `setCanvasFocus(groupId)` so the inner canvas opens. **Double-click** leaf → `setCanvasFocus` to containing canvas + center on box (camera focus unchanged). **Canvas** double-click on group body still opens inner canvas (`setCanvasFocus(group)`).

**Rationale**: Follow-up clarify: opening the group canvas from Project structure is more intuitive than centering the parent view on the group box, but that should be **double-click** so single-click can inspect Parameters without navigating.

**Alternatives considered**:
- Group single-click opens the inner canvas — rejected; Parameters-only on first click.
- Double-click group centers the parent canvas on the outer box — rejected; users expect the inner canvas.
- Double-click leaf opens something other than camera-center — rejected; leave leaf double-click unchanged.

## R5 — Reparent semantics: disconnect then add-like-new

**Decision**: Add `NodeGraph` helpers roughly: `disconnectAllLinksForBox(id)` (remove every link incident to the box’s pins, mirroring `removeNode` link cleanup without deleting the box) then `reparentBox(id, newParentOrNull)` using `removeFromGroup` / `addToGroup` (or root) with **default new-item placement** in the destination scope (same defaults as `addNode` / `adoptNewBox` into that parent). Reject cycles with no mutation. One undoable document gesture.

**Rationale**: User clarification: not sibling-index DnD; behave like adding a new item after disconnecting. Existing `addToGroup` keeps cables — unsafe across canvases.

**Alternatives considered**:
- Keep legal cables only — complex shape rules; not what the user asked.
- Delete and re-import via `exportBox`/`importBox` — new ids, breaks selection continuity and is heavier than needed.

## R6 — Library / element-list → Project structure

**Decision**: Accept existing payloads (`OPENYOURBOX_NODE_TYPE`, `OPENYOURBOX_BOX_LIBRARY_ID`) on Project structure drop targets. On drop: insert via `addNode` or `UserBoxLibrary::insertBox` into the highlighted parent (or root), using new-item placement for that scope; then select result, force Parameters, focus destination canvas when needed. Canvas drops keep today’s pointer position behavior.

**Rationale**: Spec FR-015/015a/015b. Reuses proven insert paths; only the drop target surface is new.

**Alternatives considered**:
- Structure-only insert API — duplication.
- Force focus group before canvas drop — fails when targeting a non-focused group from the tree.

## R7 — Library click read-only inspect

**Decision**: Introduce an editor **selection context**: `LiveBox` (editable Parameters) vs `LibraryInspect` (read-only Parameters bound to snapshot entry/subpart). Library panel click sets inspect context and forces Parameters tab even if a canvas box is already selected (clear leftover editor selection so inspect is not overwritten the same frame). Commits no-op / controls disabled. Clearing inspect when the user next selects a live canvas box or clicks the canvas background.

**Rationale**: Clarified FR-008. Avoids writing catalog JSON from the inspector.

**Alternatives considered**:
- Editable library properties — rejected in clarify.
- Separate “Library preview” tab — extra chrome; Parameters tab already required.

## R8 — Post-drop focus

**Decision**: After successful structure reparent or structure insert: select the box, force Parameters, call `setCanvasFocus` for the destination scope’s canvas when different, optionally center on the new box.

**Rationale**: Clarified Q4; matches click→Parameters discoverability.

**Alternatives considered**: Leave focus unchanged — user loses the dropped item in nested graphs.
