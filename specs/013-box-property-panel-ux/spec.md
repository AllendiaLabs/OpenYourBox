# Feature Specification: Box Property Panel UX

**Feature Branch**: `013-box-property-panel-ux`

**Created**: 2026-08-28

**Status**: Draft

**Input**: User description: "Move parameters from boxes to a new tab in the right menu; simplify boxes to show only name and pins; selecting or clicking a box opens its property panel; fix one-click select-and-drag on boxes; fix group box movement glitches and move randomize control to the property panel; restore double-click on group boxes to open them; from user library or project structure, single-click opens the property panel; from project structure, double-click navigates the camera to the element (groups go to their box on the parent canvas without opening); from project structure, drag to reorganize hierarchy with drop-target highlighting (disconnect then add like a new item); from user library or element list, drag to the canvas or to Project structure to add to the project."

## Clarifications

### Session 2026-08-28

- Q: When the user single-clicks a user library entry (or a nested subpart), should the right property panel let them edit the saved catalog item, or only inspect it? → A: Read-only — the panel shows saved properties but does not write back to the library; live edits happen on canvas instances
- Q: When the user selects or clicks a canvas box, should the right menu automatically switch to the Parameters tab, or stay on whichever tab they already had open? → A: Always switch to the Parameters tab when a box is selected or clicked
- Q: When the user drops a dragged row onto a highlighted group in Project structure, where should the item be placed, and can library/element-list items drop on Project structure too? → A: Disconnect any existing cables, then add the item into the target group (or project root) using the same placement rules as inserting a new item into the project; users may also drag from the user library or the element list onto Project structure (not only onto the canvas) to add items the same way
- Q: After a successful drop onto Project structure (reparent or insert from library/element list), should the editor select that item and open its Parameters panel? → A: Select the dropped item and open Parameters (and focus its canvas if the destination scope differs)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Simplified boxes with property panel editing (Priority: P1)

A designer works on a dense graph. Each box on the canvas shows only its name and connection pins. Parameters, randomize controls, and other configuration that previously lived on the box body are moved to a dedicated **Parameters** tab in the right-side property menu. When the designer selects or clicks any box on the canvas, the right menu shows that box’s property panel and **always switches to the Parameters tab**, even if another tab was previously active.

**Why this priority**: Cluttered boxes are the root UX problem; centralizing editing in the property panel unlocks cleaner layouts and fixes related sizing glitches.

**Independent Test**: Place several element types on the canvas; confirm boxes render name and pins only; click each box and edit parameters exclusively through the right menu Parameters tab; confirm values persist and affect graph behavior.

**Acceptance Scenarios**:

1. **Given** a box with configurable parameters on the canvas, **When** the user views the box, **Then** the box body shows only its display name and input/output pins (no inline parameter fields, randomize button, or other controls on the box)
2. **Given** the right property menu, **When** the user inspects it, **Then** a **Parameters** tab is available for the selected box and lists all parameters previously shown on the box
3. **Given** no box is selected, **When** the user clicks a box on the canvas, **Then** that box becomes selected and the right menu shows its property panel on the Parameters tab
4. **Given** a box is already selected, **When** the user clicks a different box, **Then** the property panel updates to the newly clicked box and the Parameters tab is shown
5. **Given** the right menu is open on a non-Parameters tab, **When** the user selects or clicks a box, **Then** the menu switches to the Parameters tab for that box
6. **Given** a box whose type supports randomize, **When** the user opens its property panel, **Then** randomize is available in the property menu (not on the box body)
7. **Given** the user edits a parameter in the Parameters tab, **When** they commit the change, **Then** the box behavior updates and the canvas box appearance does not resize or reflow because of parameter UI

---

### User Story 2 - One-click select and drag boxes (Priority: P1)

A designer repositions boxes frequently. A single press-and-hold on a box immediately selects it and allows dragging without requiring multiple clicks to “activate” the box first.

**Why this priority**: Repeated clicks to move boxes block basic editing flow and were called out as a top annoyance.

**Independent Test**: Click once on an unselected box and drag without releasing; the box moves on the first attempt. Repeat for selected, grouped, and nested-canvas contexts.

**Acceptance Scenarios**:

1. **Given** an unselected box on the canvas, **When** the user presses the primary button on the box body and moves the pointer before release, **Then** the box is selected and follows the drag in one continuous gesture
2. **Given** a box already selected, **When** the user press-drags on its body, **Then** it moves immediately without an extra activation click
3. **Given** the user press-drags on a pin or connection handle (not the box body), **When** the gesture is recognized as wiring, **Then** box drag does not steal the interaction
4. **Given** multiple overlapping boxes, **When** the user clicks the topmost box once and drags, **Then** only that box moves

---

### User Story 3 - Project structure tree integration (Priority: P2)

A designer uses the **Project structure** tree (left menu, under Library) to inspect and reorganize the graph. Single-clicking any element or group row opens that item’s property panel in the right menu. Double-clicking navigates the view: for a leaf element, the camera centers on that box on the appropriate canvas; for a group, the camera centers on the group’s box on the **parent** canvas without opening the group’s inner canvas. Dragging a live row within the tree moves it to another group or to the project root: any existing cables on that item are **disconnected**, then the item is added to the drop target using the **same placement rules as inserting a new item** into that scope (not a special sibling-order insert). During drag, the current drop target group/folder (or root) is highlighted by drawing a rectangle around its name and contained rows.

**Why this priority**: Project structure was added for hierarchy navigation; tying it to properties, camera focus, and drag-reparent completes its role as a primary navigation surface.

**Independent Test**: Build a nested graph with a wired element; use Project structure alone to open properties, jump to elements, and drag that wired element onto a sibling group; confirm cables are gone, the element appears in the target like a newly added item, and drop-target highlighting worked during the drag.

**Acceptance Scenarios**:

1. **Given** Project structure is expanded, **When** the user single-clicks an element row, **Then** the right property panel shows that element’s properties on the Parameters tab
2. **Given** Project structure is expanded, **When** the user single-clicks a group row, **Then** the right property panel shows that group’s properties on the Parameters tab
3. **Given** a leaf element row in Project structure, **When** the user double-clicks it, **Then** the editor navigates to the canvas containing that element and centers the view on the element’s box
4. **Given** a group row in Project structure, **When** the user double-clicks it, **Then** the editor navigates to the parent canvas and centers the view on the group’s outer box without entering the group’s inner canvas
5. **Given** a draggable element or group row, **When** the user starts dragging it within Project structure, **Then** valid drop targets (project root and eligible groups/folders) are visually indicated
6. **Given** a drag over a target group/folder or root, **When** the pointer is inside that target, **Then** a rectangle highlight wraps the target’s label and its nested rows to show where the item will land
7. **Given** a connected element dragged onto a valid group (or root) target, **When** the user releases, **Then** all of that element’s cables are disconnected, the element is added into the target scope using the same placement rules as inserting a new item there, the element becomes selected, the Parameters tab opens for it, and if the destination canvas differs the editor focuses that canvas
8. **Given** an unconnected element dragged onto a valid target, **When** the user releases, **Then** the element is added into the target scope using the same placement rules as inserting a new item, becomes selected with Parameters open, and the destination canvas is focused when it differs from the current one
9. **Given** an invalid drop (e.g., dropping a group into its own descendant), **When** the user releases, **Then** the drag is cancelled with no hierarchy change, no disconnection, and prior selection/state is preserved

---

### User Story 4 - Library and element-list insert via canvas or Project structure (Priority: P2)

A designer browses saved entries in the user box library and built-in items in the element list. Single-clicking a library entry (root or nested subpart when the entry is expanded) opens a **read-only** property panel for that saved item in the right menu so they can inspect parameters without changing the catalog. Live edits apply only after the item is inserted as a project instance. The designer can add an item by dragging from the **user library** or the **element list** onto either the **canvas** or a highlighted row in **Project structure**. Canvas drop places the instance at the pointer using existing insert rules; Project structure drop adds the instance into the highlighted group or project root using the **same placement rules as adding a new item** to that scope.

**Why this priority**: Library, palette, canvas, and Project structure should share one insert model so hierarchy placement does not require a canvas-only detour.

**Independent Test**: Drag a library entry and an element-list item onto the canvas and onto a group row in Project structure; confirm both destinations create instances with the same add-to-project rules; confirm library single-click stays read-only.

**Acceptance Scenarios**:

1. **Given** an entry in the user library, **When** the user single-clicks it, **Then** the right property panel opens a read-only view of that saved item’s properties
2. **Given** an expanded library entry with nested children, **When** the user single-clicks a nested child row, **Then** the property panel shows that subpart’s saved properties in read-only form
3. **Given** a library entry’s properties are shown, **When** the user attempts to change a parameter in the panel, **Then** the catalog entry is not modified (controls are not editable, or edits are refused)
4. **Given** a library entry, **When** the user drags it from the library and drops it on the canvas, **Then** a new instance is inserted at the drop position using existing whole-root insert rules
5. **Given** an expanded library entry, **When** the user drags a nested subpart row to the canvas and drops, **Then** only that subpart is inserted per existing subpart insert rules
6. **Given** a library entry or element-list item, **When** the user drags it onto a highlighted group (or project root) in Project structure and releases, **Then** a new instance is added into that scope using the same placement rules as inserting a new item there, becomes selected with Parameters open, and the destination canvas is focused when it differs from the current one
7. **Given** an element-list item, **When** the user drags it onto the canvas and drops, **Then** a new instance is inserted at the drop position using the same rules as today’s palette/canvas add
8. **Given** a drag that ends outside the canvas and outside a valid Project structure drop target, **When** the user releases, **Then** no instance is created

---

### User Story 5 - Group box navigation and stable group chrome (Priority: P3)

A designer works with group boxes. Double-clicking a group box on the canvas opens that group’s inner canvas (restoring previously broken behavior). Moving group boxes no longer causes erratic size changes; group box dimensions remain stable during drag and after parameter edits from the property panel.

**Why this priority**: These are regressions and polish items that depend on the simplified box model but are essential for trustworthy group editing.

**Independent Test**: Create nested groups; double-click to enter/exit; drag group boxes repeatedly; confirm size and randomize placement remain stable.

**Acceptance Scenarios**:

1. **Given** a group box on the canvas, **When** the user double-clicks its body, **Then** the editor opens that group’s inner canvas (same as entering via hierarchy navigation)
2. **Given** a group box at a stable size, **When** the user drags it across the canvas, **Then** its width and height do not change during or after the drag
3. **Given** a group box, **When** the user edits group parameters from the property panel, **Then** the on-canvas box does not spuriously resize except when a parameter explicitly defines group bounds (if applicable)
4. **Given** a group that supports randomize, **When** the user triggers randomize from the property panel, **Then** randomize runs without altering group box dimensions on the canvas

---

### Edge Cases

- What happens when the user single-clicks a deleted or stale row in Project structure after an undo/redo? The tree refreshes; stale rows disappear and no property panel is shown for missing ids.
- How does the system handle clicking a box while the property panel is showing another box’s Parameters tab? The panel switches to the newly clicked box and remains on Parameters; unsaved in-progress edits follow existing commit-or-discard rules for property fields.
- How does the system handle selecting a box while the right menu is on a non-Parameters tab? The menu switches to Parameters for the selected box.
- What happens when dragging in Project structure would create a cycle (group into its descendant)? Drop is rejected; highlight clears; hierarchy unchanged; cables stay as they were.
- What happens when a live Project structure row is dropped on a valid new parent? All cables attached to that moved item are removed first; then the item is placed in the target scope with the same rules used when adding a new item to that scope (including default on-canvas position in that scope). The dropped item is selected, Parameters opens, and if the destination canvas differs the editor focuses that canvas.
- What happens when library or element-list drag targets a nested canvas via the canvas surface (user is inside a group)? Insert occurs in the **current** focused canvas at the drop position, same as manual insert today.
- What happens when library or element-list drag targets a group via Project structure while another canvas is focused? The new instance is added into the highlighted structure target’s scope (not merely the currently focused canvas), using new-item placement rules for that scope; the new instance is selected with Parameters open and that destination canvas is focused.
- What happens when double-click navigate targets an element on a canvas that is not currently focused? The editor switches to the correct canvas first, then centers on the target box.
- What happens when multiple boxes are multi-selected on canvas? Property panel behavior follows a clear rule: show shared/common properties if supported, otherwise show a concise multi-selection state without Parameters editing until a single box is selected.
- What happens when the user tries to edit parameters shown after clicking a library entry? The panel remains read-only; the saved catalog item is unchanged. To edit values, the user must insert an instance into the project and edit that live box.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Canvas boxes MUST display only the element or group display name and connection pins; all other controls (parameters, randomize, auxiliary buttons) MUST be removed from the box body.
- **FR-002**: The right-side property menu MUST include a **Parameters** tab that exposes all parameters previously editable on the box body for the current selection.
- **FR-003**: Single-clicking or selecting a box on the canvas MUST open or update the right property menu for that box and MUST switch the menu to the **Parameters** tab (even if another tab was active).
- **FR-004**: Randomize actions for applicable box types MUST be available only through the property menu, not on the canvas box.
- **FR-005**: A single press-and-hold drag gesture on a box body MUST select (if needed) and move the box without requiring prior activation clicks.
- **FR-006**: Pin/connection interactions MUST take precedence over box-body drag when the user initiates a wiring gesture.
- **FR-007**: Single-clicking an element or group row in **Project structure** MUST open that item’s property panel in the right menu and MUST switch to the **Parameters** tab.
- **FR-008**: Single-clicking a user library entry (root or expanded nested row) MUST open a **read-only** property panel for that saved item in the right menu on the **Parameters** tab; the panel MUST NOT write changes back to the library catalog.
- **FR-009**: Double-clicking a leaf element row in **Project structure** MUST navigate to the canvas containing that element and center the view on its box.
- **FR-010**: Double-clicking a group row in **Project structure** MUST navigate to the parent canvas and center the view on the group’s outer box without opening the group’s inner canvas.
- **FR-011**: Double-clicking a group box on the canvas MUST open that group’s inner canvas.
- **FR-012**: Users MUST be able to drag live element or group rows within **Project structure** onto the project root or another group, subject to valid hierarchy rules. On a successful drop, the system MUST disconnect all cables attached to the moved item (if any), then add it into the target scope using the **same placement rules as inserting a new item** into that scope. The system MUST then select the moved item, switch to the **Parameters** tab, and focus the destination canvas when it differs from the current canvas.
- **FR-013**: During any drag that can target **Project structure** (live reparent, library insert, or element-list insert), the system MUST highlight eligible drop targets by drawing a rectangle around the target folder/group/root label and its nested content rows.
- **FR-014**: Invalid hierarchy drops (including cycles) MUST be rejected with no graph mutation and no cable changes.
- **FR-015**: Users MUST be able to drag a user library entry (root or nested subpart) onto the **canvas** to insert a new instance at the drop location using existing whole-root and subpart insert semantics.
- **FR-015a**: Users MUST be able to drag a user library entry (root or nested subpart) or an **element-list** item onto a highlighted **Project structure** target (group or project root) to insert a new instance into that scope using the same placement rules as adding a new item there. On success, the system MUST select the new instance, switch to the **Parameters** tab, and focus the destination canvas when it differs from the current canvas.
- **FR-015b**: Users MUST be able to drag an **element-list** item onto the canvas to insert a new instance at the drop position using the same rules as today’s palette/canvas add.
- **FR-016**: Moving group boxes MUST NOT cause unintended size changes; group box dimensions MUST remain stable during drag unless the user explicitly resizes via a dedicated resize affordance (if one exists).
- **FR-017**: Property panel edits MUST NOT cause canvas box bodies to grow or shrink to accommodate controls removed from boxes.

### Key Entities

- **Canvas box**: Visual representation of an element or group on a graph canvas; reduced chrome (name + pins only).
- **Property panel**: Right-side menu with tabs including **Parameters** for the current selection context.
- **Selection context**: The currently targeted box, group, library entry, or project-structure row driving property panel content.
- **Project structure row**: Tree node representing a live graph element or group; supports click, double-click navigate, drag-reparent (disconnect-then-add-like-new), and receives drops from the library and element list.
- **Library entry row**: Saved user-box library item (expandable for nested subparts); supports click-for-read-only-properties and drag-to-canvas or drag-to-Project-structure insert. Catalog values are not edited from this panel.
- **Element-list item**: Built-in palette entry the user can drag onto the canvas or onto Project structure to create a new project instance.
- **Drop target highlight**: Transient visual bounds rectangle around a group/folder/root row during a Project structure–targeted drag.
- **Navigation focus**: Editor state pairing active canvas with viewport center on a target box.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In usability testing, 95% of reposition attempts succeed on the first click-and-drag without a prior activation click.
- **SC-002**: Users can locate and edit any box parameter exclusively via the property panel Parameters tab in under 5 seconds after selecting the box.
- **SC-003**: Double-click navigation from Project structure centers the target box in the viewport within 1 second in graphs with up to 100 boxes across nested canvases.
- **SC-004**: After simplifying box chrome, average on-canvas box footprint (width × height) decreases compared to the pre-change baseline for the same graph, without loss of pin legibility.
- **SC-005**: Zero reported instances of group boxes changing size spontaneously during drag in acceptance testing of nested-group scenarios (minimum 20 drag cycles per group type tested).
- **SC-006**: Library and element-list drag-to-canvas, drag-to-Project-structure insert, and Project structure disconnect-then-reparent complete with correct hierarchy (and expected cable removal on reparent) in 100% of valid drop scenarios in the acceptance test matrix.
- **SC-007**: Double-click on group boxes successfully opens inner canvas in 100% of tested group types where inner navigation is supported.

## Assumptions

- **Project structure** and expandable **user library** trees from feature 011-editor-ux-params are already available; this feature extends their interaction model rather than introducing the trees.
- Whole-root and subpart library insert semantics, save rules, and alphabetical ordering from prior specs remain unchanged; this feature adds click-for-read-only-properties plus drag-to-canvas and drag-to-Project-structure placement. Editing saved catalog definitions is out of scope for this feature.
- “Same placement rules as inserting a new item” means the product’s existing add-to-project / add-into-group behavior (default position in that scope, collapsed groups on insert where already defined, etc.), not a separate sibling-index drag API.
- Reparent via Project structure always clears cables on the moved item before re-adding it; preserving cross-group wiring across a reparent is out of scope.
- After a successful Project structure drop (reparent or insert), the result MUST be selected with Parameters open, and the destination canvas focused when it differs from the current canvas.
- The right property menu already exists for some element settings; this feature adds or consolidates a **Parameters** tab rather than inventing a new panel location.
- Multi-selection property editing may show a simplified panel; full multi-edit of parameters is out of scope unless already supported elsewhere.
- Canvas zoom/pan and undo/redo for view changes follow existing behavior; hierarchy moves and inserts remain undoable per current edit-history rules.
- “Center on box” means the box’s visual bounds are brought into the main viewport area with reasonable padding; exact framing matches existing canvas focus behavior used elsewhere in the editor.
- Group box explicit resize handles, if any, are unchanged; this feature only prevents unintended auto-resize from removed inline controls or drag glitches.
