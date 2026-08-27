# Feature Specification: Editor UX & Parameter Flexibility

**Feature Branch**: `011-editor-ux-params`

**Created**: 2026-08-27

**Status**: Draft

**Input**: User description: "add leakyrelu negative slope parameter (default 0.01; RAVE often uses 0.2); expandable project structure/tree and save parts of these instead of just top-level folder — in the library, saved elements must be expandable like folders so the user can insert the whole thing or a subpart; make left and right menus resizable; in top active hierarchy tree, when going back to a parent canvas, children that were once open should stay displayed and clickable until branching elsewhere; add keyword to preserve whatever input dim/channels/features as output even when dims change because of group copies if e.g. a list of the keyword is provided; when a group with N copies is nested in groups with M and O copies, elements display M×N×O parameter values but must also accept N, M×N, or 1 value with under-the-hood copying"

## Clarifications

### Session 2026-08-27

- Q: Should the expandable project structure tree be a separate in-plugin browser of the current graph hierarchy (for navigating and saving any nested box), or only the expandable tree inside saved user-library entries? → A: New project structure tree for the live graph (navigate + save any nested node), plus expandable library entries; placed as a new collapsible section on the left menu under Library
- Q: After the user commits a short parameter list (e.g. N values when full nest needs P), should the editor keep showing the short authored list or always expand to all P values? → A: Keep short authored form in the editable field; also show a read-only expanded preview of all P values
- Q: When an ancestor group’s copy count changes so full length P changes, what happens to an already-saved short authored parameter list? → A: Re-tile when authored length is still in the new dividing set; otherwise flag the parameter invalid with a clear message until the user edits (do not invent a new length)
- Q: What exact token should users type in a dim/channels/features field to mean “keep the corresponding input shape”? → A: `in`
- Q: If the user sets LeakyReLU negative slope outside the allowed range, should the editor refuse the edit or clamp? → A: Refuse out-of-range edits; keep prior value; show a clear message
- Q: (Follow-up) Should the user box library list be ordered? → A: Yes — order library entries by name

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Nested-group parameter lists of any dividing length (Priority: P1)

A designer nests groups with copy counts O (outer), M (middle), and N (inner). An element inside the innermost group shows a parameter list whose full length is M×N×O. Today they must enter the full expanded list. They instead enter a shorter list whose length is 1, N, or M×N (any length that tiles evenly along the nesting axes). The product expands that shorter list by repeating values under the hood so every materialized copy gets a concrete value. The editable field keeps the short authored list; a separate read-only preview shows the full expanded P values.

**Why this priority**: Nested copies already multiply parameter surfaces; without length flexibility, editing deep nests is error-prone and blocks practical RAVE-style stacks.

**Independent Test**: Build O×M×N nesting, set a parameter to one value, then to N values, then to M×N values, then to M×N×O values; confirm the editable field retains the authored length, the read-only preview shows the correctly tiled P values, and audio/graph behavior uses the expanded list.

**Acceptance Scenarios**:

1. **Given** nested groups with copy counts O, M, N (product P = M×N×O > 1) and an element parameter that expects P values, **When** the user enters a single value, **Then** all P slots use that value and the read-only preview shows P copies of that value
2. **Given** the same nest, **When** the user enters exactly N values, **Then** those N values tile across the remaining outer axes to fill P slots; the editable field still shows N values and the preview shows P
3. **Given** the same nest, **When** the user enters exactly M×N values, **Then** those values tile across the outermost axis (O) to fill P slots; the editable field still shows M×N values and the preview shows P
4. **Given** the same nest, **When** the user enters exactly P values, **Then** each slot uses the corresponding entry with no further expansion; editable field and preview both show P
5. **Given** a length that does not divide P along the nesting axes (not in {1, N, M×N, P} for this three-level case, or the analogous dividing lengths for other depths), **When** the user commits the list, **Then** the edit is refused with a clear message and prior values remain
6. **Given** a committed short authored list, **When** the user views the parameter UI, **Then** they can edit the short list and see a read-only expanded preview of all P values that they cannot edit directly
7. **Given** a short authored list of length L that remains valid after an ancestor copy count change, **When** that copy count changes, **Then** L is kept, values re-tile to the new P, and the preview updates
8. **Given** a short authored list whose length is no longer in the dividing set after an ancestor copy count change, **When** that copy count changes, **Then** the parameter is flagged invalid with a clear message until the user edits; authored values are not silently rewritten to a different length

---

### User Story 2 - Preserve-input shape keyword (Priority: P1)

A designer configures an element whose output channels/features/dims should always track its input shape. With group copies, that input shape may itself be a list. They enter the keyword `in` (including a list of `in` when the input side is multi-valued) instead of hard-coding numbers. Output shape stays locked to whatever the corresponding input dim/channels/features resolve to after copy expansion.

**Why this priority**: Hard-coded dims break as soon as copy counts change; `in` keeps graphs legal without manual retuning.

**Independent Test**: Set an output dim field to `in` (and a list of `in` when inputs are multi-valued), change parent/sibling copy counts so input dims change, and confirm outputs follow inputs without illegal connections.

**Acceptance Scenarios**:

1. **Given** an element with a numeric output dim/channels/features field, **When** the user sets it to `in`, **Then** the effective output matches the corresponding input dim/channels/features
2. **Given** group copies that expand an input into a list of shapes, **When** the user provides a list of `in` (or a single `in` that tiles per Story 1 rules), **Then** each expanded output slot preserves its paired input slot
3. **Given** `in` is set, **When** input dims change because copy counts or upstream shapes change, **Then** output dims update accordingly and previously legal wiring stays consistent or is revalidated with the same rules as numeric dims
4. **Given** a field that does not support shape binding, **When** the user enters `in`, **Then** the edit is refused with a clear message

---

### User Story 3 - Project structure tree and expandable library subpart insert (Priority: P2)

A designer uses a **Project structure** section on the left menu (directly under Library). It can be expanded or collapsed. When expanded, it shows the live graph hierarchy as a tree so they can navigate into nested groups and save any nested box—not only a top-level root. Separately, when they save a nested group to the user box library, that library entry expands like a folder tree. They can insert the entire saved root or drill into a nested subgroup/element and insert only that subpart as a new independent instance. Within each library folder, entries are listed in alphabetical order by name so catalogs stay scannable.

**Why this priority**: Reuse today is all-or-nothing at the saved root; a live project tree plus library subpart insert multiplies the value of large architectures.

**Independent Test**: Open Project structure under Library, navigate to a nested box and save it; save a multi-level group, expand it in the library, insert the root once and a nested child once, and confirm both instances match the corresponding saved structure/parameters independently. With several differently named entries in one folder, confirm they appear sorted by name.

**Acceptance Scenarios**:

1. **Given** the left menu, **When** the user looks under Library, **Then** a Project structure section is present and can be expanded or collapsed
2. **Given** Project structure is expanded and the graph has nested groups, **When** the user browses the tree, **Then** they see the live hierarchy and can navigate to nested canvases/boxes
3. **Given** a nested box selected or targeted in Project structure (or on canvas), **When** the user saves to the user library, **Then** that nested part is saved (not only top-level roots)
4. **Given** a saved library entry whose target contains nested groups/elements, **When** the user opens the library browser, **Then** the entry is expandable and shows nested children as a tree
5. **Given** an expanded library entry, **When** the user inserts the root, **Then** a full independent instance of the saved hierarchy is placed (existing collapsed-on-insert rules for groups still apply)
6. **Given** an expanded library entry, **When** the user inserts a nested child node, **Then** only that subtree (element or group) is placed as a new independent instance with its saved parameters
7. **Given** a library entry that is a single leaf element, **When** the user browses it, **Then** it appears as a non-expandable (or empty-children) leaf and inserts as today
8. **Given** two or more user library entries in the same folder with distinct names, **When** the user browses that folder, **Then** entries appear in alphabetical order by display name (case-insensitive)

---

### User Story 4 - Hierarchy trail keeps visited children (Priority: P2)

While diving into nested group canvases, the designer uses the top active hierarchy tree. After opening child A then returning to the parent, child A remains visible and clickable in the trail so they can jump back. If they instead open a different sibling branch, the previous branch’s lingering children are cleared according to the new path.

**Why this priority**: Constant re-drilling into the same nested groups wastes time; sticky trail children match expected spatial navigation.

**Independent Test**: Open parent → child → grandchild, navigate up to parent, confirm child and grandchild remain clickable; navigate into a different sibling and confirm the old branch no longer clutters the trail.

**Acceptance Scenarios**:

1. **Given** the user has opened a child canvas from a parent, **When** they navigate back to the parent, **Then** that child remains displayed in the top hierarchy tree and is clickable
2. **Given** a deeper path parent → A → A1, **When** the user returns to parent, **Then** A and A1 remain displayed and clickable until the user branches elsewhere
3. **Given** visited children still shown under the current parent, **When** the user navigates into a different sibling branch, **Then** the previous branch’s sticky children are removed from the trail and the new path is shown
4. **Given** sticky children in the trail, **When** the user clicks one, **Then** the editor opens that canvas without recreating the path from scratch

---

### User Story 5 - Resizable side menus (Priority: P3)

The designer drags the boundary of the left and/or right side menus to widen or narrow them so the graph canvas and inspectors fit their screen and workflow.

**Why this priority**: Comfort and readability on varied displays; independent of graph semantics.

**Independent Test**: Drag left and right panel edges, confirm widths change within min/max bounds and persist for the session (and across sessions if layout persistence already exists for the plugin).

**Acceptance Scenarios**:

1. **Given** the plugin window with left and right menus visible, **When** the user drags a menu’s resize handle, **Then** that menu’s width updates live
2. **Given** a resize attempt below the minimum or above the maximum allowed width, **When** the user releases, **Then** width clamps to the allowed range and the canvas remains usable
3. **Given** resized menus, **When** the user continues editing the graph, **Then** layout remains stable (no jump-back on the next frame)

---

### User Story 6 - LeakyReLU negative slope (Priority: P3)

A designer places a LeakyReLU element and sets its negative slope. The default is 0.01. For RAVE-like stacks they set 0.2. The value participates in live inference and freeze/compile like other numeric element parameters.

**Why this priority**: Small but required for faithful RAVE-style activations; isolated element parameter work.

**Independent Test**: Place LeakyReLU, leave default 0.01, change to 0.2, save/reload, freeze if applicable, and confirm the slope is retained and affects processing.

**Acceptance Scenarios**:

1. **Given** a newly placed LeakyReLU, **When** the user inspects its parameters, **Then** negative slope is present and defaults to 0.01
2. **Given** a LeakyReLU, **When** the user sets negative slope to 0.2 (or another valid value), **Then** the value is stored and used in live processing
3. **Given** a non-default slope, **When** the project is saved and reloaded (and when the element is included in a freeze), **Then** the slope is preserved
4. **Given** a LeakyReLU with a valid slope, **When** the user enters an out-of-range value (e.g. negative), **Then** the edit is refused with a clear message and the prior slope remains

---

### Edge Cases

- Parameter list length valid for a shallower nest but wrong after an outer copy count changes — if the authored length remains in the new dividing set, keep it and re-tile into the new P (preview updates); if not, mark the parameter invalid with a clear message until the user fixes the list; do not silently invent a new authored length or misalign slots
- Preserve keyword (`in`) mixed with numeric entries in the same list — refuse with a clear message (v1: no mixing in one field)
- Inserting a library subpart that contained external cables to siblings outside that subpart — place the subpart without those external cables; do not invent connections
- Project structure section collapsed by default or remembering last expand/collapse — either is acceptable if expand/collapse works; stale nodes after delete/ungroup must disappear from the tree
- Hierarchy sticky trail when the remembered child group is deleted or ungrouped — remove stale entries from the trail
- Resizing side menus until the canvas width would become unusable — enforce minimum canvas width as well as menu min/max
- LeakyReLU negative slope outside the allowed range (non-negative finite; practical [0, 1]) — refuse the edit, keep the prior value, and show a clear message; do not clamp

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: LeakyReLU MUST expose a negative-slope parameter defaulting to **0.01**, editable by the user, persisted with the element, and honored in live processing and freeze/compile paths. Out-of-range values (outside non-negative finite / practical [0, 1]) MUST be refused with a clear message; the prior value MUST remain unchanged (no clamping).
- **FR-002**: Left and right side menus MUST be user-resizable by dragging, with enforced minimum and maximum widths so the graph canvas remains usable.
- **FR-003**: The top active hierarchy tree MUST retain previously opened child canvases as visible, clickable entries when the user navigates up to an ancestor, until the user navigates into a different branch (at which point the prior branch’s sticky children are cleared).
- **FR-004**: User box library entries that contain nested structure MUST be presented as an expandable tree; users MUST be able to insert either the saved root or any nested subpart as a new independent instance.
- **FR-004a**: Within each user box library folder, entries MUST be listed in alphabetical order by display name (case-insensitive). Rename and save MUST keep the visible order consistent with the new name.
- **FR-005**: The left menu MUST include a **Project structure** section directly under Library that the user can expand or collapse; when expanded it MUST show the live graph hierarchy as a tree for navigation and for targeting nested boxes.
- **FR-005a**: Users MUST be able to save nested parts of the project/group tree (not only a top-level root), including via Project structure targeting, following existing single-box library save rules (Audio I/O excluded).
- **FR-006**: Shape-related dim/channels/features fields that support binding MUST accept the reserved keyword **`in`** meaning “use the corresponding input dim/channels/features as this output (or bound) value,” including when those inputs are lists expanded by group copies.
- **FR-007**: When an `in` keyword list is provided (or a shorter list that tiles per FR-008), each expanded slot MUST track its paired input slot as copy counts and upstream shapes change.
- **FR-008**: For an element under nested groups whose copy counts multiply to total length P, multi-value parameters MUST accept any length L in the dividing set induced by the nest (for three levels O, M, N: L ∈ {1, N, M×N, P}; generally, lengths formed by taking a suffix of the copy-count product along the nesting axes from the innermost group outward), MUST store the authored list at length L, MUST expand by tiling/copying under the hood to P concrete values for processing, and MUST show a read-only expanded preview of all P values alongside the editable authored list.
- **FR-009**: Parameter lists whose length is not in the allowed dividing set for the current nest MUST be rejected with a clear, user-visible explanation; prior committed values MUST remain unchanged.
- **FR-010**: Expanded parameter values and preserve bindings MUST remain correct after project save/load. When any ancestor group’s copy count changes, if the authored list length is still in the new dividing set the system MUST keep that authored list and re-tile to the new P; if not, the parameter MUST be marked invalid with a clear user-visible message until the user edits it—MUST NOT invent a new authored length silently.
- **FR-010a**: While a parameter is invalid due to FR-010, live shape inference and processing MUST treat it as unresolved (same class of user-visible illegality as other invalid parameter edits), without corrupting sibling parameters.

### Key Entities

- **Hierarchy Trail Entry**: A remembered open child (or deeper descendant) shown in the top hierarchy tree while viewing an ancestor; cleared on branch change or when the target no longer exists.
- **Project Structure Section**: Collapsible left-menu section under Library that mirrors the live graph hierarchy for navigation and save targeting.
- **Library Tree Node**: A browsable node in a user library entry representing the saved root or a nested element/group; insertable independently.
- **Preserve Keyword (`in`)**: The token `in` in a shape/dim field that binds the field to the corresponding input dim/channels/features rather than a fixed number.
- **Copy-Expanded Parameter List**: The full P-length value list for a parameter under nested group copies, derived by tiling a shorter authored list; shown read-only as an expanded preview while the editable field retains the authored length.
- **LeakyReLU Negative Slope**: Per-element numeric parameter controlling the slope for negative inputs; default 0.01.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a three-level nest with P = M×N×O ≥ 8, a user can successfully commit parameter lists of lengths 1, N, M×N, and P within one continuous edit session; for each short length the editable field retains that length, the read-only preview shows the correctly tiled full list, and processing uses the expanded values.
- **SC-002**: After changing an ancestor copy count that alters input shapes, graphs using `in` update output dims to match inputs without the user retyping numeric dims, in under 10 seconds of interaction (change copy count → observe updated shapes).
- **SC-003**: From a multi-level library entry, a user can insert the root and at least one nested subpart as separate instances in under 2 minutes, with structure/parameters matching the saved subtrees.
- **SC-003a**: A user can open Project structure under Library, locate a nested box, and save it to the user library in under 2 minutes without flattening the graph to the top-level canvas first.
- **SC-003b**: With three or more differently named entries in one library folder, the visible list order matches case-insensitive alphabetical order by display name after browse, save, and rename.
- **SC-004**: After diving at least two levels deep and returning to the parent, the user can re-enter a previously visited child with one click on the hierarchy trail (no need to re-expand the group from the canvas).
- **SC-005**: Users can resize left and right menus and still keep a usable canvas; width remains stable for the rest of the session after release.
- **SC-006**: A LeakyReLU placed with default slope 0.01, then set to 0.2, retains 0.2 after save/reload in 100% of manual verification trials.

## Assumptions

- The preserve-input keyword is the reserved token **`in`** (case-sensitive lowercase), entered in place of a number; a list may be all `in` entries (or a shorter all-`in` list that tiles). Mixing `in` with numbers in one field is out of scope for v1.
- “Corresponding input” means the natural shape peer for that field (e.g. output channels follow input channels on the same element), consistent with existing live shape inference.
- Dividing lengths for nested copies always include 1 and P, plus products of the innermost contiguous copy counts (N, then M×N, then O×M×N, …) matching the user’s N / M×N / M×N×O examples.
- Authored short lists are what persist on save; the expanded P preview is derived and not separately edited. On ancestor copy-count change: re-tile if authored length still valid; otherwise flag invalid until user edit.
- Library entry sort is by display name, case-insensitive, within each folder; folder names remain sorted as today. Nested member rows inside an expanded library entry SHOULD also list alphabetically by member label when labels are available.
- Library subpart insert creates a standalone instance; cables that crossed the subpart boundary in the original save are not recreated.
- Save-of-nested-part follows existing single-box library save rules (one target box; Audio I/O excluded) from the groups/library feature; this feature adds the left-menu Project structure section, library tree expansion, and nested save/insert targets—not multi-select save.
- Group insert collapsed-on-place behavior from the existing library feature remains unchanged unless the user inserts a nested subpart that is itself a single element.
- Side-menu width persistence across plugin sessions follows whatever layout persistence the plugin already uses; if none exists, session-only persistence is acceptable for v1.
- LeakyReLU negative slope allowed range is non-negative finite values in practical range [0, 1]; out-of-range edits are refused (not clamped) with a clear message.
- Resizable menus refer to the main left and right chrome panels of the graph editor (element library / inspectors / Project structure), not floating popups.
- This feature extends existing groups, library, and hierarchy navigation; it does not replace the user box library or training library.
