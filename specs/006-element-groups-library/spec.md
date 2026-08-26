# Feature Specification: Element Groups & User Box Library

**Feature Branch**: `006-element-groups-library`

**Created**: 2026-08-26

**Status**: Draft

**Input**: User description: "add ability to group elements and subgroups; add ability to open/extend/expand/collapse groups in the ui to show inside or hide under one box; add ability to save boxes (elements or groups) with their parameters in a user library; group copies parameter N (independent serial materialized blocks when I/O allows)"

## Clarifications

### Session 2026-08-26

- Q: Which kinds of graph boxes are allowed inside groups and as targets when saving to the user box library? → A: Any except Audio I/O (live processing, Knob/XY and other non-audio-I/O sources, nested groups, and Gold BlackBoxes are allowed; Audio Input and Audio Output are excluded)
- Q: Implementation preference for groups and library UI? → A: Prefer built-in Dear ImGui / imgui-node-editor capabilities as much as possible for grouping, collapse/expand, and library browsing/placement rather than bespoke parallel UI systems
- Q: When the user freezes while a group (or multi-selection) is selected, what should happen? → A: Freeze each freezable member of the selection individually; do not replace the whole selection with one Gold BlackBox; group/selection structure remains with individually frozen elements
- Q: If several ungrouped boxes are selected, can that selection be saved as one library entry? → A: No — save to library is an action on a single box (Gold or live; element or group), not on a multi-selection
- Q: When a group is saved to the library and later inserted, should expand/collapse states match the save-time tree? → A: No — always insert with the root group collapsed and nested groups collapsed (Option B)
- Q: Should a group expose a copy-count (N blocks) parameter? → A: Yes — integer N (default 1); independent weight copies (not shared); materialize N copies on the canvas; wire as a serial chain when I/O shapes allow; refuse/clamp N with a clear message if chaining is illegal; new copies clone the last copy’s parameters/weights; nested groups each have their own N; library save includes N and all materialized copies

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Group Elements and Nested Subgroups (Priority: P1)

A sound designer building a large graph selects several related elements (for example a Conv1D plus an activation and a residual branch) and groups them into a single named box. They can place additional elements inside that group, and can also create subgroups inside an existing group so the architecture stays hierarchical and readable. External cables into and out of the group remain meaningful: the group exposes the connections needed to wire it into the rest of the graph without losing the internal wiring.

**Why this priority**: Grouping is the foundation for collapse/expand and for saving reusable compound boxes. Without durable groups and subgroups, the other stories cannot deliver value.

**Independent Test**: Select three connected elements, create a group, rename it, add a fourth element into the group, create a nested subgroup of two members, save and reload the project, and confirm structure and connections are intact.

**Acceptance Scenarios**:

1. **Given** two or more allowed boxes selected on the canvas (no Audio I/O), **When** the user creates a group, **Then** those boxes become members of one named group box and remain connected to each other as before
2. **Given** a selection that includes Audio Input or Audio Output, **When** the user attempts to create a group or save to the user library, **Then** the action is refused with a clear message and nothing is grouped or saved
3. **Given** an existing group, **When** the user selects elements inside it and creates a subgroup, **Then** a nested group appears inside the parent without breaking parent or sibling wiring
4. **Given** a group with members, **When** the user adds or removes members (including dragging an element into or out of the group), **Then** membership updates and external connections to non-members remain valid or are clearly refused with an explanation
5. **Given** a graph containing nested groups, **When** the project/plugin state is saved and reloaded, **Then** group hierarchy, names, membership, and internal/external connections are restored unchanged
6. **Given** a group, **When** the user ungroups it, **Then** members return to the parent scope (or top-level canvas) with their connections preserved
7. **Given** a group (collapsed or expanded) with several freezable live members selected via the group, **When** the user chooses Freeze, **Then** each freezable member becomes its own Gold box in place, the group container remains, and the selection is not replaced by a single Gold BlackBox

---

### User Story 5 - Stack Group Copies (N Blocks) (Priority: P1)

A designer builds one processing block as a group (for example a residual unit). On the group they set a **copies** parameter **N** (like repeated blocks in a deep network). When N increases and the group’s external I/O shapes allow chaining, the graph **materializes** N independent copies on the canvas (each with its own weights/parameters), wired in **series** (copy i outputs → copy i+1 matching inputs). When shapes do not allow a legal chain, raising N is refused or clamped with a clear explanation. Decreasing N removes trailing copies. Nested groups each expose their own N.

**Why this priority**: Repeat-count is a core architecture control for deep graphs; it depends on groups existing (US1) and should ship with grouping rather than as a later add-on.

**Independent Test**: Create a chainable group, set N from 1→3, confirm three independent editable copies appear in series with distinct weights after randomizing one copy; set N to an illegal value for a non-chainable group and confirm refusal; save/reload preserves N and all copies.

**Acceptance Scenarios**:

1. **Given** a group with default N=1, **When** the user views group parameters, **Then** a copies/N control is visible and editable
2. **Given** a group whose external I/O shapes allow serial chaining, **When** the user sets N to K (K&gt;1), **Then** the canvas shows K independent copies (separate parameters/weights), wired in series, and external graph cables attach to the first copy’s inputs and last copy’s outputs as appropriate
3. **Given** N&gt;1, **When** the user randomizes or edits weights on copy 2 only, **Then** other copies are unchanged
4. **Given** N=K&gt;1, **When** the user sets N to K+1, **Then** the new trailing copy is initialized by cloning the previous last copy’s structure and parameters/weights
5. **Given** N=K&gt;1, **When** the user sets N to a smaller value, **Then** trailing copies are removed and remaining wiring stays consistent
6. **Given** a group whose I/O cannot form a legal serial chain for N&gt;1, **When** the user attempts N&gt;1, **Then** the change is refused or clamped with a clear message and the graph is not partially corrupted
7. **Given** nested groups each with their own N, **When** the user changes an inner group’s N, **Then** only that group’s copies update
8. **Given** a group with N&gt;1, **When** the project is saved and reloaded, **Then** N and all materialized copies (structure, links, parameters) are restored
9. **Given** a group with N&gt;1 saved to the user box library and later inserted, **Then** the library entry restores N and all copies (insert still respects collapsed-group presentation rules)

---

### User Story 2 - Expand and Collapse Groups in the UI (Priority: P1)

While editing, the designer collapses a group so it appears as a single compact box on the canvas (hiding internal elements and cables). They expand it again when they need to inspect or edit the interior. Nested groups can be collapsed independently so only the currently relevant branch is expanded. Collapsed groups still show enough information (name and external ports) to stay connectable and identifiable in the larger graph.

**Why this priority**: Collapse/expand is the primary usability payoff of grouping on dense neural graphs; it is independently valuable even before the library exists.

**Independent Test**: Create a group with several members, collapse it to a single box, confirm internals are hidden and external cables still work, expand it again, and verify all internals reappear in editable form. Repeat with one nested subgroup collapsed while the parent stays expanded.

**Acceptance Scenarios**:

1. **Given** an expanded group, **When** the user collapses it, **Then** member elements and internal cables are hidden and the group renders as one box with its name and external connection points
2. **Given** a collapsed group, **When** the user expands it, **Then** members and internal layout become visible and editable again
3. **Given** nested groups, **When** the user collapses an inner group only, **Then** the outer group remains expanded and only the inner content is hidden under the inner box
4. **Given** a collapsed group with external cables, **When** the graph is processing audio, **Then** audio behavior is unchanged by the visual collapse (collapse is presentation-only for editing clarity)
5. **Given** a graph with mixed expanded/collapsed groups, **When** the project is saved and reloaded, **Then** each group's expand/collapse state is restored

---

### User Story 3 - Save Boxes to a User Library (Priority: P1)

The designer finds a useful single element or a whole group box (Gold or live, with nested structure and parameters) and uses a **per-box** “save to library” action on that one box. Later, in the same or a new project, they browse the library, pick a saved box, and place an instance onto the canvas with the stored structure and parameters applied. They can rename, overwrite, or delete library entries so the catalog stays useful over time. Multi-select is not a save target: to save several siblings together, the user must group them first, then save that group box.

**Why this priority**: Reuse of tuned boxes is the long-term productivity goal of grouping; saving parameters with structure is what makes the library more than a template of empty nodes.

**Independent Test**: Configure an element (and separately a multi-element group) with non-default parameters, invoke save on each box individually under distinct names, clear or open a fresh graph, insert both entries from the library, and confirm structure and parameter values match what was saved. Confirm that with multiple boxes selected, save-to-library is unavailable or refused.

**Acceptance Scenarios**:

1. **Given** a single focused/target box (element or group, Gold or live), **When** the user runs save to library with a name, **Then** a library entry appears storing that box’s structure and parameters
2. **Given** multiple boxes selected and no single group box as the save target, **When** the user attempts save to library, **Then** the action is unavailable or refused with a clear message that save applies to one box (element or group)
3. **Given** a library entry for a single element, **When** the user inserts it onto the canvas, **Then** a new element appears with the saved parameter values
4. **Given** a library entry for a group (including subgroups), **When** the user inserts it onto the canvas, **Then** the full hierarchy, internal connections, and member parameters are recreated as a new independent instance with the root group and all nested groups collapsed
5. **Given** an existing library entry name, **When** the user saves again under the same name and confirms overwrite, **Then** the entry is replaced; if they cancel, the previous entry remains unchanged
6. **Given** library entries, **When** the user renames or deletes an entry (delete with confirmation), **Then** the catalog updates and deleted entries are no longer insertable
7. **Given** library entries created in a previous session, **When** the user reopens the plugin, **Then** the user library is available without re-importing

---

### User Story 4 - Browse and Place From the User Library (Priority: P2)

From the element/menu area of the plugin, the user opens their box library, scans saved names (and basic type hints such as “element” vs “group”), and places entries onto the graph the same way they place built-in elements—by drag or insert onto the canvas.

**Why this priority**: Discovery and placement UX makes the library habitual; core save/insert (Story 3) already delivers reuse if the list is reachable.

**Independent Test**: With at least two library entries, open the library browser, identify each by name and kind, and place one via the primary insert gesture used for built-in elements.

**Acceptance Scenarios**:

1. **Given** the plugin UI, **When** the user opens the user box library, **Then** saved entries are listed with name and whether each is a single element or a group
2. **Given** a library entry in the list, **When** the user places it on the canvas, **Then** an instance is created at the drop/insert location without modifying the stored library original; if the entry is a group, it appears collapsed (including nested groups)
3. **Given** an empty library, **When** the user opens the library browser, **Then** they see an empty state with a clear prompt that they can save boxes from the canvas

---

### Edge Cases

- What happens if the user tries to group a single element? Grouping requires at least two members; otherwise the action is disabled or refused with a clear message (user may still save a single element to the library without grouping).
- What happens if the selection (or target for library save) includes Audio Input or Audio Output? The action is refused with a clear message naming the disallowed I/O boxes; the graph and library are left unchanged.
- What happens if grouping would create a cycle or illegal nesting (e.g., moving a group into one of its descendants)? The action is refused with a clear explanation; the graph is left unchanged.
- What happens if a collapsed group is selected for freeze or similar existing graph operations? Freeze applies to freezable members inside the selection individually (each becomes its own Gold box where freeze applies); the group container and non-freezable members stay in place — the selection is never collapsed into a single Gold BlackBox. The user can still expand the group afterward.
- What happens if some members of a selection are already Gold or otherwise not freezable? Those members are skipped; freezable members still freeze individually; the user sees a clear summary if anything was skipped.
- What happens if a library insert references an element type that is no longer available in this plugin version? Insert fails with a clear message; other library entries remain usable.
- What happens if two library saves use the same name? User must confirm overwrite or choose a different name; silent overwrite is not allowed.
- What happens if the user tries to save a multi-selection to the library? Save is refused or disabled; only one box (element or group) can be the save target.
- What happens if the user deletes a group that still has members? Ungroup/delete-group behavior preserves members on the parent canvas unless the user explicitly chooses a destructive delete-all-members action (if offered), which requires confirmation.
- How does the system handle very deep nesting? Nesting is allowed; if a practical depth limit is enforced for stability, exceeding it is refused with a clear message (default target: at least 5 levels deep supported).
- What happens if the user sets N very large? A practical upper bound may clamp N with a clear message (implementation safety); default product intent supports useful deep stacks (at least N=8 in guided tests).
- What happens to collapse when N&gt;1? Collapse still applies to the group container presentation; materialized copies remain members of the repeat structure and follow the same collapse rules as other group content.
- What if mid-chain copies are manually rewired in a way that breaks serial topology? Subsequent N changes re-validate chainability; illegal N updates are refused until the user fixes shapes/wiring (or the product re-applies serial wiring on N change—prefer re-assert serial chain between copies on successful N update).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to create a named group from a selection of two or more allowed graph boxes (or groups) on the canvas.
- **FR-001a**: Groups and user-library saves MUST allow any box kinds except Audio Input and Audio Output — including live processing elements, Knob/XY and other non-audio-I/O sources, nested groups, and Gold BlackBoxes. Selections that include Audio I/O MUST be refused with a clear message.
- **FR-001b**: When Freeze is invoked on a selection that includes a group and/or multiple elements, the system MUST freeze each freezable member individually (each eligible live element becomes its own Gold box). The system MUST NOT replace the whole selection or whole group with a single Gold BlackBox. Group membership and layout of non-targeted members MUST remain intact.
- **FR-002**: Users MUST be able to create nested subgroups inside an existing group.
- **FR-003**: Users MUST be able to add elements to a group, remove elements from a group, rename a group, and ungroup without losing member connections that remain valid.
- **FR-004**: Group membership and hierarchy MUST persist across project/plugin save and load.
- **FR-005**: Users MUST be able to collapse a group so it appears as a single box that hides internal members and internal cables while still exposing the group’s name and external connection points.
- **FR-006**: Users MUST be able to expand a collapsed group to reveal and edit its contents.
- **FR-007**: Nested groups MUST support independent expand/collapse states.
- **FR-008**: Expand/collapse MUST be a visual/editing affordance only: collapsing MUST NOT change audio processing behavior of the enclosed graph.
- **FR-009**: Expand/collapse state per group MUST persist across project/plugin save and load.
- **FR-010**: Users MUST be able to save exactly one target box at a time — a single element or a single group (Gold or live, including nested structure for groups) — to a user box library together with its parameters (Audio I/O excluded per FR-001a). Save MUST NOT operate on a multi-selection of sibling boxes; the user MUST group first if they want those siblings saved together.
- **FR-011**: Saved library entries MUST include enough information to recreate structure, internal connections (for groups), and parameter values on insert.
- **FR-012**: Users MUST be able to insert a library entry onto the canvas as a new independent instance without altering the stored entry.
- **FR-012a**: When inserting a group library entry, the placed root group and all nested groups MUST start collapsed (save-time expand/collapse presentation is not restored on insert).
- **FR-013**: Users MUST be able to browse, rename, overwrite (with confirmation), and delete (with confirmation) user library entries from inside the plugin.
- **FR-014**: The user box library MUST persist in local user data across plugin sessions (distinct from the training sample library used for Train workflows).
- **FR-015**: Illegal grouping moves (cycles, nesting a group into its descendant) MUST be refused with a clear user-visible reason and no partial corruption of the graph.
- **FR-016**: Library insert of an entry that cannot be reconstructed (e.g., missing element type) MUST fail safely with a clear message without corrupting the current graph.
- **FR-017**: Each group MUST expose an integer **copies** parameter N (default 1, N ≥ 1) controlling how many times the group’s block is instantiated.
- **FR-017a**: For N &gt; 1, the system MUST **materialize** N **independent** copies on the canvas (separate parameters and weights—not shared across copies) and MUST wire them as a **serial chain** when external I/O shapes allow (copy i outputs to copy i+1 matching inputs). Outside-world cables MUST connect to the first copy’s inputs and the last copy’s outputs as appropriate.
- **FR-017b**: If serial chaining is not shape-legal for the requested N, the system MUST refuse or clamp N with a clear user-visible reason and MUST NOT leave a partially corrupted chain.
- **FR-017c**: When N increases, new trailing copies MUST be initialized by cloning the previous last copy’s structure and parameters/weights. When N decreases, trailing copies MUST be removed safely.
- **FR-017d**: Nested groups MUST each have their own independent N. Freeze and library save/load MUST treat all materialized copy members as normal freezable/saveable graph content (per-member freeze still applies).
- **FR-017e**: N and all materialized copies MUST persist across project/plugin save and load and MUST be included when saving a group to the user box library.

### Key Entities

- **Group**: A named container of allowed graph members (elements and/or nested groups, including Gold BlackBoxes and non-audio-I/O sources; never Audio I/O); has expand/collapse state, external connection points, membership, and a **copies** count N.
- **Group Copy Instance**: One materialized independent replica of a group’s block when N &gt; 1; has its own members/weights; participates in the serial chain with sibling copies.
- **Graph Element (Box)**: A single processing or I/O node on the canvas with typed ports and parameters; may be saved alone to the library.
- **User Box Library Entry**: A named, reusable snapshot of either one element or one group hierarchy, including parameters and (for groups) internal structure, connections, N, and all materialized copies; stored in the user’s local library catalog.
- **User Box Library**: The per-user catalog of saved entries, browsable and editable inside the plugin, independent of any open project’s canvas and independent of the training sample library.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In guided tests, a user can create a named group of at least three elements, nest one subgroup, and restore that hierarchy after save/reload with 100% structural fidelity (membership and connections).
- **SC-002**: A user can collapse a group to a single box and expand it again in under 2 seconds of interaction time per action; collapsed groups remain identifiable by name on the canvas.
- **SC-003**: Collapse/expand never changes audible output for an otherwise unchanged graph (verified by A/B listening or equivalent signal comparison in test scenarios).
- **SC-004**: A user can save a configured element and a multi-level group to the library and re-insert both into a clean graph in under 3 minutes, with parameter values matching the saved originals; inserted groups appear fully collapsed until the user expands them.
- **SC-005**: At least 90% of first-time test users successfully complete “group → collapse → expand → save to library → insert from library” without assistance beyond in-UI labels.
- **SC-006**: Nested expand/collapse works for at least 3 levels of depth without loss of editability for currently expanded scopes.
- **SC-008**: Freezing a selected group with M freezable live members results in M individual Gold boxes (or fewer if some are skipped as non-freezable), never a single Gold box standing in for the entire group.
- **SC-009**: On a chainable group, a user can set copies N from 1 to at least 8 and see N independent serially wired copies on the canvas; editing weights on one copy does not change the others.
- **SC-010**: Attempting N &gt; 1 on a non-chainable group is refused or clamped with an explanatory message in 100% of guided tests, with no partial/orphan copy nodes left behind.
- **SC-011**: After save/reload (and after box-library save/insert of a group with N &gt; 1), N and all copy instances match the pre-save structure and parameter fidelity.

## Assumptions

- Groups are organizational and structural editing constructs on the existing graph canvas; they do not replace Freeze/BlackBox compilation and do not by themselves change live vs frozen execution semantics. The copies parameter materializes real graph content (independent instances), which then participate in live/frozen execution like any other nodes.
- Freeze on a group or multi-selection freezes freezable members one-by-one (individual Gold boxes), preserving group hierarchy; it does not compile the selection into one combined BlackBox in this feature’s behavior. All materialized copy members are included when expanding a group selection for freeze.
- Audio Input and Audio Output stay at project graph scope and are never members of groups or contents of user box library entries; other box kinds (including Gold BlackBoxes and non-audio-I/O sources such as Knob/XY) may be grouped and saved.
- Group copies use **independent** weights/parameters (not shared). Serial chaining is the only v1 topology for N &gt; 1; residual-around-stack and parallel fan-out are out of scope unless added later.
- Default N is 1. New copies clone the last copy. Nested groups each own their N. A practical max N may be enforced for stability with a clear message.
- “Parameters” saved with a box include the user-facing configuration values of each element (and, for weighted elements, the current weight/bias state when those parameters already exist), so re-inserted boxes sound/behave like the saved originals under the same inputs. Saving a Gold BlackBox preserves enough state to restore that frozen box on insert.
- The user box library is local to the machine/user profile for v1; cloud sync and marketplace publishing of boxes are out of scope (marketplace remains a later phase).
- The user box library is separate from the Training Library (sample pairs/clips used for Train); naming and UI placement should keep the two catalogs distinguishable.
- Grouping requires at least two members; saving to the library is always a per-box action on one element or one group (never a raw multi-selection).
- Library inserts of groups always place the root and nested groups collapsed; expand/collapse state inside an open project still persists via project/plugin save (FR-009) after the user changes it on the canvas.
- External ports of a group are derived from connections that cross the group boundary (or equivalent clear aggregation); users do not need a separate “port definition” workflow for v1 beyond what grouping naturally exposes.
- Built-in element menu behavior remains the source of stock element types; the user library is an additional catalog for user-saved boxes.
- Groups, collapse/expand, and the user box library MUST reuse the plugin’s existing embedded node-editor UI capabilities as far as practical (native group/collapse patterns and standard in-plugin list/menu patterns for the library) instead of introducing a separate parallel UI surface for these flows.
- Sharing library files between machines via manual export/import is deferred unless already trivial; v1 success is defined by persistence on the same user environment.
- Depth of nesting is supported to at least five levels; any hard cap is an implementation safety limit communicated in the UI, not a product goal to encourage shallow-only graphs.
