# Feature Specification: Preset Management & Undo/Redo History

**Feature Branch**: `008-preset-undo-history`

**Created**: 2026-08-26

**Status**: Draft

**Input**: User description: "- implement preset management - implement undo/redo history"

## Clarifications

### Session 2026-08-26

- Q: When saving a preset, should the stored patch include current model weights and Gold (frozen) artifacts for identical sound, or only structure/parameters? → A: Full sonic recall — graph + parameters + weights and Gold artifacts needed for identical sound
- Q: After loading a named preset and then editing, how should current-preset identity work? → A: Keep loaded name with dirty indicator; Save overwrites that entry, Save As creates another
- Q: Should weight randomization (and similar one-shot sonic actions) be undoable as a single history step? → A: Yes — randomization and similar one-shot sonic actions are one undo/redo step
- Q: When importing a preset under a name that already exists in the catalog, what should happen? → A: Prompt to overwrite; cancel leaves the catalog unchanged
- Q: Should Undo/Redo include view-only editor changes (canvas pan, zoom, selection), or only patch-affecting edits? → A: Patch-affecting edits only (structure, params, weights, box layout positions); exclude pan/zoom/selection

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Save and Recall Named Presets (Priority: P1)

A sound designer finishes a useful plugin setup (graph layout, connections, parameters, weights, and Gold state so the sound can be recalled identically). They save it under a clear name from inside the plugin. Later—in the same project or a different one—they open the preset browser, pick that name, and the plugin restores to that saved patch. The UI shows that name as the current preset. If they edit further, a dirty/modified indicator appears; **Save** writes back to that named entry, and **Save As** creates a new catalog name.

**Why this priority**: Named full-patch recall is the core of preset management and the primary productivity win for returning to known sounds.

**Independent Test**: Configure a non-default graph with non-default weights (and Gold state if present), save as a named preset, change the graph and weights substantially, load the preset, and confirm structure and sonic result match what was saved. Edit once, confirm dirty indicator, Save, reload elsewhere, and confirm the overwrite stuck; Save As under a second name and confirm both entries exist.

**Acceptance Scenarios**:

1. **Given** a configured plugin state with no current preset (or after Save As to a new unique name), **When** the user saves with a unique name, **Then** a named preset entry appears in the catalog, that name becomes the current preset, and it remains available after closing and reopening the plugin
2. **Given** a named preset in the catalog, **When** the user loads it, **Then** the plugin restores the saved graph, parameters, weights, and Gold artifacts so the result matches the saved sonic patch, and the UI shows that name as the current preset (not dirty)
3. **Given** a current preset that is dirty, **When** the user chooses Save, **Then** that catalog entry is overwritten with the current full sonic patch and the dirty indicator clears (overwrite of the current entry does not require picking a new name; a brief confirm MAY still be shown)
4. **Given** any current patch, **When** the user chooses Save As with a new unique name, **Then** a new catalog entry is created and becomes the current preset (not dirty)
5. **Given** Save As (or first save) targeting an existing catalog name that is not the current preset’s name, **When** the user confirms overwrite, **Then** that entry is replaced; if they cancel, the catalog and current association remain unchanged
6. **Given** audio is playing, **When** the user loads a preset, **Then** the new state applies without stopping the host transport and without audible dropouts beyond a brief, seamless transition
7. **Given** a preset saved in one project/session, **When** the user opens another project or a fresh plugin instance, **Then** they can load that same named preset without re-importing

---

### User Story 2 - Browse, Rename, and Delete Presets (Priority: P1)

The designer opens an in-plugin preset browser, scans saved names, renames entries that are unclear, and deletes obsolete ones (with confirmation) so the catalog stays usable over time.

**Why this priority**: Without browse/rename/delete, saved presets become unmanageable; this completes the minimum viable preset library alongside save/load.

**Independent Test**: Save two presets, rename one, delete the other with confirmation, and verify the catalog and load behavior match the updated set.

**Acceptance Scenarios**:

1. **Given** one or more saved presets, **When** the user opens the preset browser, **Then** each entry is listed by name in a scannable list
2. **Given** a preset in the list, **When** the user renames it to a new unique name, **Then** the catalog shows the new name and loading that name restores the same patch
3. **Given** a preset in the list, **When** the user deletes it and confirms, **Then** it no longer appears and cannot be loaded; if they cancel, it remains
4. **Given** an empty catalog, **When** the user opens the preset browser, **Then** they see an empty state with a clear prompt that they can save the current patch as a preset

---

### User Story 3 - Undo and Redo Graph and Parameter Edits (Priority: P1)

While editing, the designer makes a sequence of changes (add/remove/move boxes, reconnect cables, change parameters, group or ungroup, randomize weights, and similar editor or one-shot sonic actions). They undo one or more steps to reverse mistakes, then redo to restore undone steps, without losing work that came before the undone portion.

**Why this priority**: Session edit history is essential for safe experimentation on dense graphs; it is independently valuable even without presets.

**Independent Test**: Perform at least five distinct edits including one weight randomization, undo through the randomization and confirm prior weights return, redo to restore the randomized state, then make a new edit and confirm redo is cleared.

**Acceptance Scenarios**:

1. **Given** at least one undoable edit since the last history reset, **When** the user invokes Undo, **Then** the plugin returns to the immediately previous editable state
2. **Given** one or more undone steps still available to redo, **When** the user invokes Redo, **Then** the plugin restores the next previously undone state
3. **Given** the user has undone one or more steps, **When** they perform a new edit, **Then** the redo chain is cleared and only the new forward history remains
4. **Given** a continuous adjustment of one control (for example dragging a knob or trackpad), **When** the gesture ends, **Then** that gesture counts as a single undo step rather than many tiny steps
5. **Given** the user triggers weight randomization (or a similar one-shot sonic action), **When** they undo once, **Then** the prior weights/sonic state are restored as a single step; redo restores the post-action state
6. **Given** the user only pans, zooms, or changes selection, **When** they invoke Undo, **Then** those view-only actions are not undone (history is unchanged by them)
7. **Given** audio is playing, **When** the user undoes or redoes, **Then** the restored state applies without stopping host transport and without requiring a restart of the plugin
8. **Given** standard shortcuts (Undo / Redo) and on-screen actions, **When** the user triggers either, **Then** both paths perform the same history operation

---

### User Story 4 - Export and Import Presets for Sharing (Priority: P2)

The designer exports a preset to a portable file to share with a collaborator or back up outside the plugin. Another user (or the same user on another machine) imports that file into their catalog and can load it like any other named preset.

**Why this priority**: Sharing and backup extend the preset library beyond one machine; core save/load (Stories 1–2) already deliver local recall.

**Independent Test**: Export a named preset to a file, remove or ignore the local entry, import the file under a chosen name, and load it successfully.

**Acceptance Scenarios**:

1. **Given** a named preset, **When** the user exports it, **Then** they obtain a single portable file representing that full sonic patch
2. **Given** a valid preset file that embeds a full sonic patch and a unique import name, **When** the user imports it with that name, **Then** a catalog entry appears and loading it restores the same sonic patch
3. **Given** a valid preset file and an import name that already exists in the catalog, **When** the user confirms overwrite, **Then** that entry is replaced; if they cancel, the catalog is unchanged
4. **Given** an invalid or incompatible preset file, **When** the user attempts import, **Then** the action is refused with a clear message and the catalog is unchanged

---

### User Story 5 - Preset Load Interacts Predictably with Undo History (Priority: P2)

After editing with an active undo stack, the designer loads a preset. They can undo that load as one step to return to the pre-load state, then continue editing with a coherent history.

**Why this priority**: Clarifies the relationship between the two capabilities so users are not surprised when switching patches mid-session.

**Independent Test**: Make edits, load a preset, undo once and confirm the pre-load state returns; redo once and confirm the preset state returns.

**Acceptance Scenarios**:

1. **Given** an editable state with prior undo history, **When** the user loads a preset, **Then** the load is recorded as a single undoable step and the loaded name becomes the current preset (not dirty)
2. **Given** a preset was just loaded, **When** the user undoes, **Then** the plugin returns to the state immediately before the load, including the previous current-preset name and dirty flag
3. **Given** the user undid a preset load, **When** they redo, **Then** the loaded preset state and current-preset association are restored again

---

### Edge Cases

- Saving with an empty or whitespace-only name is refused with a clear message
- Loading a preset whose stored content is missing or corrupted is refused with a clear message; current state is left unchanged
- A preset that cannot restore embedded weights or Gold artifacts MUST refuse load (or complete restore) with a clear message rather than silently producing a different sound
- Undo when there is nothing to undo (and redo when nothing to redo) is a no-op with clear disabled affordance in the UI
- Extremely large patches still save/load and undo/redo without freezing the UI beyond a brief busy indication
- Host/DAW project save and restore continue to work independently of the named preset catalog
- Preset management does not replace the user box library: boxes remain reusable components; presets remain full plugin patches
- Concurrent overwrite: if the user confirms overwrite of an existing name (Save As / first save to a colliding name / import to a colliding name), only that entry is replaced
- After load, any undoable edit that changes the patch marks the current preset dirty; successfully Saving the current entry or Save As to a new name clears dirty
- Undoing back to the exact post-load (or post-save) patch state clears the dirty indicator again
- History depth is finite; when the limit is reached, the oldest undo steps drop off and cannot be recovered
- Canvas pan, zoom, and selection changes do not create history steps and do not mark the current preset dirty by themselves

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to save the current full plugin patch (graph structure, connections, parameters, model weights, and Gold/frozen artifacts required for identical sonic recall) as a named preset from inside the plugin
- **FR-001a**: Saved and loaded presets MUST embed the weights and Gold/frozen artifacts required so that successful recall reproduces the same sonic result as at save time (not structure/parameters alone)
- **FR-002**: Users MUST be able to browse named presets and load any selected preset to restore that full sonic patch; on successful load the UI MUST show that name as the current preset (clean / not dirty)
- **FR-002a**: After any undoable edit that changes the patch while a current preset is set, the UI MUST show a dirty/modified indicator for that current preset
- **FR-002b**: Users MUST be able to Save the current dirty preset to overwrite its catalog entry (without choosing a new name), and MUST be able to Save As under a new name; successful Save or Save As clears dirty and updates the current-preset association
- **FR-003**: Users MUST be able to rename and delete presets; delete MUST require confirmation
- **FR-004**: Saving under an existing catalog name that is not the current preset’s own Save target MUST require explicit overwrite confirmation
- **FR-005**: Named presets MUST persist across plugin and host sessions in the user’s local preset catalog
- **FR-006**: Loading a preset MUST NOT stop host playback; transition MUST remain seamless for live use
- **FR-007**: The system MUST provide Undo and Redo for user edits that change the editable patch (graph structure, membership/grouping, connections, box layout positions, parameter/control values, and one-shot sonic actions such as weight randomization)
- **FR-007a**: Weight randomization and similar one-shot actions that change sonic state without restructuring the graph MUST be recorded as a single undo/redo step that restores the prior sonic state
- **FR-007b**: Undo and Redo MUST NOT record view-only editor changes (canvas pan, zoom, or selection); those MUST NOT consume history steps or mark the current preset dirty by themselves
- **FR-008**: Continuous single-control gestures MUST coalesce into one undo step
- **FR-009**: Performing a new edit after Undo MUST clear the redo chain
- **FR-010**: Undo and Redo MUST be available via both standard keyboard shortcuts and explicit UI actions
- **FR-011**: Applying a preset load MUST be recorded as a single undoable history step so the previous patch can be restored with one Undo
- **FR-012**: Users MUST be able to export a named preset to a portable file and import a valid preset file into the catalog; exported files MUST include the same full sonic payload as catalog presets
- **FR-012a**: Importing under an existing catalog name MUST require explicit overwrite confirmation; cancel MUST leave the catalog unchanged
- **FR-013**: Invalid save names, failed loads (including inability to restore embedded weights/Gold artifacts), and invalid imports MUST be refused with clear user-facing messages and MUST leave the current patch and catalog consistent
- **FR-014**: Named preset management MUST be distinct from DAW/host project state save/restore (both MUST continue to work)
- **FR-015**: Named preset management MUST be distinct from the user box library (component reuse vs full-patch recall)
- **FR-016**: Undo history MUST have a documented finite depth; when exceeded, oldest steps are discarded
- **FR-017**: The UI MUST indicate when Undo and Redo are unavailable (nothing to undo/redo)
- **FR-018**: The UI MUST display the current preset name (when set) and whether it is dirty

### Key Entities

- **Preset**: A named, user-managed snapshot of a full sonic plugin patch suitable for later recall and sharing. Attributes: name, saved patch contents (graph, parameters, weights, Gold artifacts), optional metadata such as last-modified time
- **Current Preset**: Session association of the active plugin instance to a catalog name, plus a dirty flag indicating whether the live patch differs from the last loaded/saved contents for that name
- **Preset Catalog**: The user’s collection of named presets available for browse, load, rename, delete, export, and import
- **Edit History**: An ordered session sequence of undoable patch states (or equivalent reverse/forward steps) supporting Undo and Redo
- **History Step**: One logical user change (or coalesced gesture, one-shot sonic action such as weight randomization, or preset load) that can be undone or redone as a unit
- **Plugin Patch**: The complete editable and sonic configuration of one plugin instance (graph + parameters + weights + Gold/frozen artifacts needed for identical recall)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can save a named preset and successfully reload it in a new plugin instance in under 30 seconds of interaction time
- **SC-002**: After loading a previously saved preset, the restored patch matches the saved graph structure, parameter values, and sonic result (including weights/Gold state) with no manual repair required
- **SC-003**: Users can undo at least 20 consecutive distinct edits and redo them in order with correct intermediate states each time
- **SC-004**: A continuous knob or trackpad gesture undoes as a single step in 100% of tested cases
- **SC-004a**: After weight randomization, one Undo restores the immediately prior weights/sonic state in 100% of tested cases
- **SC-005**: Preset load and undo/redo complete their user-visible state update within 1 second for typical patches (under a few hundred boxes) without requiring a plugin restart
- **SC-006**: At least 90% of first-time testers can load a preset, see it as current, edit until dirty, Save to overwrite, and Save As a second name without assistance
- **SC-007**: Host project save/load still restores the session patch correctly after users have also used the named preset catalog in that session
- **SC-008**: Export then import of a preset on a clean catalog yields a loadable entry whose sonic recall matches the original patch

## Assumptions

- “Preset” means a full sonic plugin patch (graph + parameters + weights + Gold artifacts), not a partial parameter bank and not a single box; the existing user box library remains the mechanism for reusable components
- Preset export/import carries the same full sonic payload so shared files recall identically when compatible
- After load or save, the UI tracks a current preset name and a dirty flag; Save overwrites the current entry, Save As creates or overwrites another name with confirmation when colliding
- Import into an existing catalog name uses the same overwrite-confirmation rule as Save As name collisions
- In-plugin user preset catalog is in scope; a curated factory preset bank and MIDI program-change banks are out of scope for this feature
- DAW/host session state persistence already exists and remains the source of truth for project recall; named presets are an additional user-facing catalog
- Undo/redo is per plugin instance and session-scoped (not persisted across plugin close), except that loading a preset remains undoable within the session as one step
- Default undo depth is at least 50 steps unless planning chooses a higher bound for memory reasons
- Training runs and other long-running background jobs are not mid-flight “scrubbed” by undo; when such a job finishes and applies a patch change (for example replacing part of the graph with a finished trained result), that applied change MUST be recorded as a normal single history step (same class as weight randomization)
- Weight randomization and similar one-shot sonic actions are undoable as one history step each
- Undo/redo covers patch-affecting edits only (including box layout positions); canvas pan, zoom, and selection are excluded from history
- Export/import uses a single-file portable representation suitable for sharing between users of this product and MUST carry the same full sonic payload (graph, parameters, weights, Gold artifacts)
- Keyboard shortcuts follow platform conventions (for example Undo / Redo on the host OS)
