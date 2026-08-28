# Contract: Parameters Panel

## Purpose

Right-menu **Parameters** tab for live editing and read-only library inspect (FR-002, FR-003, FR-004, FR-007, FR-008).

## Tab placement

- Add **Parameters** to `PluginEditor` `SideTabs` alongside Info, Library (training), Capture, Train, Presets.
- Selecting a live box (canvas or Project structure) **MUST** switch the active tab to Parameters even if another tab was active.
- Selecting a library entry for inspect **MUST** switch to Parameters in read-only mode.

## Live binding

- Source: primary selected node or group (`SelectionContext::Live`).
- Content: all properties previously editable on the box body, plus randomize/reset actions for applicable types (FR-004).
- Edits use existing property change callbacks and undo gestures (`documentChanged` / patch gestures).
- Group name and copy-count (repeats) editors that leave the canvas live in this tab (or equivalent Parameters sections) so the canvas body stays non-stealing.

## Library inspect binding

- Source: `SelectionContext::LibraryInspect` (entry id + optional nested subpart).
- Display the same property fields as live where snapshot data allows.
- Controls **MUST** be disabled or non-committing; catalog files unchanged.
- No path from this mode into `UserBoxLibrary` property write/save APIs.

## Multi-selection

- Show a concise multi-select state; do not offer full Parameters editing until a single live box is selected (spec assumption).

## Non-goals

- Merging Parameters into Info analysis UI
- Editing library catalog definitions from the panel
- Changing Train/Capture/Presets tab behavior
