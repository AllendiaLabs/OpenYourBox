# Quickstart: Box Property Panel UX

**Feature**: `013-box-property-panel-ux`  
**Date**: 2026-08-28

Manual validation guide for the VST graph editor after implementation. Prefer building the plugin target used for local OpenYourBox development, then run through the scenarios below.

## Prerequisites

- Build OpenYourBox VST (Debug) with existing project CMake/Xcode workflow
- Host that loads the plugin UI (or in-repo editor harness if used)
- Optional: run C++ tests

```bash
# From repo root — adjust to the project’s usual test invocation
ctest --test-dir build -R 'GraphGroup|UserBoxLibrary|EditHistory' --output-on-failure
```

Expected unit coverage (see contracts / data-model): disconnect-all links, reparent into group/root, cycle reject, undo of reparent.

## Scenario A — Slim boxes + Parameters (P1)

1. Add several element types and a group to the canvas.
2. Confirm each box shows **name and pins only** (no inline params / randomize).
3. Click a box → right menu switches to **Parameters**; edit a property; confirm audio/graph behavior updates and the box does **not** resize from the edit.
4. Switch to Info (or another tab), click a different box → tab returns to **Parameters** for that box.
5. Trigger **Randomize** from Parameters (not from the box); confirm weights change and box size stays stable.

## Scenario B — One-click drag (P1)

1. Click empty canvas to clear selection.
2. Press on an unselected box body and drag without releasing → box moves on the first gesture.
3. Drag from a pin → wiring starts; box does not move.

## Scenario C — Project structure navigate (P2)

1. Build a nested group with an inner element; expand **Project structure**.
2. Single-click the inner element → Parameters shows that element (editable).
3. Single-click the group → Parameters for the group; inner canvas does **not** open.
4. Double-click the inner element → view centers on that box on its canvas (camera focus; group is not opened from a non-group row).
5. Double-click the group row → inner canvas opens with the camera fitted to all inner boxes and centred (does **not** jump to the parent canvas).
6. On canvas, double-click the group box → inner canvas opens with the camera fitted to all inner boxes and centred.

## Scenario D — Structure reparent (P2)

1. Wire an element to a neighbor; note the cable.
2. Drag that element’s Project structure row onto another group (highlight rectangle visible).
3. On drop: cables gone; element lives under the target like a new insert; Parameters open; destination canvas focused if needed.
4. Undo → hierarchy and cables restored per history semantics.
5. Attempt to drop a group into its descendant → rejected; no change.

## Scenario E — Library / palette → canvas and structure (P2)

1. Single-click a user library entry → Parameters **read-only**; changing a field does not alter the catalog.
2. Drag library entry to canvas → instance at drop point (existing insert rules).
3. Drag library entry (or nested subpart) onto a Project structure group → instance in that group; selected; Parameters editable.
4. Drag an **element list** item onto Project structure root/group → same add-like-new behavior.
5. Drag element list item onto canvas → existing palette drop behavior.
6. Release drag outside canvas and structure → no insert.

## Scenario F — Group size stability (P3)

1. Note a group box size.
2. Drag the group across the canvas ≥ 20 times; size unchanged.
3. Randomize from Parameters; size unchanged.

## Pass criteria

- Matches acceptance scenarios in `spec.md` user stories 1–5
- Contracts under `contracts/` respected
- Unit tests for graph reparent/disconnect green

## References

- `spec.md` — requirements and clarifications
- `data-model.md` — SelectionContext, mutations
- `contracts/parameters-panel-contract.md`
- `contracts/slim-box-chrome-contract.md`
- `contracts/project-structure-navigation-contract.md`
- `contracts/structure-hierarchy-dnd-contract.md`
