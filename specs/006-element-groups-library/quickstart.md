# Quickstart: Element Groups & User Box Library

Manual / integration validation for `006-element-groups-library`. Prefer a Debug/Release plugin build loaded in a host (or the project’s usual test harness).

## Prerequisites

- Branch `006-element-groups-library` built (`OpenYourBox` VST3/AU).
- Host session with one OpenYourBox instance.
- Graph with several non-I/O elements (e.g. Conv1D, Activation, Linear) plus Audio In/Out on the canvas.

## 1. Create nested groups

1. Select ≥2 processing nodes (no Audio I/O) → context **Group**.
2. Rename the group.
3. Inside it, select ≥2 members → **Group** again (subgroup).
4. Save host/project state; reload plugin/session.
5. **Expect**: hierarchy, names, membership, and links restored (`SC-001`, FR-004).

## 2. Refuse Audio I/O

1. Include Audio Input or Output in a selection → **Group** / **Save to Box Library**.
2. **Expect**: refused with clear message; graph/library unchanged (FR-001a).

## 3. Open / close group canvases

1. Double-click a group box → interiors become the canvas; parent boxes disappear; breadcrumb shows `Graph > Group`.
2. Click **Graph** in the breadcrumb → group is a box again with external pins; name visible.
3. Open a nested group; breadcrumb shows `Graph > Parent > Subgroup`; click Parent to return.
4. Process audio before/after opening a group.
5. **Expect**: UI matches FR-005–007 via scoped canvases; audible output unchanged (`SC-003`); focused canvas survives save/reload.

## 4. Freeze per member

1. Select a group containing M freezable live members → **Freeze Selection**.
2. **Expect**: up to M individual Gold members; group container remains; **not** one BlackBox for the whole group (`SC-008`, FR-001b).
3. If some members already Gold, they are skipped with feedback.

## 5. Group copies (N blocks)

1. Build a group whose external I/O can chain in series; set **copies** N from 1 → 3.
2. **Expect**: the canvas still shows one chain of elements; DSP unrolls three independent copies in series; randomizing/editing copy slots on a member does not share weights (`SC-009`, FR-017a).
3. Set N → 4; **Expect**: new trailing copy slots clone the previous last (`FR-017c`).
4. On a non-chainable group, attempt N → 2; **Expect**: refuse/clamp with message; no extra UI nodes (`SC-010`).
5. Save/reload with N &gt; 1; **Expect**: N and per-element copy slots restored (`SC-011`).
6. Double-click the group to open its canvas; drag an element onto a group box (highlight) to add; **Remove from Group** lifts a member to the parent canvas. The map shows only the focused canvas (group members when a group is open).

## 6. Save and place from Box Library

1. Configure one element’s parameters (and seed/weights if applicable) → **Save to Box Library** with name `E1`.
2. Save a nested group (optionally with N &gt; 1) as `G1`.
3. With multi-select of siblings (no single group target), confirm save is disabled/refused.
4. Open **Boxes** / Box Library list — see `E1` (element) and `G1` (group).
5. Clear or use empty area of graph; place both.
6. **Expect**: parameters match (including N and copies if saved); `G1` appears with root + nested groups **collapsed**; originals in library unchanged (`SC-004`, FR-012a, FR-017e).
7. Quit host; reopen; library still lists `E1`/`G1` (FR-014).

## 7. Library manage

1. Rename, overwrite-with-confirm, delete-with-confirm.
2. **Expect**: catalog updates; deleted entries not placeable (FR-013).

## Automated hooks (when implemented)

- Unit: group cycle refusal; ValueTree round-trip for groups including `copies`; library index CRUD.
- Unit/integration: insert assigns new IDs; collapsed flags forced true on group insert; setCopies legal/illegal paths.
- Freeze: selection expansion yields M freeze targets for M freezable members across all copies in a group.

## References

- `data-model.md`
- `contracts/group-editor-ui-contract.md`
- `contracts/group-copies-contract.md`
- `contracts/user-box-library-contract.md`
- `contracts/freeze-per-member-contract.md`
