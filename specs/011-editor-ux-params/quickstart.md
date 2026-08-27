# Quickstart: Editor UX & Parameter Flexibility

**Feature**: `011-editor-ux-params`  
**Spec**: [spec.md](./spec.md) · **Plan**: [plan.md](./plan.md)

Manual validation after implementation. Prefer Debug VST build with graph editor visible.

## Prerequisites

- Build/load OpenYourBox plugin in a DAW or host that shows the ImGui graph.
- Familiarity with groups + copies (006) and user box library save/insert.

## 1. Copy-list tiling (P1)

1. Create nested groups with copies `O=2`, `M=2`, `N=2` (P=8) around an element with a multi-value property (e.g. Conv `channels`).
2. Enter `16` → editable shows one value; preview shows eight `16`s; processing uses 16 everywhere.
3. Enter eight distinct values → both field and preview length 8.
4. Enter two values → field length 2; preview length 8 with correct tiling ([copy-list-tiling-contract.md](./contracts/copy-list-tiling-contract.md)).
5. Enter three values → refused; previous list kept.
6. Change outer copies so authored length becomes illegal → parameter flagged invalid; fix list → valid again.

## 2. Keyword `in` (P1)

1. On Linear/Conv, set output `features`/`channels` to `in`.
2. Confirm output pin channels track input.
3. Change upstream channels / parent copies → outputs follow without retyping.
4. Try `in, 32` → refused. Try `in` on gain → refused.

## 3. Project structure & library subpart (P2)

1. Left menu under Library: expand **Project structure**; confirm live hierarchy.
2. Navigate into a nested group from the tree; save that nested box to the user library.
3. In User Library, expand the saved group entry; insert root once and a nested child once.
4. Confirm subpart has no invented external cables; groups insert collapsed per 006.
5. With several entries in one folder (e.g. `Zeta`, `alpha`, `Beta`), confirm they list alphabetically by name (case-insensitive).

## 4. Sticky hierarchy trail (P2)

1. Open Parent → Child → Grandchild via canvas.
2. Click Parent in the trail → Child and Grandchild remain visible/clickable.
3. Click Child → returns in one click.
4. From Parent open a different sibling → prior sticky branch cleared.

## 5. Resizable menus (P3)

1. Drag left palette edge and right panel edge; widths update live.
2. Drag past limits → clamps; canvas remains usable.
3. Continue editing → widths stay put for the session.

## 6. LeakyReLU slope (P3)

1. Place Activation, choose LeakyReLU → `negative_slope` defaults to `0.01`.
2. Set `0.2` → persists after save/reload; freeze/train path uses `0.2` if exercised.
3. Set `-0.1` → refused; prior value kept.

## Automated hooks (when present)

- C++ tests for dividing-set parse/tile and trail prune (see `Tests/` once added in tasks).
- Optional Python unit for `_activation(..., negative_slope=0.2)`.

## Done when

- Scenarios 1–6 match [spec.md](./spec.md) success criteria SC-001…SC-006 / SC-003a.
- No audio glitch from UI-only actions (resize, trail, tree expand).
