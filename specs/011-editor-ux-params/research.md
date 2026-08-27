# Research: Editor UX & Parameter Flexibility

**Feature**: `011-editor-ux-params` | **Date**: 2026-08-27

## R1 — Nested copy-list dividing lengths

**Decision**: Extend `parsePropertyCopyList` / commit path to accept any length L in the dividing set induced by ancestor copy counts from innermost outward: for nest `[…, O, M, N]` with `P = O×M×N`, allow `L ∈ {1, N, M×N, P}` (generalize: suffix products of the copy-count vector). Persist **authored** vectors at length L; derive expanded P by tiling along outer axes. UI: editable authored CSV + read-only P preview. On `setGroupCopies` / nest change: if L still valid → re-tile; else mark property invalid (keep authored bytes, clear expanded-for-runtime until fixed).

**Rationale**: Spec clarifications (session 2026-08-27). Current code only allows L∈{1,P} (`GraphTypes.h` `parsePropertyCopyList`) and `ensurePropertyCopyCount` silently pads/truncates — contradicts “flag invalid.”

**Alternatives considered**:
- Always store P only — rejected (loses short authoring).
- Accept any divisor of P — rejected (spec wants nest-axis suffixes, not arbitrary divisors like M alone when axes are O,M,N).

## R2 — Preserve keyword `in`

**Decision**: Reserved lowercase token `in` in integer (and list) dim/channels/features fields that support binding. Resolve to the paired input shape’s corresponding dimension after copy expansion. All-`in` lists (and short all-`in` that tile per R1) allowed; mixing `in` with numbers refused. Store binding flag / authored tokens separately from resolved ints used by pins.

**Rationale**: Clarified token; Shape Integrity needs concrete ints before cable checks. Activation-like passthrough already inherits input — `in` is for fields that today force a numeric property (e.g. Linear `features`, Conv `channels`).

**Alternatives considered**: `same`, `preserve`, `*` — rejected by clarify. Auto-infer without keyword — rejected (user must opt in).

## R3 — Project structure + library subtrees

**Decision**: Add collapsible **Project structure** ImGui section in the left palette **under** User Library. Tree mirrors live `NodeGraph` hierarchy (groups as folders, elements as leaves). Click navigates (`setCanvasFocus`); context/save uses existing per-box `exportBox` on the targeted id (already supports nested ids). Library entries with group payloads expose an **expandable member tree**; DnD/insert may target root entry id **or** a nested node path within the payload (`importBox` of that subtree only; no external cables). **Within each folder, entries are sorted alphabetically by display name (case-insensitive)**; nested member rows SHOULD sort by label when expanded.

**Rationale**: Clarify Option A. `exportBox` already accepts nested ids; gap is UI + sub-insert API.

**Alternatives considered**: Library-only expand (clarify B) — rejected. Separate top-level window — rejected (must live under Library on left).

## R4 — Sticky hierarchy trail

**Decision**: Extend breadcrumb state beyond `focusedGroupId` ancestor chain: remember the last visited **descendant spine** under the current focus. When navigating up, keep those children visible/clickable. Opening a different sibling branch clears the prior spine and starts a new one.

**Rationale**: Spec US4; current `renderScopeBreadcrumb` is ancestor-only.

**Alternatives considered**: Persist full visit history forest — rejected (clutter; spec is branch spine until branch change). Persist across sessions — deferred (session-only v1 unless viewport already serializes focus).

## R5 — Resizable side menus

**Decision**: ImGui splitter between left palette and graph canvas (replace fixed `200.f`); splitter between graph area and right Info panel (replace fixed `~332` / `-340`). Clamp menu min/max and enforce minimum canvas width. Persist widths in session; write through existing UI prefs if present, else session-only (per spec assumption).

**Rationale**: Spec FR-002; exploration confirmed fixed widths in `NodeRenderer` / `PluginEditor`.

**Alternatives considered**: Only left resizable — rejected (spec both sides). Percentage-only layout — optional later; absolute px with clamps is enough for v1.

## R6 — LeakyReLU negative slope

**Decision**: Add real `NodeProperty` (e.g. `negative_slope`) on activation elements when activation is LeakyReLU (always present but only meaningful for that choice — or always editable and ignored for other activations; prefer **visible when LeakyReLU selected**). Default `0.01`, range `[0, 1]`, refuse OOR (no clamp). Replace hardcoded `0.01` in `LiveGraphEngine`, `TCNModel`, `train_worker._activation`, `freeze_worker`.

**Rationale**: Spec FR-001; RAVE uses 0.2 via discriminator path today but element should drive slope. Mirror `gain` property pattern.

**Alternatives considered**: Global preference — rejected. Clamp OOR — rejected by clarify. Always show for all activations — weaker UX; hide unless LeakyReLU.

## R7 — Tiling orientation

**Decision**: Treat copy-count vector as outermost → innermost when multiplying (consistent with `effectiveCopyCount` ancestor product). Short list of length N tiles across all outer axes; length M×N tiles across outermost O only. Document ordering in `copy-list-tiling-contract.md` with a worked O=2,M=2,N=2 example.

**Rationale**: Must match user’s “N, M×N, M×N×O” examples and existing copy slot indexing.

## Open items deferred to tasks (not blocking plan)

- Exact ImGui splitter widget vs drag on `InvisibleButton` edge.
- Whether Project structure default is collapsed (either OK per edge cases).
- Schema version bump for library payloads if nested insert needs stable member ids in index (prefer path-in-snapshot over index schema change).
