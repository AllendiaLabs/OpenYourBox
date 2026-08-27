# Research: Math Expression Node & Parameter Index Expressions

**Feature**: `012-math-expr-param-i` | **Date**: 2026-08-28

## R1 — Shared expression grammar

**Decision**: Implement one C++ lexer/parser/evaluator for infix expressions supporting numeric literals (decimals + scientific `1e-7`), identifiers, and operators `()`, `+`, `-` (binary and unary), `*`, `^` (right-associative). Context binds identifiers: Math Expression → `x1`…`xN` (N = Inputs); parameters → `i` only. Reject unknown identifiers, `/`, functions, and empty/whitespace-only strings. Non-finite results (`NaN`/`Inf`, including `0^0` if produced) are invalid.

**Rationale**: Spec FR-003/009/014; clarify session (refuse invalid; same operator set for both contexts).

**Alternatives considered**:
- Embed Lua/exprtk — rejected (dependency, audio-thread risk, overkill for tiny grammar).
- Separate parsers for node vs params — rejected (drift vs FR-014).
- Add division/`pow()` function syntax — rejected (out of scope).

## R2 — Math Expression element (Utility-style Inputs, pins `x1`…`xN`)

**Decision**: New `NodeType` persisted as `"math_expression"` (label “Math Expression”). Properties: `inputs` (integer ≥ 1, default 1) rebuilding pins labelled `x1`…`xN` (same rebuild loop as Utility’s `setMixerInputCount`, different label prefix); `expression` (string, default e.g. `x1` or `2 * x1^2.3 + 1e-7` placeholder per UX). No Mode/concatenate. Connect/shape: reuse Utility add/multiply broadcast (`channelsAreBroadcastCompatible`); output channels = max of **referenced** connected inputs; rate/bands must match. Only pins named in the expression must be connected (FR-017). Palette entry; no mod-sigmoid element.

**Rationale**: Clarify (Utility Inputs; `x1`…; unused pins optional; Utility broadcast including scalar×vector).

**Alternatives considered**:
- Unary-only `x` — rejected by clarify.
- Utility pin labels `in N` — rejected (spec/`x1` in user formula).
- Hardcoded mod_sigmoid activation — rejected (framework principle).

## R3 — Live + freeze/train execution

**Decision**: On property commit / graph compile (GUI thread), parse expression to an AST (or bytecode). Live engine evaluates elementwise with LibTorch (`add`/`mul`/`pow`/`neg`) and existing channel broadcast helpers. Freeze/train workers get an explicit `math_expression` builder that reconstructs the same ops in PyTorch — **must not** rely on RAVE graph `else` passthrough (silent skip).

**Rationale**: Constitution dual-engine + zero audio-thread alloc; current freeze_worker raises on utility and train RAVE builder skips unknowns.

**Alternatives considered**:
- Interpret string every buffer — rejected (alloc/parse on audio thread).
- TorchScript `torch.jit.script` from string — rejected (unsafe/fragile for user strings).

## R4 — Parameter `i` expressions + storage

**Decision**: Extend authored copy-list / scalar numeric commit so each token may be a literal **or** an expression over `i`. Dividing-set length rules from 011 unchanged. Evaluation: for expanded slot `k` in `0…P−1`, bind `i=k` (ungrouped or P=1 → `i=0`). Persist authored expression text (not only expanded numbers) so P changes re-eval (FR-012). Integer-typed fields: result must be whole number for every required slot or **refuse** commit. UI: on `IsItemDeactivatedAfterEdit`, refuse restores buffer from last good authored form (same as 011 invalid lists).

**Rationale**: Spec US2/US3 + clarify refuse paths; research shows no expression support and JSON today lacks string property values.

**Alternatives considered**:
- Expand `i` to numbers on commit and discard formula — rejected (FR-012).
- Flag-invalid while storing bad string — rejected by clarify.
- Round non-integers — rejected by clarify.

## R5 — Property model for strings

**Decision**: Extend `NodeProperty` / JSON serialization to carry authored strings where needed (`expression` on Math Expression; per-slot or whole-field authored formula text on numeric params). Prefer a dedicated `PropertyKind` or `stringValue` / authored-token list that round-trips ValueTree and `toJson`/`fromJson` without breaking older documents (missing → defaults).

**Rationale**: Current kinds are integer/choice/readOnly/real; JSON emits int + optional float only — blocking persistence of expressions.

**Alternatives considered**:
- Encode expression in unused float bits — rejected.
- Side-channel only in UI session — rejected (presets/freeze need document).

## R6 — Implicit multiplication / `2i`

**Decision**: Require explicit `*` (e.g. `2*i`, not `2i`). Refuse `2i` as unknown/invalid tokenizing.

**Rationale**: Spec lists operators explicitly; keeps lexer simple; consistent refuse UX.

**Alternatives considered**: Juxtaposition multiply — deferred (not requested; ambiguous with scientific notation).
