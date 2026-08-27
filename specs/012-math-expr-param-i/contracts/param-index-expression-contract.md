# Contract: Parameter Index Expressions

## Purpose

Allow copy-expanded numeric parameters to accept expressions over `i` (FR-007–FR-015), preserving 011 dividing-set list rules.

## Index `i`

| Situation | `i` value |
|-----------|-----------|
| Not in a group, or expanded length P = 1 | `0` |
| Expanded list length P | Slot index `0 … P−1` in the same order as copy-expanded previews |

## Authored form

- Single expression containing `i` → evaluate once per expanded slot with that slot’s `i`.
- Short lists (length L ∈ dividing set): each entry may be a literal or an expression; after tiling, `i` is the **expanded** slot index (US3.4).
- Constant expressions without `i` (e.g. `2^3`) resolve to the same value for applicable slots.
- Persist authored strings so P changes re-evaluate without rewrite (FR-012).

## Grammar

See [expression-grammar-contract.md](./expression-grammar-contract.md). Parameter context allows ident `i` only (not `x1`…).

## Commit / refuse

| Case | Behavior |
|------|----------|
| Bad syntax / unknown symbol / non-finite | Refuse; prior authored kept; invalid string not stored |
| Integer-typed property, any slot non-integer | Refuse; message that integer required; no round/floor (FR-015) |
| Length L ∉ dividing set | Existing 011 refuse/flag rules |

## UI

- Editable field shows authored form (expression or short list).
- Read-only expanded preview shows resolved numeric P values (extend 011 preview).
- Deactivate-after-edit: refuse restores buffer from last good authored text.

## Eligible fields

Numeric (integer/real) properties that already participate in copy-expanded value lists. Exclude: `inputs`/`ports`-style structural counts if today’s `propertySupportsRepeatValueList` excludes them; booleans; enums; Math Expression’s own `expression` string (different contract).

## Implementation anchors

- Extend `parsePropertyRepeatList` / commit helpers in `GraphTypes` / `NodeGraph`
- `NodeRenderer` property edit path
- Re-eval on `setGroupCopies` / nest changes alongside 011 tiling

## Non-goals

- Multi-dimensional indices `(o,m,n)` instead of flattened `i`
- Expressions on non-numeric properties
- Storing invalid expressions as flagged document state
