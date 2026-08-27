# Data Model: Math Expression Node & Parameter Index Expressions

**Feature**: `012-math-expr-param-i` | **Date**: 2026-08-28  
**Extends**: `specs/006-element-groups-library/data-model.md`, `specs/011-editor-ux-params/data-model.md`

## Entities

### ExpressionAst (shared)

| Field | Type | Notes |
|-------|------|--------|
| Root | binary / unary / literal / ident node | Infix AST after successful parse |
| Operators | `+`, `-`, `*`, `^`, unary `-`, grouping `()` | `^` right-associative |
| Literals | float | Includes scientific notation |
| Idents | string | Context-validated (`x1`…`xN` or `i`) |

**Relationships**: Produced by shared parser; consumed by Math Expression runtime and parameter slot evaluation. Not required to persist as AST — persist source string; rebuild AST on load/commit.

**Validation**:
- Parse success required before commit.
- Eval must be finite for every required binding.
- Integer parameter context: each result must be an integer (exact whole number within float tolerance defined in contract).

### MathExpressionNode (extends `GraphNode`)

| Field | Type | Notes |
|-------|------|--------|
| `type` | `math_expression` | Persisted type string |
| `inputs` | int ≥ 1 | Default 1; rebuilds pins |
| Pins | `x1`…`xN` | Labels; N = `inputs` |
| `expression` | string | Authored formula; legal idents ⊆ configured pins |
| Derived AST | ExpressionAst | GUI/compile cache; not authoritative store |

**Relationships**: Processing element; wires like Activation/Utility. Freeze/train includes full properties + expression string.

**Validation**:
- Expression may only reference `xK` with `K ≤ inputs`.
- Reducing `inputs` refused if expression still references removed pins (FR-016).
- Only referenced pins need connections (FR-017).
- Connect: Utility add/multiply channel broadcast + rate/band match (FR-004).

### ParameterIndexExpression (extends authored numeric property)

| Field | Type | Notes |
|-------|------|--------|
| Authored tokens | list of strings/literals | Length L ∈ dividing set; each token literal or `i`-expression |
| Expanded values | vector length P | Derived: eval token (after tiling) at `i = slotIndex` |
| `i` | int | `0…P−1`; always `0` if P=1 or ungrouped |

**Relationships**: Applies to numeric properties that already participate in copy-expanded lists (011). Non-numeric / enum / boolean unchanged.

**Validation**:
- Grammar + finite eval for every expanded slot.
- Integer properties: refuse non-integer results (FR-015).
- Invalid commit: refuse; do not store invalid string; keep prior authored form (FR-013).
- On nest/`copies` change: re-eval same authored expressions for new P (FR-012); dividing-set invalidation rules from 011 still apply to list **length**.

### ExpressionGrammar (product documentation entity)

| Field | Notes |
|-------|--------|
| Operators | `() + - * ^` |
| Math symbols | `x1`…`xN` |
| Param symbols | `i` |
| Discovery | Tooltip / placeholder (FR-014) |

## State transitions

```text
[edit buffer] --commit parse OK--> [authored stored] --compile--> [AST prepared]
[edit buffer] --commit parse FAIL--> [refuse; buffer restored; document unchanged]

[authored i-expr] --copies/P change--> [re-eval slots 0..P'-1]
[authored i-expr] --length L no longer dividing--> [011 copyListInvalid path]
```

## Persistence notes

- Documents without `math_expression` or string properties load as today.
- Math Expression default on insert: `inputs=1`, expression suitable for passthrough or mod_sigmoid composition (exact default string in tasks).
- Presets/undo capture expression strings via existing graph snapshot paths once properties serialize.
