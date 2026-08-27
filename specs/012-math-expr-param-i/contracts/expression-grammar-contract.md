# Contract: Expression Grammar

## Purpose

Define the shared infix grammar for Mathematical Expression (`x1`…`xN`) and parameter index expressions (`i`) (FR-003, FR-009, FR-014).

## Lexical

| Token | Form |
|-------|------|
| Number | Decimal literals; scientific notation (`1e-7`, `2.3E+1`) |
| Ident | `[A-Za-z_][A-Za-z0-9_]*` — validated against context allow-list |
| Operators | `+`, `-`, `*`, `^`, `(`, `)` |
| Whitespace | Ignored between tokens |

**Not allowed**: `/`, juxtaposition multiply (`2i`), function calls, commas inside a single expression token.

## Precedence (high → low)

1. `()` grouping  
2. unary `-`  
3. `^` (right-associative)  
4. `*` (left-associative)  
5. `+`, binary `-` (left-associative)

## Contexts

| Context | Allowed idents | Eval bindings |
|---------|----------------|---------------|
| Math Expression | `x1`…`xN` where N = Inputs | Each ident → connected tensor (elementwise) |
| Parameter field | `i` only | Integer slot index |

## Commit policy

| Outcome | Behavior |
|---------|----------|
| Syntax / unknown ident / empty | Refuse commit; clear message; prior authored text kept; invalid string not stored |
| Non-finite eval | Refuse (params: any slot; Math Expression: constant-fold where applicable, else runtime guard at compile) |
| Success | Store authored string; rebuild AST on GUI/compile thread |

## Discovery

UI MUST expose allowed symbols/operators via placeholder and/or tooltip on Math Expression `expression` and on numeric fields that accept `i` (FR-014).

## Implementation anchors

- New shared module under `OpenYourBox/Source/graph/` (e.g. `ExpressionParser`)
- Call sites: `NodeGraph` property commit, `NodeRenderer` deactivate-after-edit, `LiveGraphEngine` compile

## Non-goals

- Division, comparisons, boolean ops, named functions (`sin`, `pow`, …)
- Implicit multiplication
