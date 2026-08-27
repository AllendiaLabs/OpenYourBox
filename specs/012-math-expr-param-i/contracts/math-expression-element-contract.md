# Contract: Mathematical Expression Element

## Purpose

Palette element that evaluates a user expression over Utility-style inputs `x1`…`xN` so RAVE mod_sigmoid is composed with existing Sigmoid (FR-001–FR-006, FR-016–FR-017).

## Identity

| Item | Value |
|------|-------|
| UI label | `Math Expression` |
| Persisted type | `math_expression` |
| Palette | Yes |
| Dedicated mod-sigmoid | **No** |

## Properties

| Key | Kind | Default | Notes |
|-----|------|---------|-------|
| `inputs` | integer | `1` | Min 1; rebuilds pins `x1`…`xN` |
| `expression` | string | `x1` (or product-chosen default) | Grammar: [expression-grammar-contract.md](./expression-grammar-contract.md) |

## Pins

- Inputs: labelled `x1`, `x2`, … `xN` (N = `inputs`).
- One output; shape from referenced connected inputs (below).
- Rebuild pattern mirrors Utility `setMixerInputCount` with `x` prefix instead of `in `.

## Wiring & shape (FR-004, FR-017)

| Rule | Behavior |
|------|----------|
| Channel compatibility | Same as Utility add/multiply: equal **or** either width is 1 (scalar broadcast) |
| Output channels | `max` of referenced connected inputs’ channels |
| Rate / bands | Must match across referenced inputs; else refuse cable + tooltip |
| Required connections | Only pins **referenced** by current `expression`; unused configured pins may be empty |
| Inputs reduction | Refuse if `expression` still references `xK` with K &gt; new N (FR-016) |

## Runtime

| Path | Behavior |
|------|----------|
| Live | Prepared AST → elementwise LibTorch ops; broadcast like Utility multiply |
| Freeze / train | Explicit `math_expression` builder; identical arithmetic; **no** silent skip |

## Invalid expression

Refuse on commit (FR-006): prior expression kept; document does not store invalid string; clear message.

## Composition target

Activation (Function = Sigmoid) → Math Expression `x1` with `2 * x1^2.3 + 1e-7` realizes mod_sigmoid (FR-005, SC-001).

## Implementation anchors

- `GraphTypes.h` `NodeType`, `nodeTypeName` / `nodeTypeFromName`
- `NodeGraph::makeNode`, pin rebuild, `connect` validation
- `NodeRenderer` palette + property UI
- `LiveGraphEngine` compile/execute
- `Backend/freeze_worker.py`, `Backend/train_worker.py`

## Non-goals

- Concatenate mode / Utility Mode property
- Hardcoded mod-sigmoid activation choice
- Audio-thread parsing
