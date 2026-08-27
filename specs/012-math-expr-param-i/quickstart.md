# Quickstart: Math Expression Node & Parameter Index Expressions

**Feature**: `012-math-expr-param-i`  
**Spec**: [spec.md](./spec.md) · **Plan**: [plan.md](./plan.md)

Manual validation after implementation. Prefer Debug VST build with graph editor visible.

## Prerequisites

- Build/load OpenYourBox with graph editor.
- Familiarity with Activation/Sigmoid, Utility Inputs, and nested group copies (006/011).

## 1. Mod_sigmoid composition (P1)

1. Place **Activation**, set Function = Sigmoid.
2. Place **Math Expression**; confirm Inputs defaults to 1 and pin `x1` exists; confirm no “mod sigmoid” palette entry.
3. Set expression to `2 * x1^2.3 + 1e-7`; wire Sigmoid → `x1` → Audio Out (or analysis).
4. Confirm audible/analysis behavior matches expected mod_sigmoid shaping ([math-expression-element-contract.md](./contracts/math-expression-element-contract.md)).
5. Enter invalid expression `2 * x1 / 3` → refused; prior expression restored.

## 2. Multi-input Utility-style shapes (P1)

1. Set Math Expression Inputs = 2; pins `x1`, `x2` appear.
2. Expression `x1 * x2`: connect multi-channel to `x1` and width-1 to `x2` → cable accepted (scalar broadcast); output channels = max.
3. Connect two incompatible multi-channel widths (neither 1) → cable refused with tooltip.
4. Expression only `x1` with Inputs = 2 and `x2` empty → graph still valid ([FR-017](./spec.md)).

## 3. Parameter `i` expressions (P1)

1. Group with copies N = 8 around Conv (or other copy-list numeric param).
2. Enter `2*i+1` → editable keeps expression; preview shows eight resolved integers; processing uses those values.
3. Outside a group, enter `i+3` → resolves as `3` (`i=0`).
4. On an integer field, enter `i^0.5` (non-integer for some slots) → refused; prior value kept.
5. Change copies N → same expression re-evaluates for new P without retyping ([param-index-expression-contract.md](./contracts/param-index-expression-contract.md)).

## 4. Freeze / train smoke (when exercising Gold)

1. Freeze Sigmoid + Math Expression subgraph (or include in a freezable chain).
2. Confirm compile succeeds and Gold node matches live formula (not silent passthrough).

## Automated hooks (when present)

- C++: expression grammar, Math Expression broadcast, `i` expand/refuse (`Tests/`).
- Python: freeze/train `math_expression` builder unit.

## Done when

- SC-001…SC-005 scenarios in [spec.md](./spec.md) pass under guided check.
- Contracts above satisfied without storing refused strings.
