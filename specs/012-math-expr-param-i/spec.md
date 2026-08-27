# Feature Specification: Math Expression Node & Parameter Index Expressions

**Feature Branch**: `012-math-expr-param-i`

**Created**: 2026-08-28

**Status**: Draft

**Input**: User description: "I want to implement rave's mod_sigmoid: 2 * sigmoid(x1)^2.3 + 1e-7, but oyb is meant to be a general framework so I don't want to hardcode it. I want a new mathematical expression node to combine with the implemented sigmoid activation. Also implement this: parameters should accept simple expressions of i, i being the index in the repeats list, or if 1 repeats or not in group: i is always 0. accept ()+-*^"

## Clarifications

### Session 2026-08-28

- Q: When an `i`-expression yields a non-integer for a parameter that must be a whole number (e.g. channels or kernel size), how should the product handle it? → A: Refuse the commit; keep the prior valid value; show a clear message that an integer is required (no silent rounding or truncate)
- Q: When the user commits an invalid expression (bad syntax, unknown symbol, or empty) on a parameter field or on the Mathematical Expression element, should the product refuse the edit entirely or accept the string and mark it invalid? → A: Refuse commit; prior valid expression/value stays; clear error message; document never stores the invalid string
- Q: Should the Mathematical Expression element support only one signal input (`x`), or multiple named signal inputs in this version? → A: Like Utility: a parameter chooses the number of inputs; pins labelled `x1`, `x2`, … (expression references those names)
- Q: When Mathematical Expression uses two or more inputs, how must their shapes relate for the graph to stay legal? → A: Same as Utility add/multiply: channels compatible if equal or either is 1 (scalar broadcast); output channels = max; temporal rate and band count must still match; incompatible mixes refuse the cable
- Q: If Inputs is greater than the number of symbols used in the expression (e.g. Inputs = 2 but expression only mentions `x1`), must unused pins still be connected? → A: Only expression-referenced pins must be connected; unused configured pins may stay empty

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compose RAVE-style mod_sigmoid without a hardcoded node (Priority: P1)

A designer wants the RAVE `mod_sigmoid` behavior (`2 * sigmoid(·)^2.3 + 1e-7`) but OpenYourBox must stay a general framework. They place the existing Sigmoid activation, then a Mathematical Expression element wired after it, set **Inputs** to 1 (pins labelled `x1`, like Utility’s input-count control), and enter an expression such as `2 * x1^2.3 + 1e-7`. The cascade matches the intended nonlinearity. No dedicated “mod sigmoid” element is added to the palette. When they need to combine several streams, they raise Inputs and use `x1`, `x2`, … in the same expression.

**Why this priority**: Unlocks faithful RAVE-style stacks while keeping the product composable instead of growing one-off activation variants.

**Independent Test**: Build Sigmoid → Math Expression (Inputs = 1) with expression `2 * x1^2.3 + 1e-7`, drive a known input, and confirm the cascade matches the expected mod_sigmoid curve; set Inputs ≥ 2 and confirm pins `x1`…`xN` appear and are usable in the expression; confirm the palette has no separate mod-sigmoid element.

**Acceptance Scenarios**:

1. **Given** the element palette, **When** the user browses available elements, **Then** they can add a Mathematical Expression element and there is no dedicated mod-sigmoid (or equivalent one-off) element
2. **Given** a Sigmoid element feeding Mathematical Expression input `x1` (Inputs = 1) whose expression is `2 * x1^2.3 + 1e-7`, **When** the graph processes a signal, **Then** the Math Expression output matches applying that formula to the Sigmoid output (within ordinary numeric tolerance)
3. **Given** a Mathematical Expression element with a valid expression using configured input symbols `x1`…`xN`, constants, and operators `()`, `+`, `-`, `*`, `^`, **When** the referenced inputs are connected with Utility-compatible shapes (equal channels or either side width 1), **Then** the expression is evaluated element-wise with scalar broadcast where applicable and the output channel count is the max of the referenced inputs’ channel counts (rate and bands unchanged / matching)
4. **Given** a Mathematical Expression element with Inputs ≥ 2, **When** the user connects inputs whose channels are incompatible under Utility add/multiply rules (not equal and neither is 1), or whose temporal rate or band count disagree, **Then** the cable is refused with a clear tooltip
5. **Given** a Mathematical Expression element, **When** the user changes the Inputs count parameter, **Then** input pins are labelled `x1`, `x2`, … up to that count (Utility-style), and only those symbols are legal in the expression
6. **Given** a Mathematical Expression element, **When** the user enters an invalid expression (syntax error, unknown symbol, empty, or a symbol for an input that is not configured), **Then** the commit is refused, the prior valid expression is kept (document does not store the invalid string), and a clear message explains the error
7. **Given** Sigmoid → Math Expression configured as above, **When** the subgraph is frozen or left live, **Then** the composed behavior remains available under the same dual-engine rules as other processing elements
8. **Given** Mathematical Expression with Inputs = 2 and expression that only references `x1`, **When** only `x1` is connected and `x2` is empty, **Then** the graph remains valid for processing (unused configured pins may stay empty)

---

### User Story 2 - Author per-copy parameter values with expressions of `i` (Priority: P1)

A designer works inside a group with multiple copies (or nested groups whose expanded parameter list has length P). Instead of typing every numeric slot by hand, they enter a simple expression that uses `i`—the index in that repeats/copies list. Each expanded slot evaluates the expression with its own `i`. When the element is not in a group, or the group has a single copy (P = 1), `i` is always `0`.

**Why this priority**: Nested RAVE-style stacks need systematically varying parameters (channels, dilations, gains, etc.); hand-listing P values does not scale.

**Independent Test**: In a group with N > 1 copies, set a numeric parameter to an expression such as `2*i+1` or `0.5^i`; confirm slot k uses the value of the expression at `i = k`. Outside a group (or N = 1), confirm `i` evaluates as `0`.

**Acceptance Scenarios**:

1. **Given** an element not inside a group (or inside a group with copies = 1), **When** the user sets a numeric parameter to an expression containing `i` (e.g. `i+1`, `2*i`), **Then** `i` is treated as `0` and the resolved value is that expression at `i = 0`
2. **Given** a group with copies N > 1 and an element parameter whose expanded length is P = N (or the nested product P from ancestor copies), **When** the user enters a single expression using `i`, **Then** each expanded slot k (k from 0 to P−1) resolves to the expression evaluated at `i = k`
3. **Given** nested groups whose expanded parameter list length is P, **When** the user authors an `i`-expression, **Then** `i` is the 0-based index in that same expanded repeats list order used for copy-expanded parameter previews
4. **Given** a parameter field, **When** the user enters an expression using only `i`, numeric literals, and operators `()`, `+`, `-`, `*`, `^`, **Then** the expression is accepted if syntactically valid and yields a finite number for every required slot
5. **Given** an invalid parameter expression (bad syntax, disallowed symbols, non-finite result for some slot), **When** the user commits it, **Then** the commit is refused, prior valid values remain, a clear message is shown, and the invalid string is not stored in the document
6. **Given** an authored `i`-expression that is valid for the current P, **When** ancestor copy counts change so P changes, **Then** the same expression is re-evaluated for the new indices `0 … P'−1` (expression form is kept; values update with new length)
7. **Given** a parameter that must be a whole number (e.g. channels, kernel size), **When** the user commits an expression that yields a non-integer for any required slot, **Then** the commit is refused, the prior valid value is kept, and a clear message states that an integer is required

---

### User Story 3 - Reuse expression grammar for constants and mixed lists (Priority: P2)

A designer still enters plain numbers where no variation is needed, and may mix literal numbers with the existing short-list / tiling rules. Expressions and literals share the same operator vocabulary so learning one grammar covers both the Math Expression element and parameter fields.

**Why this priority**: Consistency lowers errors; literal entry must remain as easy as today.

**Independent Test**: Enter plain `3`, then `2^3`, then a short list of literals under dividing-length rules; confirm behavior matches numeric entry plus expression evaluation without breaking existing list tiling.

**Acceptance Scenarios**:

1. **Given** a numeric parameter, **When** the user enters a plain number (integer or decimal, including scientific form such as `1e-7` where literals are allowed), **Then** behavior matches today’s numeric entry (including short-list tiling rules where applicable)
2. **Given** a numeric parameter, **When** the user enters a constant expression without `i` (e.g. `2^3`, `(1+2)*0.5`), **Then** every applicable slot resolves to that constant result
3. **Given** dividing-length short lists from prior parameter-list flexibility, **When** list entries are literals or constant expressions (no `i`), **Then** tiling/expansion rules are unchanged
4. **Given** a short authored list that includes an `i`-expression in one or more entries, **When** the list is expanded by tiling, **Then** each materialized slot evaluates `i` as that slot’s expanded index (not the index inside the short authored list alone)

---

### Edge Cases

- Empty expression, whitespace-only, or only operators → refuse commit; clear message; prior valid expression kept; invalid string not stored
- Unknown identifiers other than configured `x1`…`xN` (Math Expression) or `i` (parameters) → refuse commit; same as above
- Expression references `xK` when Inputs &lt; K → refuse commit
- Reducing Inputs so a previously valid expression still references a removed pin → refuse the Inputs change or refuse until the expression is edited first (document stays consistent; no dangling symbols)
- Division and other operators beyond `() + - * ^` → refuse commit (not accepted in this feature)
- Unary minus (e.g. `-i`, `2*-3`, `-(1+2)`) → allowed as part of `-`
- Exponentiation associativity: right-associative (`2^3^2` = `2^(3^2)`); document in assumptions
- `0^0` or overflow/NaN/Inf from extreme exponents → refuse commit (invalid result for that slot / element); clear message; prior valid kept
- Math Expression with a referenced input disconnected → same illegal/incomplete wiring feedback as other processing elements
- Math Expression with unused configured pins (Inputs higher than symbols used; those pins empty) → allowed; only expression-referenced pins must be connected
- Math Expression multi-input channel mismatch (not equal and neither width is 1) or rate/band mismatch → refuse cable; clear tooltip (Utility add/multiply rules)
- Math Expression with broadcast-compatible widths (e.g. multi-channel × width-1) → legal; output channels = max of referenced inputs
- Parameter expression with `i` on a field that is not a per-copy numeric list (e.g. boolean/enum) → refuse commit with clear message
- Very large P: expression must still resolve for every slot; failure to resolve any slot → refuse commit (or keep prior if mid-re-eval fails validation)
- Integer-typed parameter (channels, kernel size, copy counts, etc.) where any slot’s expression result is not a whole number → refuse commit; keep prior value; clear “integer required” message (no round/floor)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The product MUST provide a Mathematical Expression processing element that applies a user-authored arithmetic expression to its signal input(s), without adding a hardcoded mod-sigmoid (or equivalent one-off activation) element.
- **FR-002**: The Mathematical Expression element MUST expose (a) an **Inputs** count parameter (Utility-style) that creates that many input pins labelled `x1`, `x2`, … `xN`, and (b) an editable expression string in which those pin names are the only legal signal symbols.
- **FR-003**: Mathematical Expression expressions MUST support numeric literals (including scientific notation such as `1e-7`) and the operators parentheses `()`, addition `+`, subtraction/negation `-`, multiplication `*`, and exponentiation `^`.
- **FR-004**: Mathematical Expression evaluation MUST be element-wise across corresponding samples of the referenced inputs. Connected inputs involved in the expression MUST follow the same channel compatibility rules as Utility add/multiply: channels are legal when they are equal or either side is 1 (scalar broadcast); output channel count MUST be the maximum among those inputs; temporal rate and band count MUST still match. Incompatible mixes MUST refuse the cable with a clear tooltip. With a single input, output shape MUST pass through that input’s shape.
- **FR-005**: Users MUST be able to wire the existing Sigmoid activation into Mathematical Expression input `x1` (typically Inputs = 1) so that an expression equivalent to `2 * x1^2.3 + 1e-7` realizes RAVE-style mod_sigmoid behavior.
- **FR-006**: Invalid Mathematical Expression strings (syntax error, unknown symbol, empty, reference to `xK` when Inputs &lt; K, non-finite evaluation) MUST be refused on commit: the prior valid expression remains, the invalid string MUST NOT be stored in the document, and a clear user-visible message MUST explain the error; processing MUST NOT silently use an undefined expression.
- **FR-007**: Numeric element parameters that participate in copy-expanded value lists MUST accept simple expressions over the index variable `i` in addition to plain numbers.
- **FR-008**: For parameter expressions, `i` MUST be the 0-based index in the expanded repeats/copies list for that parameter. If the element is not in a group, or effective expanded length P is 1, `i` MUST be `0`.
- **FR-009**: Parameter expressions MUST support the same operator set as FR-003: `()`, `+`, `-`, `*`, `^`, plus numeric literals and the symbol `i` (signal symbols `x1`…`xN` are not legal in parameter fields).
- **FR-010**: A single authored parameter expression containing `i` MUST be evaluated independently for each expanded slot with that slot’s `i`.
- **FR-011**: Plain numeric literals and constant expressions (no `i`) MUST continue to work, including existing short authored list lengths and under-the-hood tiling/expansion behavior.
- **FR-012**: When expanded length P changes because group copy counts change, an authored `i`-expression MUST be re-evaluated for the new index range without requiring the user to rewrite the expression.
- **FR-013**: Invalid parameter expressions or non-finite results for any required slot MUST be refused on commit: prior valid values remain, the invalid string MUST NOT be stored, and a clear message MUST be shown.
- **FR-014**: Expression grammar for Math Expression (`x1`…`xN`) and for parameters (`i`) MUST be documented consistently in-product (tooltip, placeholder, or equivalent) so users can discover allowed symbols and operators.
- **FR-015**: For parameters that require whole numbers, if any required slot’s expression result is not an integer, the product MUST refuse the commit, keep the prior valid value, and show a clear message that an integer is required—without rounding or truncating.
- **FR-016**: Changing Inputs MUST rebuild pins Utility-style; the product MUST keep the document consistent so expressions never retain references to removed pins (refuse the Inputs reduction or require editing the expression first).
- **FR-017**: Only input pins whose symbols appear in the current expression MUST be required to be connected for a valid graph; configured pins that are not referenced MAY remain empty.

### Key Entities

- **Mathematical Expression Element**: A processing graph element with an Inputs count, signal inputs `x1`…`xN`, a signal output, and an authored expression string; evaluates arithmetic over those inputs and constants.
- **Parameter Index Expression**: An authored string on a numeric parameter that may reference `i` (copy/repeats list index) and resolves to one concrete number per expanded slot.
- **Expanded Slot Index (`i`)**: Integer from `0` to `P−1` identifying a position in the copy-expanded parameter list; always `0` when P = 1 or the element is ungrouped.
- **Expression Grammar**: Shared operator vocabulary `() + - * ^` with literals; binding of `x1`…`xN` (signals) vs `i` (parameter index) depending on context.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A designer can reproduce RAVE-style mod_sigmoid (`2 * sigmoid(·)^2.3 + 1e-7`) using only Sigmoid + Mathematical Expression (`2 * x1^2.3 + 1e-7`) in under 2 minutes without any dedicated mod-sigmoid element.
- **SC-002**: For a group with at least 8 expanded parameter slots, entering one `i`-expression (e.g. `2*i+1`) correctly fills all slots on first commit in ≥95% of guided attempts (no manual per-slot typing).
- **SC-003**: Invalid expressions (Math Expression or parameter) are refused within 1 second of commit with a clear message; the document retains the prior valid formula and never processes an undefined expression.
- **SC-004**: When copy count changes after an `i`-expression is authored, all new slots show correctly re-evaluated values without the user re-entering the expression, in 100% of guided nest-depth tests (single group and at least one nested case).
- **SC-005**: At least three distinct non-mod-sigmoid compositions (including at least one that uses Inputs ≥ 2) can be demonstrated with the same Mathematical Expression element, showing the framework stays general.

## Assumptions

- OpenYourBox remains a general node-graph framework: RAVE-specific nonlinearities are composed from primitives, not hardcoded as one-off palette entries.
- Sigmoid activation already exists and is the intended partner for mod_sigmoid composition; this feature does not change Sigmoid’s core definition.
- Mathematical Expression uses a Utility-style **Inputs** count (minimum 1); pins are labelled `x1`…`xN` and those are the only legal signal symbols in the expression. Default Inputs = 1 covers mod_sigmoid after Sigmoid.
- Multi-input shape / wiring legality for Math Expression matches Utility add/multiply: channel widths must be equal or either side may be 1 (scalar broadcast); output channels = max; temporal rate and band count must match; otherwise refuse the cable.
- Unused configured Math Expression inputs (not named in the expression) may remain disconnected; only referenced pins are required.
- Operator set is exactly what was requested: `()`, `+`, `-`, `*`, `^`. No division, functions (sin, exp, log), or comparisons in this feature.
- `^` denotes exponentiation and is right-associative; `*` and `/`-style division are not both present—only `*`.
- Numeric literals include decimals and scientific notation (`1e-7`) so the mod_sigmoid constant can be typed literally.
- `i` is a single flattened index into the expanded repeats list (same order as existing copy-expanded parameter previews), not a multi-dimensional `(o,m,n)` tuple.
- Parameter expressions apply to numeric (and numeric-list) fields that already participate in copy expansion; non-numeric properties are unchanged.
- Integer-typed parameters never silently coerce non-integer expression results; refusal matches existing out-of-range parameter edit behavior.
- Invalid expression commits are always refused (not stored-as-flagged); prior valid authored form remains authoritative.
- Short-list dividing-length and tiling rules from the prior parameter-flexibility work remain in force; this feature extends what each authored token may contain (literal or expression).
- When a short list tiles, `i` always means the expanded slot index after tiling, so tiled copies of an `i`-expression still see distinct indices.
- Freeze/live dual-engine behavior for the new element follows the same user-visible rules as other processing elements (no special freeze exemption).
- In-product discovery (placeholder/tooltip) is sufficient documentation for v1; a separate user manual chapter is not required for readiness.
