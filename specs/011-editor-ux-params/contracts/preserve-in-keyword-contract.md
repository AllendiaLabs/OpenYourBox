# Contract: Preserve Keyword `in`

## Purpose

Bind shape-driving dim/channels/features fields to the corresponding input shape using the reserved token `in` (FR-006, FR-007).

## Token

- Exact string: `in` (lowercase, case-sensitive).
- Single-field or list-of-`in` (tiling rules from copy-list-tiling-contract).
- Mixing `in` with numeric tokens in one field: **refused**.

## Applicable fields (v1)

Bindable integer properties that set output pin channels/features (at least):

- Linear: `features`
- Conv / ConvT / TCN (and similar): `channels`
- Other documented shape drivers that today take a user integer dim

Non-bindable fields (e.g. kernel size, stride, gain, activation choice): refuse `in` with clear message.

## Resolution

1. Determine paired input pin(s) for the element (same rules as live shape inference peers).
2. For each copy slot, read input `ShapeSignature.channels` (or relevant dim).
3. Write resolved integer into the effective property used for pin refresh and DSP.
4. Re-resolve on upstream shape or copy-count changes while binding remains active.

## Persistence

- Store binding intent (`in` / list of `in`), not only the last resolved integer.
- Round-trip: reload → still bound → re-resolve from current inputs.

## Failure modes

| Case | Behavior |
|------|----------|
| Unsupported field | Refuse commit |
| Mixed list | Refuse commit |
| Missing/illegal input shape | Unresolved / illegal cable messaging consistent with Shape Integrity |
| User replaces with number | Clear binding; numeric mode |

## Implementation anchors

- `parsePropertyCopyList` / property setters in `GraphTypes.h`, `NodeGraph.cpp`
- `refreshPropagatedPinShapes` / `setProperty` channel updates
- `NodeRenderer` property editors

## Non-goals

- Binding temporal rate / n_band in v1 unless already user-editable the same way
- Cross-element references (only corresponding input on same element)
