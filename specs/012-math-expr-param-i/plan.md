# Implementation Plan: Math Expression Node & Parameter Index Expressions

**Branch**: `012-math-expr-param-i` | **Date**: 2026-08-28 | **Spec**: `specs/012-math-expr-param-i/spec.md`

**Input**: Feature specification from `specs/012-math-expr-param-i/spec.md`

## Summary

Add a **Mathematical Expression** graph element (Utility-style **Inputs** count, pins `x1`…`xN`, authored expression with `() + - * ^` and literals including scientific notation) so RAVE-style mod_sigmoid is composed as Sigmoid → Math Expression (`2 * x1^2.3 + 1e-7`) without a hardcoded activation. Multi-input shapes follow Utility add/multiply broadcast rules; only expression-referenced pins must be connected. Separately, extend **copy-expanded numeric parameters** to accept the same arithmetic grammar over index `i` (0…P−1; `i=0` when ungrouped/P=1), refusing invalid or non-integer results on integer fields and never storing refused strings. Shared expression parser/evaluator on the GUI/document path; live and freeze/train execute the same semantics with zero audio-thread parsing.

## Technical Context

**Language/Version**: C++17 (VST / graph / UI / live DSP); Python 3.x (freeze/train workers for Math Expression codegen)

**Primary Dependencies**: JUCE, Dear ImGui, imgui-node-editor, `NodeGraph` / `NodeRenderer` / `LiveGraphEngine`, LibTorch (`pow`, elementwise ops), existing Utility broadcast helpers

**Storage**: Expression strings on Math Expression nodes and on authored parameter tokens in `GraphDocument` / `NodeProperty` (extend property model beyond int/`float_value` JSON); ValueTree + `toJson`/`fromJson` round-trip

**Testing**: C++ unit tests under `Tests/` (grammar parse/eval, Math Expression pins/broadcast, param `i` over P, refuse paths); Python smoke for freeze/train Math Expression module; manual scenarios in `quickstart.md`

**Target Platform**: macOS VST3/AU (primary); Windows/Linux secondary if build allows

**Project Type**: Desktop audio plugin with embedded ImGui node editor

**Performance Goals**: Expression commit/validate interactive (&lt; 100 ms typical); UI 60 FPS; live Math Expression elementwise on audio thread with precompiled/prepared ops only (no parse/alloc on audio thread); constitution latency budgets unchanged

**Constraints**: VST-only UI; zero audio-thread allocations; Shape Integrity (refuse illegal cables); operator set exactly `() + - * ^` (no `/`, no functions); refuse invalid commits (no store-as-flagged); integer params refuse non-integers; no dedicated mod-sigmoid palette entry

**Scale/Scope**: One new element type; shared expression grammar for signal (`x1`…`xN`) and params (`i`); nested groups with P ≥ 8; Inputs ≥ 2 with broadcast; freeze/train registration for the new type

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Single Interface, Decoupled Compute**: Pass. Expression editing and parameter `i` live in the VST; workers only receive already-validated architecture snapshots.
- **Dual-Engine Execution Model**: Pass. Math Expression is Blue live + Gold freezeable like Activation; same formula semantics in both engines.
- **Manual Granular Freeze Policy**: Pass. No auto-freeze; Math Expression included in freeze payload like other processing nodes (must not fall through silent RAVE `else`).
- **Shape Integrity & Legal Constraints**: Pass. Multi-input connect uses Utility broadcast rules; mismatched rate/bands/channels refuse cables with tooltips.
- **Zero Audio-Thread Allocation Rule**: Pass. Parse/AST build on GUI thread at commit/compile; audio thread runs prepared elementwise LibTorch ops only.
- **Complexity Justification**: Pass. Shared grammar avoids one-off mod_sigmoid; Utility-style Inputs matches existing UX; `i` expressions are required for nested RAVE stacks.

**Post–Phase 1 re-check**: Still Pass. Contracts keep parsing on GUI/document path; live/freeze consume committed AST or equivalent prepared form; unused pins optional per FR-017 without weakening Shape Integrity on referenced inputs.

## Project Structure

### Documentation (this feature)

```text
specs/012-math-expr-param-i/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── expression-grammar-contract.md
│   ├── math-expression-element-contract.md
│   └── param-index-expression-contract.md
└── tasks.md                 # /speckit-tasks (not created here)
```

### Source Code (repository root)

```text
OpenYourBox/
├── Source/
│   ├── graph/
│   │   ├── GraphTypes.h / Expression*.{h,cpp}   # Shared lexer/parser/eval; PropertyKind / authored expr storage
│   │   ├── NodeGraph.cpp / .h                  # NodeType::mathExpression; Inputs pin rebuild x1…xN; connect/broadcast; property commit
│   │   └── NodeRenderer.cpp / .h               # Palette; expression field; param field parse refuse/restore
│   └── dsp/
│       └── LiveGraphEngine.cpp / .h            # Compile/execute Math Expression; prepared AST; broadcast
Backend/
├── freeze_worker.py                            # Explicit math_expression module (no silent skip)
└── train_worker.py                             # Same for RAVE/graph builders
Tests/
├── ExpressionGrammarTests.cpp                  # ()+-*^, literals, i/x1, refuse cases
├── MathExpressionNodeTests.cpp                 # pins, broadcast, mod_sigmoid cascade
└── (extend GraphGroupTests / RepeatListTilingTests for i-expressions)
```

**Structure Decision**: Extend existing `OpenYourBox` + `Backend` trees. No new app or process. Reuse Utility Inputs rebuild + `channelsAreBroadcastCompatible`; extend `parsePropertyRepeatList` / NodeRenderer refuse path from 011.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Shared expression AST + dual binding (`xK` vs `i`) | Spec requires one grammar for Math Expression and parameters | Two parsers diverge and break FR-014 discovery |
| String-valued / authored-expression property storage | Expressions must persist and re-eval when P changes | Storing only expanded numbers loses `i` authoring |
| Explicit freeze/train Math Expression branch | Unknown types are silently skipped in RAVE builder | Silent skip would break Gold parity for mod_sigmoid stacks |
