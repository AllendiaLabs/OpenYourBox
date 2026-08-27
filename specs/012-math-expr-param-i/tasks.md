---
description: "Task list for Math Expression Node & Parameter Index Expressions"
---

# Tasks: Math Expression Node & Parameter Index Expressions

**Input**: Design documents from `specs/012-math-expr-param-i/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Spec does not require TDD. Optional targeted unit tests included where the plan lists `Tests/` coverage; no test-first gate.

**Organization**: Tasks grouped by user story (US1–US3) for independent validation checkpoints.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- Graph: `OpenYourBox/Source/graph/`
- DSP: `OpenYourBox/Source/dsp/`
- Backend: `Backend/`
- Tests: `Tests/`
- Specs: `specs/012-math-expr-param-i/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design artifacts and inventory touch-points before code changes.

- [ ] T001 Verify design artifact cross-links (spec clarifications → contracts → plan) in `specs/012-math-expr-param-i/plan.md`
- [ ] T002 [P] Inventory Utility Inputs pin rebuild, broadcast connect, and `setMixerInputCount` call sites in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/graph/NodeGraph.h`
- [ ] T003 [P] Inventory `parsePropertyRepeatList` / NodeRenderer deactivate-after-edit refuse path and property JSON kinds in `OpenYourBox/Source/graph/GraphTypes.h`, `OpenYourBox/Source/graph/NodeGraph.cpp`, and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [ ] T004 [P] Inventory freeze/train unknown-type handling for new elements in `Backend/freeze_worker.py` and `Backend/train_worker.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared expression grammar and string property persistence required by US1 and US2.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T005 Add expression AST types + Doxygen API surface per `specs/012-math-expr-param-i/contracts/expression-grammar-contract.md` in `OpenYourBox/Source/graph/ExpressionParser.h`
- [ ] T006 Implement lexer/parser/evaluator (`() + - * ^`, unary `-`, scientific literals, right-assoc `^`) in `OpenYourBox/Source/graph/ExpressionParser.cpp`
- [ ] T007 Wire context allow-lists (`x1`…`xN` vs `i`) and refuse unknown idents / `/` / empty in `OpenYourBox/Source/graph/ExpressionParser.cpp`
- [ ] T008 Extend `NodeProperty` / `PropertyKind` (or equivalent) for authored string values with ValueTree + `toJson`/`fromJson` round-trip in `OpenYourBox/Source/graph/GraphTypes.h` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T009 Register new graph sources (`ExpressionParser.cpp`) in `CMakeLists.txt` / OpenYourBox target sources
- [ ] T010 [P] Add grammar unit coverage (literals, precedence, refuse cases) in `Tests/ExpressionGrammarTests.cpp` and register target in `CMakeLists.txt`

**Checkpoint**: Foundation ready — expressions parse/eval on GUI path; string properties persist; no audio-thread parsing.

---

## Phase 3: User Story 1 — Compose RAVE-style mod_sigmoid via Math Expression (Priority: P1) 🎯 MVP

**Goal**: Palette Math Expression element with Utility-style Inputs (`x1`…`xN`), expression field, Utility broadcast shapes, live + freeze/train execution; Sigmoid → `2 * x1^2.3 + 1e-7` without a hardcoded mod-sigmoid node.

**Independent Test**: `specs/012-math-expr-param-i/quickstart.md` §§1–2 — Sigmoid→Math Expression mod_sigmoid; Inputs≥2 broadcast; unused pins OK; invalid expression refused.

### Implementation for User Story 1

- [ ] T011 [US1] Add `NodeType::mathExpression` / persisted `"math_expression"` / `nodeTypeName`/`nodeTypeFromName` in `OpenYourBox/Source/graph/GraphTypes.h` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T012 [US1] Implement `makeNode` factory: `inputs` (default 1) + `expression` string; rebuild pins labelled `x1`…`xN` in `OpenYourBox/Source/graph/NodeGraph.cpp` per `specs/012-math-expr-param-i/contracts/math-expression-element-contract.md`
- [ ] T013 [US1] On `inputs` property commit, rebuild pins Utility-style; refuse reduction when expression still references removed `xK` (FR-016) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T014 [US1] Validate `expression` commit via ExpressionParser (idents ⊆ configured pins); refuse invalid; do not store bad string in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T015 [US1] Connect/shape: Utility add/multiply broadcast for referenced inputs; output channels = max; rate/bands must match; only referenced pins required (FR-004/017) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T016 [US1] Add palette entry “Math Expression” (no mod-sigmoid entry) and expression/Inputs UI with placeholder/tooltip (FR-014) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [ ] T017 [US1] Refuse-on-edit UX: restore prior expression buffer after invalid commit in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [ ] T018 [US1] Live compile: parse once on GUI/compile thread; prepare elementwise LibTorch ops in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [ ] T019 [US1] Live execute Math Expression with channel broadcast helpers (no parse/alloc on audio thread) in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [ ] T020 [P] [US1] Explicit `math_expression` builder in `Backend/freeze_worker.py` (must not silent-skip)
- [ ] T021 [P] [US1] Explicit `math_expression` builder in `Backend/train_worker.py` (RAVE/graph path; must not silent-skip)
- [ ] T022 [P] [US1] Add Math Expression node/broadcast/mod_sigmoid cascade coverage in `Tests/MathExpressionNodeTests.cpp` and register in `CMakeLists.txt`

**Checkpoint**: US1 complete — mod_sigmoid composable; multi-input broadcast works; Gold path builds the expression.

---

## Phase 4: User Story 2 — Per-copy parameter expressions of `i` (Priority: P1)

**Goal**: Copy-expanded numeric parameters accept `i` expressions; evaluate per expanded slot; refuse invalid/non-integer; re-eval when P changes; keep authored form.

**Independent Test**: `quickstart.md` §3 — group copies N=8 with `2*i+1`; ungrouped `i=0`; integer refuse; re-eval on copy-count change.

### Implementation for User Story 2

- [ ] T023 [US2] Extend `parsePropertyRepeatList` (or sibling API) to accept expression tokens over `i` per `specs/012-math-expr-param-i/contracts/param-index-expression-contract.md` in `OpenYourBox/Source/graph/GraphTypes.h` / `OpenYourBox/Source/graph/ExpressionParser.cpp`
- [ ] T024 [US2] Persist authored expression tokens (not only expanded numbers) on `NodeProperty` serialize/deserialize in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T025 [US2] On commit, evaluate each expanded slot with `i = k` (ungrouped/P=1 → `i=0`); refuse non-finite in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T026 [US2] Integer-typed fields: refuse non-integer results for any required slot (no round/floor); clear message in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T027 [US2] On `setGroupCopies` / nest change, re-evaluate authored `i`-expressions for new P (FR-012) alongside dividing-set rules in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T028 [US2] Property UI: accept `i`-expressions; refuse restores authored buffer; expanded preview shows resolved numbers; tooltip for grammar (FR-014) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [ ] T029 [P] [US2] Extend `i`-expression / refuse / re-eval coverage in `Tests/RepeatListTilingTests.cpp` and/or `Tests/GraphGroupTests.cpp`

**Checkpoint**: US2 complete — one `i`-expression fills P slots; invalid/non-integer refused without storing bad text.

---

## Phase 5: User Story 3 — Constants & mixed lists with shared grammar (Priority: P2)

**Goal**: Plain numbers and constant expressions (no `i`) keep working; short lists tile as in 011; tiled `i`-expressions use expanded slot index.

**Independent Test**: `quickstart.md` continuity with 011 — plain `3`, `2^3`, short literal lists; short list containing `i`-expr tiles with expanded `i`.

### Implementation for User Story 3

- [ ] T030 [US3] Ensure constant expressions without `i` resolve identically across applicable slots and preserve dividing-set length rules in `OpenYourBox/Source/graph/GraphTypes.h` / `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T031 [US3] When a short authored list with `i`-tokens tiles, bind `i` to expanded slot index (not short-list index only) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [ ] T032 [US3] Verify NodeRenderer authored field + read-only P preview remain correct for mixed literal/expression lists in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [ ] T033 [P] [US3] Add mixed-list / constant-expression cases in `Tests/RepeatListTilingTests.cpp`

**Checkpoint**: US3 complete — grammar reuse does not regress 011 tiling; mixed lists behave per US3.4.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and doc hygiene across stories.

- [ ] T034 [P] Mark TODO items for Math Expression / param `i` done or link to this feature in `TODO.md`
- [ ] T035 Run manual scenarios in `specs/012-math-expr-param-i/quickstart.md` (mod_sigmoid, broadcast, `i`, freeze smoke)
- [ ] T036 [P] Confirm Doxygen on new public APIs in `OpenYourBox/Source/graph/ExpressionParser.h` and Math Expression helpers in `OpenYourBox/Source/graph/NodeGraph.h`
- [ ] T037 Run registered C++ test targets (`ExpressionGrammarTests`, `MathExpressionNodeTests`, RepeatList/Group) via CTest entries declared in `CMakeLists.txt`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)** / **US2 (Phase 4)**: Both depend on Foundational; can proceed in parallel after T010 if staffed (share `ExpressionParser` / property strings — coordinate `NodeGraph.cpp` / `NodeRenderer.cpp` edits)
- **US3 (Phase 5)**: Depends on US2 commit/eval path (T023–T028); builds on tiling + `i`
- **Polish (Phase 6)**: Depends on desired stories complete (MVP can polish after US1 only)

### User Story Dependencies

- **US1 (P1)**: After Foundational — no dependency on US2/US3
- **US2 (P1)**: After Foundational — independent of Math Expression element (uses shared parser only)
- **US3 (P2)**: After US2 — mixed-list/`i` tiling semantics

### Parallel Opportunities

- T002–T004 (inventory) in parallel
- T010 grammar tests once T006–T007 exist
- After Foundational: US1 (T011–T022) and US2 (T023–T029) in parallel by different owners if `NodeGraph` merge discipline is agreed
- T020/T021 backend builders in parallel with live engine work
- T034/T036 polish docs in parallel

---

## Parallel Example: After Foundational

```bash
# Developer A — US1 Math Expression element:
Task: "Add NodeType::mathExpression in OpenYourBox/Source/graph/GraphTypes.h"
Task: "Live execute Math Expression in OpenYourBox/Source/dsp/LiveGraphEngine.cpp"
Task: "Explicit math_expression builder in Backend/freeze_worker.py"

# Developer B — US2 parameter i:
Task: "Extend parsePropertyRepeatList for i in OpenYourBox/Source/graph/GraphTypes.h"
Task: "Property UI i-expressions in OpenYourBox/Source/graph/NodeRenderer.cpp"
Task: "Extend i-expression coverage in Tests/RepeatListTilingTests.cpp"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 Setup
2. Complete Phase 2 Foundational (expression grammar + string properties)
3. Complete Phase 3 US1 (Math Expression + Sigmoid composition + freeze)
4. **STOP and VALIDATE** via `quickstart.md` §§1–2
5. Demo mod_sigmoid without hardcoded node

### Incremental Delivery

1. Setup + Foundational → shared grammar ready
2. US1 → mod_sigmoid MVP
3. US2 → `i` parameter expressions
4. US3 → mixed lists / constants polish
5. Phase 6 quickstart + CTest

### Parallel Team Strategy

1. Team finishes Setup + Foundational together
2. Then: A → US1, B → US2 (coordinate graph property files)
3. Either finishes US3 after US2
4. Shared polish / quickstart

---

## Notes

- [P] = different files / no incomplete-task dependency
- Refuse-invalid commits: never store refused strings (clarify session)
- Utility pin labels are `in N`; Math Expression must use `x1`…`xN`
- Do not add a palette “mod sigmoid” entry
- Prefer `/speckit-implement` after this file is accepted
