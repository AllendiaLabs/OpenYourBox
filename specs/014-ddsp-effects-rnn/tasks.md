---
description: "Task list for DDSP Effects & Recurrent Layers"
---

# Tasks: DDSP Effects & Recurrent Layers

**Input**: Design documents from `specs/014-ddsp-effects-rnn/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Spec does not require TDD. Targeted live/worker tests are included where `plan.md` lists `Tests/` coverage; no test-first gate. Manual UI flows validated via `quickstart.md`.

**Organization**: Tasks grouped by user story (US1–US5) for independent validation checkpoints.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label on story-phase tasks only
- Every task includes an exact file path

## Path Conventions

- Plug-in: `OpenYourBox/Source/`
- Graph: `OpenYourBox/Source/graph/`
- DSP: `OpenYourBox/Source/dsp/`
- UI: `OpenYourBox/Source/ui/`
- Backend: `Backend/`
- Tests: `Tests/`
- Specs: `specs/014-ddsp-effects-rnn/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design artifacts and inventory registration / live / worker touch-points before code changes.

- [X] T001 Verify design artifact cross-links (spec clarifications → contracts → plan → data-model) in `specs/014-ddsp-effects-rnn/plan.md`
- [X] T002 [P] Inventory `NodeType`, chrome helpers, and `isTrainableType` / `isShapePassthroughType` in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T003 [P] Inventory `nodeTypeName` / `nodeTypeFromName` / `isKnownPersistedNodeType` / `makeNode` in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T004 [P] Inventory Factory `paletteItems` and property-row patterns in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T005 [P] Inventory `LiveGraphEngine` compile/process/`histories` reset and `NoiseSynthesizer` FFT helpers in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` and `OpenYourBox/Source/dsp/NoiseSynthesizer.h`
- [X] T006 [P] Inventory freeze/train `build_module` type branches in `Backend/freeze_worker.py` and `Backend/train_worker.py`
- [X] T007 [P] Add Magenta/DDSP Apache 2.0 behavioral-reimplementation attribution block in `NOTICE`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared type registry, palette category scaffolding, DSP file shells, and live-safe warning plumbing required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T008 Add `NodeType` values for `reverb`, `expDecayReverb`, `filteredNoiseReverb`, `firFilter`, `modDelay`, `lstm`, `rnn` plus chrome/`isTrainableType`/`isShapePassthroughType` (and related predicates) in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T009 Wire `nodeTypeName` / `nodeTypeFromName` / `isKnownPersistedNodeType` persist strings (`reverb`, `exp_decay_reverb`, `filtered_noise_reverb`, `fir_filter`, `mod_delay`, `lstm`, `rnn`) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T010 Implement categorized Factory palette sections (Effects, Neural / Sequence, existing groups) per `specs/014-ddsp-effects-rnn/contracts/element-palette-categories-contract.md` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T011 Add `makeNode` stubs with labels, default ports, and empty/minimal `properties` for all seven types in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T012 Create `OpenYourBox/Source/dsp/DdspEffects.h` and `OpenYourBox/Source/dsp/DdspEffects.cpp` shells (IR helpers, convolve entry points) and register sources in the OpenYourBox CMake/target lists
- [X] T013 Create `OpenYourBox/Source/dsp/RecurrentLayers.h` and `OpenYourBox/Source/dsp/RecurrentLayers.cpp` shells (RNN/LSTM cell API) and register sources in the OpenYourBox CMake/target lists
- [X] T014 Add live-safe IR-length threshold constant and non-blocking warning hook (InfoPanel and/or property affordance) per `specs/014-ddsp-effects-rnn/contracts/ddsp-effects-contract.md` in `OpenYourBox/Source/ui/InfoPanel.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T015 Extend `LiveGraphEngine` compile switch stubs / `CompiledElement` handling so unknown new types do not crash compile (refuse or no-op with clear path) in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` and `OpenYourBox/Source/dsp/LiveGraphEngine.h`

**Checkpoint**: Foundation ready — seven types persist and appear in categorized palette; DSP shells and warning plumbing exist; live engine tolerates new types safely.

---

## Phase 3: User Story 1 — Place and Hear Differentiable Effects (Priority: P1) 🎯 MVP

**Goal**: Effects category lists all five DDSP types; ExpDecayReverb is fully placeable, editable (`gain`, `decay`, `reverb_length`, `add_dry`), and audible live without stopping audio.

**Independent Test**: `specs/014-ddsp-effects-rnn/quickstart.md` Scenario A — Audio In → ExpDecayReverb → Audio Out; decay/`add_dry` respond live; Effects menu shows five types.

### Implementation for User Story 1

- [X] T016 [US1] Complete ExpDecayReverb `makeNode` defaults and property schema (`gain`, `decay`, `reverb_length`, `add_dry`) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T017 [US1] Implement Magenta-style exponential-decay IR builder and FFT convolve wet path in `OpenYourBox/Source/dsp/DdspEffects.cpp` and `OpenYourBox/Source/dsp/DdspEffects.h`
- [X] T018 [US1] Compile and process ExpDecayReverb in the live engine (preallocate IR/history on message thread; dry-add; soft warn on long `reverb_length`) in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T019 [US1] Expose ExpDecayReverb (and Effects list entries) property rows in Parameters/panel patterns in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T020 [US1] Ensure Effects category lists Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, ModDelay as insertable items in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T021 [US1] Add ExpDecayReverb branch to freeze `build_module` in `Backend/freeze_worker.py`
- [X] T022 [US1] Add ExpDecayReverb branch to train `build_module` in `Backend/train_worker.py`
- [X] T023 [P] [US1] Add live ExpDecayReverb decay/`add_dry` coverage in `Tests/LiveGraphEngineTests.cpp`
- [X] T024 [P] [US1] Add freeze/train ExpDecayReverb smoke coverage in `Tests/test_freeze_worker.py` and `Tests/test_train_worker.py`

**Checkpoint**: US1 complete — Effects menu populated; ExpDecayReverb audible and freeze/train-eligible; other effect types placeable as stubs until later stories.

---

## Phase 4: User Story 2 — Shape Spaces with Frequency-Domain Filters (Priority: P2)

**Goal**: FIRFilter and FilteredNoiseReverb expose magnitude-grid dimensions + window size (+ reverb length / add_dry for FilteredNoise), init/randomize trainable magnitudes, refuse illegal sizes/connections; no cell painter.

**Independent Test**: `quickstart.md` Scenario B — FIRFilter/FilteredNoiseReverb dimension + randomize; illegal size refused.

### Implementation for User Story 2

- [X] T025 [US2] Complete FIRFilter and FilteredNoiseReverb `makeNode` properties (`n_frames`, `n_filter_banks`, `window_size`, plus FilteredNoise `reverb_length`/`add_dry`) and trainable magnitude ownership in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T026 [US2] Implement FilteredNoise IR synth and LTV-FIR / frequency-filter helpers in `OpenYourBox/Source/dsp/DdspEffects.cpp` and `OpenYourBox/Source/dsp/DdspEffects.h`
- [X] T027 [US2] Compile/process FIRFilter and FilteredNoiseReverb with magnitude tensors and dimension validation in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T028 [US2] Property UI: dimension editors + init/randomize for magnitudes (no cell painter) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T029 [US2] Refuse illegal magnitude dimensions / shape-breaking connections with Shape Integrity messaging in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T030 [US2] Add FIRFilter and FilteredNoiseReverb freeze modules in `Backend/freeze_worker.py`
- [X] T031 [US2] Add FIRFilter and FilteredNoiseReverb train modules (trainable magnitudes) in `Backend/train_worker.py`
- [X] T032 [P] [US2] Add live FIRFilter / FilteredNoiseReverb coverage in `Tests/LiveGraphEngineTests.cpp`
- [X] T033 [P] [US2] Add freeze/train coverage for FIRFilter and FilteredNoiseReverb in `Tests/test_freeze_worker.py` and `Tests/test_train_worker.py`

**Checkpoint**: US2 complete — spectral effect nodes editable via dims + randomize/train; illegal shapes refused.

---

## Phase 5: User Story 3 — Modulated Delay Color (Priority: P2)

**Goal**: ModDelay provides chorus/flanger/vibrato-style modulation with `center_ms`, `depth_ms`, `gain`, `phase`, `add_dry`, and validates non-negative delay.

**Independent Test**: `quickstart.md` Scenario C — sweep `depth_ms`/`phase`; confirm modulation character.

### Implementation for User Story 3

- [X] T034 [US3] Complete ModDelay `makeNode` property schema and validation (`center_ms - depth_ms` clamp/refuse) in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T035 [US3] Implement variable-length modulated delay in `OpenYourBox/Source/dsp/DdspEffects.cpp` and `OpenYourBox/Source/dsp/DdspEffects.h`
- [X] T036 [US3] Compile/process ModDelay with preallocated delay line sized for `center_ms + depth_ms` in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T037 [US3] Expose ModDelay properties in Parameters UI in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T038 [US3] Add ModDelay freeze module in `Backend/freeze_worker.py`
- [X] T039 [US3] Add ModDelay train module in `Backend/train_worker.py`
- [X] T040 [P] [US3] Add live ModDelay modulation coverage in `Tests/LiveGraphEngineTests.cpp`
- [X] T041 [P] [US3] Add freeze/train ModDelay smoke coverage in `Tests/test_freeze_worker.py` and `Tests/test_train_worker.py`

**Checkpoint**: US3 complete — ModDelay audible and freeze/train-eligible.

---

## Phase 6: User Story 4 — Convolutional Reverb with Optional IR (Priority: P2)

**Goal**: Reverb convolves with internal/trainable IR; optional external IR input with `ir_blend`; empty IR falls back with recoverable warning; long length soft-warns.

**Independent Test**: `quickstart.md` Scenario D — internal IR; blend sweep; empty IR fallback; long-length warning.

### Implementation for User Story 4

- [X] T042 [US4] Complete Reverb `makeNode` with audio + optional `ir` ports, `reverb_length`, `add_dry`, `ir_blend`, trainable internal IR in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T043 [US4] Implement IR dry-mask, internal/external blend, and FFT convolve helpers in `OpenYourBox/Source/dsp/DdspEffects.cpp` and `OpenYourBox/Source/dsp/DdspEffects.h`
- [X] T044 [US4] Compile/process Reverb dual-input path, empty-IR fallback warning, and live-safe length warning in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T045 [US4] Surface Reverb properties, blend control, and recoverable empty-IR / performance warnings in `OpenYourBox/Source/graph/NodeRenderer.cpp` and `OpenYourBox/Source/ui/InfoPanel.cpp`
- [X] T046 [US4] Add Reverb freeze path including optional IR input / blend in `Backend/freeze_worker.py`
- [X] T047 [US4] Add Reverb train path (trainable IR + multi-input DAG as needed) in `Backend/train_worker.py`
- [X] T048 [P] [US4] Add live Reverb blend / empty-IR / length-warn coverage in `Tests/LiveGraphEngineTests.cpp`
- [X] T049 [P] [US4] Add freeze/train Reverb coverage in `Tests/test_freeze_worker.py` and `Tests/test_train_worker.py`

**Checkpoint**: US4 complete — convolutional Reverb matches clarified IR blend/fallback/warn contracts.

---

## Phase 7: User Story 5 — Recurrent Layers in the Graph (Priority: P3)

**Goal**: Single-layer LSTM and RNN under Neural / Sequence with hidden size, bidirectional, bias, Activation/TCN activation+gain in-cell, inferred input size, full-sequence out, hidden-state carry/reset, freeze/train.

**Independent Test**: `quickstart.md` Scenario E — place LSTM/RNN; shapes; stack two layers; freeze; state carry/reset.

### Implementation for User Story 5

- [X] T050 [US5] Complete LSTM/RNN `makeNode` properties (`hidden_size`, `bidirectional`, `bias`, `activation`, `negative_slope`, `gain`; no `num_layers`/`input_size`) and output shape rules in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T051 [US5] Implement custom single-layer RNN and LSTM cells with in-cell activation + gain in `OpenYourBox/Source/dsp/RecurrentLayers.cpp` and `OpenYourBox/Source/dsp/RecurrentLayers.h`
- [X] T052 [US5] Allocate/carry/reset hidden (and LSTM cell) state in `LiveGraphRuntime` on prepare/rebuild/reconnect/freeze-swap/randomize in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` and `OpenYourBox/Source/dsp/LiveGraphEngine.h`
- [X] T053 [US5] Compile/process LSTM/RNN full-sequence I/O (uni: `hidden_size` ch; bi: `2 * hidden_size`) with inferred input size in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T054 [US5] List LSTM/RNN under Neural / Sequence and expose property rows (activation/gain/slope/bias/bidirectional) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T055 [US5] Refuse illegal recurrent connections with Shape Integrity tooltips in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T056 [US5] Add custom LSTM/RNN freeze modules (bake gain for Gold parity) in `Backend/freeze_worker.py`
- [X] T057 [US5] Add custom LSTM/RNN train modules in `Backend/train_worker.py`
- [X] T058 [P] [US5] Add live LSTM/RNN shape, bidirectional channels, and hidden-state carry/reset coverage in `Tests/LiveGraphEngineTests.cpp`
- [X] T059 [P] [US5] Add freeze/train LSTM/RNN shape and parameter coverage in `Tests/test_freeze_worker.py` and `Tests/test_train_worker.py`

**Checkpoint**: US5 complete — recurrent layers usable live and in freeze/train; state lifecycle matches FR-011a.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Persist, randomize, mixed freeze, NOTICE, and quickstart validation across all seven types.

- [X] T060 Verify save/reload restores all seven types, connections, and properties (SC-006) via graph persist paths in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T061 Confirm weight randomization / seed behavior for IR, magnitudes, and recurrent weights in `OpenYourBox/Source/dsp/WeightRandomizer.cpp` and `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T062 [P] Add mixed effects+recurrent legal freeze coverage in `Tests/LiveGraphEngineTests.cpp` and/or `Tests/test_freeze_worker.py`
- [X] T063 [P] Confirm Magenta/DDSP NOTICE block completeness in `NOTICE`
- [X] T064 Run `specs/014-ddsp-effects-rnn/quickstart.md` Scenarios A–G and fix residual gaps in touched OpenYourBox/Backend files
- [X] T065 Mark completed DDSP effects + LSTM/RNN TODO items in `TODO.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: Depends on Foundational — MVP
- **US2 (Phase 4)**: Depends on Foundational; benefits from US1 ExpDecay/FFT helpers but independently testable
- **US3 (Phase 5)**: Depends on Foundational; independent of US2/US4 DSP
- **US4 (Phase 6)**: Depends on Foundational; shares convolve helpers with US1
- **US5 (Phase 7)**: Depends on Foundational; independent of effects stories
- **Polish (Phase 8)**: Depends on all desired user stories

### User Story Dependencies

- **US1 (P1)**: After Foundational — no story dependencies — 🎯 MVP
- **US2 (P2)**: After Foundational — may reuse `DdspEffects` convolve from US1
- **US3 (P2)**: After Foundational — parallelizable with US2/US4
- **US4 (P2)**: After Foundational — may reuse FFT convolve from US1
- **US5 (P3)**: After Foundational — parallelizable with all effects stories

### Within Each User Story

- `makeNode` / properties → DSP helpers → live engine → UI → freeze → train → tests
- Story complete at checkpoint before treating it done

### Parallel Opportunities

- Phase 1 inventory tasks T002–T007 marked [P]
- After Foundational: US2, US3, US4, US5 can proceed in parallel (watch shared files: `DdspEffects.*`, `LiveGraphEngine.cpp`, workers)
- Within a story, paired freeze/train or dual test tasks marked [P]

---

## Parallel Example: User Story 1

```bash
# After T016–T022 implementation core:
Task: "Add live ExpDecayReverb coverage in Tests/LiveGraphEngineTests.cpp"
Task: "Add freeze/train ExpDecayReverb smoke in Tests/test_freeze_worker.py and Tests/test_train_worker.py"
```

## Parallel Example: After Foundational

```bash
# Different owners / careful merge on shared DSP files:
Task: "US2 FIRFilter + FilteredNoiseReverb (Phase 4)"
Task: "US3 ModDelay (Phase 5)"
Task: "US5 LSTM/RNN (Phase 7)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1 (ExpDecayReverb + Effects menu)
4. **STOP and VALIDATE** via `quickstart.md` Scenario A
5. Demo live differentiable reverb in-graph

### Incremental Delivery

1. Setup + Foundational → registry + palette categories ready
2. US1 → MVP ExpDecayReverb
3. US2 → spectral FIR / filtered-noise reverbs
4. US3 → ModDelay
5. US4 → full Reverb + IR blend
6. US5 → LSTM/RNN
7. Polish → persist, randomize, quickstart A–G, TODO

### Parallel Team Strategy

1. Team completes Setup + Foundational together
2. Then:
   - Dev A: US1 then US4 (shared reverb/FFT)
   - Dev B: US2 + US3 (filters / delay)
   - Dev C: US5 (recurrent; mostly `RecurrentLayers.*`)
3. Integrate at Polish

---

## Notes

- [P] = different files / no incomplete-task dependency
- [USn] maps to spec user stories for traceability
- Other effect types may place from the Effects menu after Foundational/US1 before their DSP is complete; do not claim US2–US4 done until checkpoints pass
- Prefer Magenta property key names for effects; Activation/TCN keys for recurrent `gain`/`activation` (type-scoped)
- Commit after each task or logical group
- Avoid: stock `nn.LSTM` without custom in-cell activation; hard-clamping `reverb_length`; cell painters
