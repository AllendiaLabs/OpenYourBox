---
description: "Task list for RAVE Prior Mix & Insert Catalog"
---

# Tasks: RAVE Prior Mix & Insert Catalog

**Input**: Design documents from `specs/016-rave-prior-mix/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Spec does not require TDD. Targeted `Tests/LiveGraphEngineTests.cpp` coverage is included where `plan.md` / `quickstart.md` call for engine checks; no test-first gate. Manual UI flows validated via `quickstart.md`.

**Organization**: Tasks grouped by user story (US1–US4) for independent validation checkpoints.

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
- Specs: `specs/016-rave-prior-mix/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm design artifacts and inventory BlackBox prior-path / palette / library touch-points before code changes.

- [X] T001 Verify design artifact cross-links (spec clarifications → contracts → plan → data-model) in `specs/016-rave-prior-mix/plan.md`
- [X] T002 [P] Inventory BlackBox encode→fidelity→decode, `latentInputIndex`, `latentOutputs`, and `gatherLatentInput` in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T003 [P] Inventory `applyExternalLoadSurface`, train-autoload latent pins, fidelity `NodeProperty` / Gold edit exception in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T004 [P] Inventory `latentPinLabel`, `isLatentPin`, `fidelityPercent`, `NodeProperty` patterns in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T005 [P] Inventory `paletteCategories` / `forEachPaletteItem` / Pin Add / Link Insert in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T006 [P] Inventory `UserBoxLibrary::insertBox` and expandable tree helpers in `OpenYourBox/Source/graph/UserBoxLibraryPanel.cpp`
- [X] T007 [P] Inventory TorchScript `encode`/`decode` and compactness path in `OpenYourBox/Source/dsp/TorchScriptBlackBox.cpp` and μ-only `VariationalBottleneck::encodeMean` in `OpenYourBox/Source/dsp/VariationalBottleneck.cpp`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared types, pin labels, `priorMix` property plumbing, distribution/sample helpers, and Factory catalog extraction required by later stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T008 Add `priorMix` field helpers (range `[0,1]`, default `0`, clamp) and bias/scale pin label + `isBiasPin` / `isScalePin` helpers in `OpenYourBox/Source/graph/GraphTypes.h`
- [X] T009 Add `priorMix` `NodeProperty` factory (fidelity-class real prop), ValueTree serialize/restore, and allow Gold `setProperty` for `priorMix` like `fidelity` in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T010 Extend `CompiledElement` / `RuntimeControlState` with `priorMix` and bias/scale gather indices (replace latent-input decode shortcut fields as needed) in `OpenYourBox/Source/dsp/LiveGraphEngine.h`
- [X] T011 Add RT-safe sample helpers: softplus-std convention, preallocated `ε` noise buffers at prepare/compile (no audio-thread heap) in `OpenYourBox/Source/dsp/VariationalBottleneck.cpp` and/or `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T012 Add distribution acquisition API for RAVE BlackBox: prefer `(μ, σ)`; fallback `μ←encode`, `σ←1` per `specs/016-rave-prior-mix/contracts/rave-prior-mix-runtime-contract.md` in `OpenYourBox/Source/dsp/TorchScriptBlackBox.cpp` / `.h` and call sites in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T013 Extract shared Factory palette catalog (`PaletteItem` / categories / `forEachPaletteItem` / place helpers) from `OpenYourBox/Source/graph/NodeRenderer.cpp` into `OpenYourBox/Source/graph/FactoryPalette.h` (and `.cpp` if needed); leave left Factory and context menus compiling against it

**Checkpoint**: Foundation ready — `priorMix` persists; compile state can carry mix + bias/scale indices; sample/distribution helpers exist; Factory catalog has a single source of truth.

---

## Phase 3: User Story 1 — Continuous Forward ↔ Prior Mix (Priority: P1) 🎯 MVP

**Goal**: RAVE-capable Gold / external-load boxes expose `priorMix`; runtime lerps encoder mean/spread toward 0/1, applies bias/scale defaults (0/1), samples once, decodes; at full prior skips encode.

**Independent Test**: `specs/016-rave-prior-mix/quickstart.md` §1 — sweep `priorMix` 0→1; at 1, audio input changes do not affect output.

### Implementation for User Story 1

- [X] T014 [US1] Show `priorMix` property row on RAVE-capable Gold / encode-decode external-load boxes (hide on forward-only) in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T015 [US1] Compile `priorMix` into runtime control state for blackBox elements in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T016 [US1] Implement blackBox prior-mix execute path per `specs/016-rave-prior-mix/contracts/rave-prior-mix-runtime-contract.md`: encode (or skip), fidelity-before-mix when compactness ready, lerp to (0,1), bias=0/scale=1 defaults, sample, decode in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T017 [US1] At `priorMix ≈ 1`, skip `kernel->encode` and use base (0,1) while still decoding in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T018 [P] [US1] Add engine tests for prior-mix lerp endpoints, intermediate continuity smoke, and encoder-skip independence from audio at full prior in `Tests/LiveGraphEngineTests.cpp`

**Checkpoint**: US1 complete — `priorMix` morphs forward↔prior; full prior skips encode with neutral bias/scale.

---

## Phase 4: User Story 2 — Bias and Scale Pins Replace Latent Input (Priority: P1)

**Goal**: Remove latent input; add bias/scale pins; wire gather + shape checks; apply bias/scale at every prior-mix setting.

**Independent Test**: `quickstart.md` §2 — no latent-in; disconnected defaults; connected bias/scale change sound; illegal shapes refused.

### Implementation for User Story 2

- [X] T019 [US2] Replace latent **input** with bias + scale inputs on external-load and train-autoload RAVE Gold pin surfaces per `specs/016-rave-prior-mix/contracts/bias-scale-pin-surface-contract.md` in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T020 [US2] Compile bias/scale pin indices; gather tensors (or constants 0/1 when disconnected) in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T021 [US2] Apply `μ += bias`, `σ ⊙= scale` after prior-mix lerp and before sample at all `priorMix` values in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T022 [US2] Shape Integrity: refuse illegal bias/scale connections vs effective latent width with existing mismatch feedback in `OpenYourBox/Source/graph/NodeGraph.cpp`
- [X] T023 [US2] Remove decode-from-wired-latent-in branch for RAVE-capable boxes; update any UI/docs assumptions in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T024 [P] [US2] Update/add tests for bias/scale defaults, steering, and latent-in absence in `Tests/LiveGraphEngineTests.cpp`

**Checkpoint**: US2 complete — bias/scale pins steer; latent-in gone; defaults work unwired.

---

## Phase 5: User Story 3 — Latent Output Exposes Sampled Values in Use (Priority: P2)

**Goal**: Latent out publishes the post-mix, post-bias/scale, post-sample `z` that drives decode (including full prior).

**Independent Test**: `quickstart.md` §3 — latent tap tracks `priorMix` / bias / scale; matches decode input.

### Implementation for User Story 3

- [X] T025 [US3] Ensure `runtime.latentOutputs[i]` is written with effective sampled `z` immediately before `decode` on the prior-mix path in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T026 [US3] Verify latent-out pin / `inputUseLatentTap` gather reads that buffer (not pre-mix μ) across forward, intermediate, and full prior in `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- [X] T027 [P] [US3] Add instrumented checks that latent out equals decode input (≥95% buffer agreement intent from SC-005) in `Tests/LiveGraphEngineTests.cpp`

**Checkpoint**: US3 complete — latent out is the effective sampled latent.

---

## Phase 6: User Story 4 — Right-Click Add/Insert Shows Factory and User Library (Priority: P2)

**Goal**: Pin Add and Link Insert menus list the full shared Factory catalog and an expandable User Library hierarchy for root/nested insert.

**Independent Test**: `quickstart.md` §5 — right-click menus show all Factory types + library expand/insert.

### Implementation for User Story 4

- [X] T028 [US4] Rebuild Pin **Add** and Link **Insert** menus from shared `FactoryPalette` (categorized, filter-aware) per `specs/016-rave-prior-mix/contracts/right-click-insert-catalog-contract.md` in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T029 [US4] Add expandable User Library submenu (folders → entries → snapshot members) reusing helpers from `OpenYourBox/Source/graph/UserBoxLibraryPanel.cpp` inside `OpenYourBox/Source/graph/NodeRenderer.cpp` context menus
- [X] T030 [US4] Wire library menu actions to `UserBoxLibrary::insertBox` / `placeLibraryEntryOnFocusedCanvas` (root + `nestedRootId`); best-effort pin/link wire or place-nearby per contract in `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T031 [US4] Ensure empty user library leaves Factory section complete and library section vacant (not replacing Factory) in `OpenYourBox/Source/graph/NodeRenderer.cpp`

**Checkpoint**: US4 complete — right-click catalogs match Factory + expandable user library.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: OYB `(μ, σ)` fidelity, regressions, and quickstart validation across stories.

- [X] T032 [P] If needed for learned Gold scale fidelity: expose `(μ, σ)` from OYB RAVE export / bottleneck path in `Backend/train_worker.py` and consume in `OpenYourBox/Source/dsp/TorchScriptBlackBox.cpp` (keep external single-tensor fallback)
- [X] T033 [P] Confirm forward-only TorchScript Load still omits `priorMix` / bias / scale / latent-out in `OpenYourBox/Source/graph/NodeGraph.cpp` and `OpenYourBox/Source/graph/NodeRenderer.cpp`
- [X] T034 Run `specs/016-rave-prior-mix/quickstart.md` scenarios §1–§5 and fix gaps in touched `OpenYourBox/Source/` / `Tests/` files
- [X] T035 [P] Mark completed checklist notes / follow-ups in `specs/016-rave-prior-mix/checklists/requirements.md` if any validation notes need updating after implementation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: Depends on Foundational — MVP
- **US2 (Phase 4)**: Depends on Foundational; naturally follows US1 runtime path (extends bias/scale beyond constants)
- **US3 (Phase 5)**: Depends on US1 sample path (and US2 if verifying bias/scale effects on latent out)
- **US4 (Phase 6)**: Depends on Foundational T013 (shared Factory catalog); **independent of US1–US3** and can run in parallel after Phase 2
- **Polish (Phase 7)**: After desired stories complete

### User Story Dependencies

- **US1 (P1)**: After Foundational — no story dependencies
- **US2 (P1)**: After Foundational; should integrate with US1 execute path but testable via pin surface + steering
- **US3 (P2)**: After US1 (latent out write); stronger with US2
- **US4 (P2)**: After Foundational catalog extract — parallelizable vs RAVE runtime stories

### Parallel Opportunities

- T002–T007 (Setup inventory) in parallel
- T008–T013 mostly sequential within Foundation (T013 catalog extract can parallel T008–T012 if staffing allows)
- After Phase 2: **US4** in parallel with **US1→US2→US3**
- T018 / T024 / T027 test tasks parallelizable with non-overlapping edits once impl lands
- T032 / T033 / T035 polish items marked [P]

### Parallel Example: After Foundational

```bash
# Track A — RAVE runtime MVP chain:
Task: "T014–T018 US1 prior mix"
Task: "T019–T024 US2 bias/scale pins"
Task: "T025–T027 US3 latent out"

# Track B — catalog (parallel):
Task: "T028–T031 US4 right-click Factory + User Library"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 Setup
2. Complete Phase 2 Foundational
3. Complete Phase 3 US1 (`priorMix` + encoder skip + sample/decode with bias/scale defaults)
4. **STOP and VALIDATE** via `quickstart.md` §1 (+ T018)
5. Demo morph forward↔prior

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. US1 → prior mix MVP
3. US2 → bias/scale pins
4. US3 → latent out identity
5. US4 → right-click catalog (or parallel after Foundation)
6. Polish → export `(μ,σ)` if needed + full quickstart

### Suggested MVP Scope

**US1 only** (plus Foundational): continuous prior mix with encoder skip. Bias/scale pins (US2) and catalog (US4) are the next high-value increments.

---

## Notes

- [P] = different files / no incomplete-task dependency
- Old latent-in project migration is **out of scope** (FR-011) — do not add migration tasks
- Blue modular prior-mix rebuild is out of scope
- Left Library panel behavior is out of scope except sharing Factory catalog data source via T013
- Commit after each task or logical group; stop at any checkpoint to validate independently
