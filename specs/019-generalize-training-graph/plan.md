# Implementation Plan: Generalize Training Graph

**Branch**: `019-generalize-training-graph` | **Date**: 2026-09-04 | **Spec**: `specs/019-generalize-training-graph/spec.md`

**Input**: Feature specification from `specs/019-generalize-training-graph/spec.md`

**Status**: Design complete. Ready for `/speckit-tasks`.

Next: `/speckit-tasks` (then `/speckit-implement`).

Refactor train/inference to be **architecture-agnostic**: remove mapping/reconstruction modes and user-facing RAVE/steerable/TCN recipe branding from the Train workflow; introduce a **Data Loader** node (per-output bindings, equal-count at Run, distinct cables), **Loss** nodes (stage-weighted + staged schedules), and a general **Training Configuration** surface (user library + project snapshot). Train only the **active data-loader path ∩ armed** set (passthrough otherwise; Gold always passthrough). Allow **group-of-one** with hub dedupe for shared live+loader pins; remove **TCN** and **Linear** from the palette. Ship example graph templates + training configs for prior mapping/reconstruction capability classes. Legacy project migration is out of scope. Local and cloud destinations consume the same generalized train package; VST remains the sole training UI.

## Technical Context

**Language/Version**: C++17 (JUCE plug-in / live engine), Python 3 (freeze + train worker, CloudService)

**Primary Dependencies**: JUCE 8, Dear ImGui, imgui-node-editor, LibTorch (live + TorchScript Gold), PyTorch in train/freeze workers; auraloss (or equivalent) for MR-STFT loss; existing reconstruction spectral/GAN helpers in `Backend/train_worker.py` generalized behind loss nodes

**Storage**: ValueTree graph (new node types, arm flags, data-loader bindings, loss wiring); Training Library (unchanged pool for materials); user training-config library under plugin user data; optional project training-config snapshot in patch/session; copyright log unchanged; `.pt` artifacts

**Testing**: CTest for graph rules (data-loader connect gates, group-of-one, path/arm opacity helpers, palette removals); Python tests for generalized IPC, loss stages, equal-count packaging, mapping/reconstruction recipe reproduction via configs; DAW scenarios in `quickstart.md`

**Target Platform**: Desktop AU/VST3, macOS first; same local ChildProcess and Allendia cloud path as 017/018

**Project Type**: Single desktop audio plug-in with embedded Python workers + optional CloudService

**Performance Goals**: 60 FPS UI under train load; zero audio-thread allocations; train never blocks `processBlock`; loss/progress UI ≥ ~1 Hz; live/Gold latency unchanged for non-rate-reducing graphs; Data Loader / Loss ignored on live audible path

**Constraints**: VST-only Train UI; one active Data Loader per Run (Train panel picker); equal-count only at Start for connected outputs; Data Loader on empty or Audio-In-fed pins (refuse processing-driven); Loss prediction=live / target=Data Loader; stage-only loss weights; per-stage `freeze_element_ids`; group hub dedupe for shared live+loader pins; no architecture mode selectors; no legacy project migration; copyright before first Train; mixed sample rates blocked

**Scale/Scope**: One train job per master; multiple Data Loaders on canvas (one active); tens of nodes / several loss nodes; multi-stage schedules (e.g. RAVE-like 1e6+1e6); user config library disk-limited; example templates for mapping + reconstruction classes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Single Interface, Decoupled Compute`: **PASS** — Data Loader, Loss, Train config library, and examples live inside the VST; training remains a detached Python worker (local or cloud); no standalone train app.
- `Dual-Engine Execution Model`: **PASS** — live Blue path ignores Data Loader/Loss; training uses worker-built module from graph fragment; success still auto-loads Gold; Unfreeze/Freeze unchanged in spirit.
- `Manual Granular Freeze Policy`: **PASS** — train auto-load remains explicit completion; Gold on data-loader path is passthrough-only (cannot arm).
- `Shape Integrity & Legal Constraints`: **PASS** — connect rules refuse data-loader on processing-driven pins and enforce Loss pin roles; group hubs dedupe shared live+loader pins; equal-count / missing-feed / loss / arm gates refuse Start with clear messages.
- `Zero Audio Allocations / Non-Blocking Audio Thread`: **PASS** — train IPC, corpus packaging, config I/O, and path analysis off the audio thread; Data Loader/Loss never enter live `processBlock` audible path.
- `Local vs cloud access`: **PASS** — same generalized package for both destinations; local remains account-free; cloud entitlement unchanged.

**Post-Design Re-Check**: **PASS**. Contracts keep train off the audio thread and VST-only. Adversarial/feature-matching losses remain **train-worker-only** (discriminators never loaded in the plug-in), matching 005. No unjustified constitution violations.

## Project Structure

### Documentation (this feature)

```text
specs/019-generalize-training-graph/
├── plan.md              # This file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   ├── generalized-train-ipc.md
│   ├── data-loader-graph-ui.md
│   ├── loss-nodes-and-stages.md
│   ├── training-config-library.md
│   └── train-panel-generalized-ux.md
├── checklists/
│   └── requirements.md
└── tasks.md             # /speckit-tasks (NOT this command)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── graph/           # Data Loader + Loss node types; connect gates; path/arm; group-of-one; remove TCN/Linear palette
├── dsp/             # Live engine: ignore Data Loader/Loss; remove or stop instantiating TCN/Linear for new inserts
├── train/           # Generalized package; active loader; config snapshot assemble; TrainCoordinator destination unchanged
├── library/         # Training Library pool reused; NEW TrainingConfigLibrary (user-level)
├── ui/              # TrainPanel: drop objective; active loader; HP + stages; config save/load; opacity on Train tab
├── PluginEditor.*   # handleTrainRun gates + package assembly
└── PluginProcessor.*  # project training-config snapshot; drop lastTrainObjective
Backend/
└── train_worker.py  # architecture-agnostic train from graph + loss schedule; retire objective branch
CloudService/
└── worker/train_runner.py  # materialize generalized request (same schema)
Tests/
├── (C++ graph/data-loader/loss/group/palette tests)
└── test_train_worker.py (stages, weighted losses, recipe-parity via configs)
OpenYourBox/Resources/examples/training/
└── graph templates + example training configs (T001)
```

**Structure Decision**: Extend the existing OpenYourBox + Backend + CloudService layout. Do not add a separate train executable. Prefer one generalized IPC schema consumed by local and cloud workers.

## Complexity Tracking

No constitution violations require justification. Train-only discriminators for adversarial/feature-matching losses remain worker-side (already established in 005).

## Phase 0: Research — Complete

Resolved in `research.md`: generalized IPC; Data Loader binding model; loss catalog + stages; path/arm semantics; cable visuals; config library storage; palette removals; example templates; no-regression via wiring+configs.

## Phase 1: Design — Complete

Delivered: `data-model.md`, five contracts, `quickstart.md`.

## Execution Notes

Suggested implementation order (for `/speckit-tasks`):

1. Graph: Data Loader + Loss node types, connection rules, path discovery, Train-tab opacity, group-of-one  
2. Train panel: remove objective; active loader picker; general HP + stage editor; Start gates  
3. Training config library + project snapshot persistence  
4. Worker: replace objective branches with loss-schedule trainer; keep recipe-parity helpers  
5. Cloud package/runner schema alignment  
6. Remove TCN/Linear from palette (and new-insert paths)  
7. Example templates + configs; tests + quickstart validation  

**Artifact map**

| Artifact | Path |
|----------|------|
| Research | `specs/019-generalize-training-graph/research.md` |
| Data model | `specs/019-generalize-training-graph/data-model.md` |
| Quickstart | `specs/019-generalize-training-graph/quickstart.md` |
| Contracts | `specs/019-generalize-training-graph/contracts/` |
