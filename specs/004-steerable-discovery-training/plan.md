# Implementation Plan: Steerable Discovery & Training (Phase 3)

**Branch**: `004-steerable-discovery-training` | **Date**: 2026-08-20 | **Spec**: `specs/004-steerable-discovery-training/spec.md`

**Input**: Feature specification from `specs/004-steerable-discovery-training/spec.md`

**Status**: Design + tasks complete. Path conventions verified against `OpenYourBox/Source/` (`graph/`, `dsp/`, `freeze/`, `train/`, `capture/`, `library/`, `ui/`) and `Backend/` / `Tests/`. Ready for `/speckit-implement`.

Next: `/speckit-implement` (or begin Phase 1 tasks in `tasks.md`).

Deliver Phase 3 steerable NAfx inside OpenYourBox:

- **Graph**: TCN FiLM port, residual, PReLU, **dilation growth** (Gⁿ, RONN-style UI + RF readout + presets 2/8/10, default G=2); arm only trainable params; Weights seed/path + browse.
- **Data**: Dual-instance Capture (input x/y, default bypass) **and** file import into a durable **Training Library** (select pairs for Train); copyright gate.
- **Train**: Non-blocking Python worker — Adam; MR-STFT `{32,128,512,2048}` / hops `{16,64,256,1024}`; ca=0; LR 1e-3→1e-4@80%→1e-5@95%; ~2500 steps; **always** RF-aware crops; segment ≈228308 (hidden; show window seconds); Run/Pause/Stop + live loss.
- **Result**: Auto-load Gold BlackBox for armed chain; Knob/XY stay Blue for free **c**; Unfreeze keeps trained weights until randomize/retrain.

Other armed architectures allowed; users can build steerable-nafx-equivalent TCN. ml_forge informs Train UX only (not shipped).

## Technical Context

**Language/Version**: C++17 (JUCE plug-in / live engine), Python 3 (freeze + train worker)

**Primary Dependencies**: JUCE 8, Dear ImGui, `imgui-node-editor`, LibTorch (live + TorchScript Gold), PyTorch + **auraloss** (or equivalent MR-STFT matching specified sizes) in train worker; `FreezeCoordinator` ChildProcess IPC pattern

**Storage**: JUCE `ValueTree` graph (topology, arm, Weights, FiLM/residual/PReLU/`dilationGrowth`); Training Library under plugin user data (pair audio + metadata index); copyright acknowledgment log; trained `.pt` artifacts; localhost pairing registry/IPC

**Testing**: CTest (`OpenYourBox*`) for graph/UI/runtime; Python `test_train_worker.py` for recipe/IPC; DAW manual scenarios in `quickstart.md`

**Target Platform**: Desktop AU/VST3, macOS first; same-machine dual instances

**Project Type**: Single desktop audio plug-in with embedded Python workers

**Performance Goals**: 60 FPS UI under train load; live/frozen latency unchanged (<7 ms / <5 ms @ 256); zero audio-thread allocations; train never blocks `processBlock`; loss UI ≥ ~1 Hz

**Constraints**: VST-only UI; master owns library/Train/auto-load; slave reduced capture UI; control sources never enter Gold; capture = instance inputs; default bypass; no max capture length; RF-aware crop always on; copyright before first Train

**Scale/Scope**: 2 paired instances; many library pairs (disk-limited); 1 active train job/master; ~tens of armed nodes; ~2500 steps default

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Single Interface, Decoupled Compute`: **PASS** — Library, Capture, Train, Weights inside plug-in; train in detached worker.
- `Dual-Engine Execution Model`: **PASS** — Blue live train snapshot → Gold TorchScript; Unfreeze → Blue with weights kept.
- `Manual Granular Freeze Policy`: **PASS** — Train auto-load is explicit user Train completion, not background auto-freeze; Freeze/Unfreeze remain; arm selects subgraph.
- `Shape Integrity & Legal Constraints`: **PASS** — FiLM pin validation; copyright modal + local log.
- `Zero Audio Allocations / Non-Blocking Audio Thread`: **PASS** — pairing/capture I/O/train IPC/prepare off audio thread; atomic Gold swap only.

**Post-Design Re-Check**: **PASS**. All contracts and research decisions align with gates. No unjustified violations.

## Project Structure

### Documentation (this feature)

```text
specs/004-steerable-discovery-training/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── instance-pairing-capture-contract.md
│   ├── train-worker-ipc.md
│   ├── steerable-graph-ui-contract.md
│   └── training-library-ui-contract.md
├── checklists/
│   └── requirements.md
└── tasks.md                    # /speckit-tasks (complete)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── graph/           # FiLM pin, residual, PReLU, dilationGrowth, arm, Weights
├── dsp/             # LiveGraphEngine, TCNModel — FiLM/residual/PReLU/growth^n; weight load
├── freeze/          # Existing FreezeCoordinator (pattern)
├── train/           # NEW TrainCoordinator
├── capture/         # NEW pairing + input-ring capture
├── library/         # NEW Training Library store + index (or under ui/ + persistence)
├── ui/              # Library panel, Capture ingest, Train panel, copyright, Weights
├── PluginEditor.*
└── PluginProcessor.*
Backend/
├── freeze_worker.py
└── train_worker.py  # NEW — recipe + TorchScript export
Tests/
├── (C++ graph/library/train coordinator tests)
└── test_train_worker.py
```

**Structure Decision**: Extend OpenYourBox single project. Mirror freeze IPC for train. Library is first-class persisted store; Capture is an ingest path. Knob/XY remain free-c sources outside Gold.

## Phase 0: Research — Complete

Resolved in `research.md`: pairing IPC; capture inputs + bypass; train ChildProcess + fixed recipe + specified MR-STFT; RF-aware crops + hidden segment length; library + file import; FiLM supersession; dilation growth UI; arm/Weights/Unfreeze-with-weights; ml_forge UX patterns only.

## Phase 1: Design — Complete

Delivered: `data-model.md`, four contracts (pairing, train IPC, steerable UI, **training library UI**), `quickstart.md`.

## Complexity Tracking

No constitution violations require justification.

## Execution Notes

Next: `/speckit-implement` using `tasks.md`. Suggested implementation order:

1. Graph: FiLM, residual, PReLU, dilationGrowth (+ UI readout/presets), arm, Weights  
2. Training Library persistence + v1 UI (list/detail/import/select/preview)  
3. Pairing + Capture ingest into library  
4. TrainCoordinator + `train_worker.py` (recipe, RF crops, progress)  
5. Auto-load Gold + Unfreeze weight preservation  
6. Copyright modal; tests + quickstart validation  

**Artifact map**
| Artifact | Path |
|----------|------|
| Spec | `spec.md` |
| Research | `research.md` |
| Data model | `data-model.md` |
| Pairing/capture | `contracts/instance-pairing-capture-contract.md` |
| Train IPC | `contracts/train-worker-ipc.md` |
| Graph/Train UI | `contracts/steerable-graph-ui-contract.md` |
| Library UI | `contracts/training-library-ui-contract.md` |
| Quickstart | `quickstart.md` |

**Implementation anchors**
- Graph: `GraphTypes.h`, `NodeGraph.*`, `NodeRenderer.*`
- Runtime: `dsp/LiveGraphEngine.*`, `dsp/TCNModel.*`, `PluginProcessor.*`
- Freeze → Train: `freeze/FreezeCoordinator.*` → `train/TrainCoordinator.*`
- Backend: `Backend/train_worker.py`
- Reference: `.ignore/steerable-nafx-main/steerable-nafx.ipynb`, `.ignore/ml_forge-main/ml_forge/ui/training.py` (patterns only)
