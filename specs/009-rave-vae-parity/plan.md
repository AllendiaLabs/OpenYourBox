# Implementation Plan: RAVE Variational Bottleneck Parity

**Branch**: `009-rave-vae-parity` | **Date**: 2026-08-27 | **Spec**: `specs/009-rave-vae-parity/spec.md`

**Input**: Feature specification from `specs/009-rave-vae-parity/spec.md`

**Status**: Implementation complete.

Refine the existing RAVE variational bottleneck (spec 005) to match **acids-ircam RAVE** rules:

- **VAE**: softplus variance parameterization; worker samples during stage 1; **live path always μ-only**
- **Head**: causal grouped conv `groups=2` (mean branch + variance branch), default **kernel 5**, user-configurable `kernel_size`
- **Compactness**: **98/2 train/val split** (seed 42, val cap 1000); PCA on validation **μ** at stage-1 end; singular-value cumulative fidelity; buffers on Gold + Unfreeze
- **Breaking change**: remove legacy 1×1 full-width mean/logvar heads (no migration)

## Technical Context

**Language/Version**: C++17 (JUCE plug-in / LibTorch live engine), Python 3 (train + freeze workers)

**Primary Dependencies**: LibTorch C++ API, PyTorch in workers; existing RAVE stack from spec 005 (PQMF, rate conv, TrainCoordinator, Gold export)

**Storage**: Graph ValueTree (`kernel_size`, `latent_size`, `fidelity`, `compactnessReady`, PCA tensor paths on node); TorchScript `.pt` with `latent_mean`, `latent_pca`, `cumulative_variance` buffers; train IPC events (`compactness.ready`)

**Testing**: CTest (`VariationalBottleneckTests`, `LiveGraphEngineTests` updates); Python (`test_train_worker.py` split/PCA/sampling); integration per `quickstart.md`

**Target Platform**: Desktop AU/VST3 (macOS first); same detached worker model as Phase 3

**Project Type**: Single desktop audio plug-in with embedded Python workers

**Performance Goals**: Zero audio-thread allocations (constitution); live bottleneck adds only preallocated k−1 causal history; PCA runs off audio thread at stage-1 end in worker

**Constraints**: VST-only UI; no acids-rave CLI; causal operators; live μ-only always; fidelity inactive until PCA ready; no legacy artifact compatibility

**Scale/Scope**: One variational bottleneck per armed RAVE path; touch ~8–12 source files + tests; no new palette node type

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status |
|-----------|--------|
| Single Interface, Decoupled Compute | **PASS** — parity changes inside existing train/freeze workers and live engine |
| Dual-Engine (Blue live + Gold TorchScript) | **PASS** — same split; compactness buffers on both paths after train/Unfreeze |
| Manual Freeze / Unfreeze | **PASS** — Unfreeze copies compactness to Blue bottleneck |
| Shape Integrity | **PASS** — encoder channels ÷ 2 gate for grouped head |
| Zero Audio Allocations | **PASS** — grouped conv uses preallocated history ring (GUI prepare); no PCA on audio thread |
| Latency 5 ms Gold / 7 ms live | **PASS (inherited exception)** — RAVE delay exception from spec 005 still applies; k=5 adds negligible history vs PQMF/stride |

**Post-Design Re-Check**: **PASS**. No new unjustified violations.

## Project Structure

### Documentation (this feature)

```text
specs/009-rave-vae-parity/
├── plan.md              # This file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   ├── variational-bottleneck-contract.md
│   └── compactness-pca-contract.md
├── checklists/
│   └── requirements.md
└── tasks.md             # /speckit-tasks (not this command)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── dsp/
│   ├── VariationalBottleneck.h/cpp   # softplus, grouped causal conv, fidelity
│   └── LiveGraphEngine.cpp           # weight layout, kernel_size, μ-only live
├── graph/
│   ├── GraphTypes.h                  # default bottleneck kernel = 5
│   ├── NodeGraph.cpp                 # property, shape gate, Unfreeze compactness copy
│   └── RaveLayouts.cpp               # layout defaults kernel_size=5
Backend/
├── train_worker.py                   # VariationalBottleneckLayer, split, validation PCA
└── freeze_worker.py                  # mirror bottleneck for export
Tests/
├── VariationalBottleneckTests.cpp    # new
├── LiveGraphEngineTests.cpp          # extend
└── test_train_worker.py              # split, PCA, sampling mode
```

**Structure Decision**: Extend spec 005 files in place. No new executables or node types.

## Phase 0: Research — Complete

Resolved in `research.md`: softplus+logvar mapping; grouped head layout; live vs worker sampling; 98/2 split; validation PCA timing; weight tensor layout; breaking change scope.

## Phase 1: Design — Complete

Delivered: `data-model.md`, two contracts, `quickstart.md`.

## Complexity Tracking

> No new constitution violations beyond spec 005 RAVE delay exception.

| Change | Risk | Mitigation |
|--------|------|------------|
| Weight tensor layout change (1×1 → grouped k×5) | Breaks old graphs/checkpoints | Explicit breaking change (FR-013); retrain required |
| Live always μ-only while worker samples | User expects train noise in DAW | Documented in Train panel; matches reference eval |
| Validation split reduces stage-1 train data by 2% | Slightly smaller train set | Matches acids-rave; seed 42 reproducible |

## Execution Notes

Suggested order for `/speckit-tasks`:

1. **Python reference layer** — `VariationalBottleneckLayer` (softplus, grouped causal conv, train/eval forward) + unit tests  
2. **Train split + validation PCA** — `split_dataset` 98/2, stage-1 train-only loop, eval validation pass at stage-1 end, checkpoint `compactness.ready`  
3. **C++ live bottleneck** — mirror math in `VariationalBottleneck`, grouped weights, `kernel_size` property, causal history  
4. **Graph + layouts** — property default 5, shape gate (channels % 2), layout insert, remove legacy 1×1 path  
5. **Gold / Unfreeze / freeze** — export buffers; copy to Blue on Unfreeze; fidelity inactive until ready  
6. **Integration tests + quickstart** — smoke train, fidelity sweep, Unfreeze buffer parity  

Next: `/speckit-tasks`
