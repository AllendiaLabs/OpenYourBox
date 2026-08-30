# Implementation Plan: RAVE Architecture & Training

**Branch**: `005-rave-architecture-training` | **Date**: 2026-08-25 | **Spec**: `specs/005-rave-architecture-training/spec.md`

**Input**: Feature specification from `specs/005-rave-architecture-training/spec.md`

**Status**: Design complete. Ready for `/speckit-tasks`.

Next: `/speckit-tasks` (then `/speckit-implement`).

Deliver RAVE **inside the Phase 3 Train / Library / Capture shell** (unification):

- **Graph**: PQMF analysis/synthesis, causal rate-changing conv, variational bottleneck, noise synth; domains audio/multiband/latent; original + latest-continuous **layouts** (mono|stereo at insert); fidelity always-on like FiLM-on-Gold.
- **Data**: Same library; tags `pair`/`unpaired`; Capture **Pair|Single**; reconstruction uses pair **x and y**; mapping **errors** on unpaired selection.
- **Train**: Same panel + worker; `objective` mapping|reconstruction (last-used per instance); reconstruction two-stage 1e6+1e6, encoder freeze, train-only GAN; hear-while-training checkpoints; success auto-load only after both stages.
- **Result**: Gold with `forward`/`encode`/`decode`; Unfreeze keeps weights + fidelity.

Mapping recipe and Pair capture remain Phase 3.

## Technical Context

**Language/Version**: C++17 (JUCE plug-in / live engine), Python 3 (freeze + train worker)

**Primary Dependencies**: JUCE 8, Dear ImGui, imgui-node-editor, LibTorch (live + TorchScript), PyTorch in train/freeze workers; auraloss remains for mapping; reconstruction spectral distance implemented in-worker (RAVE `AudioDistanceV1` equivalent) plus train-only discriminators

**Storage**: ValueTree graph (new node types, fidelity, last-used objective); Training Library user-data (pair + clip files, tags); copyright log (unchanged); `.pt` artifacts with compactness buffers

**Testing**: CTest for shape/domain/rate-conv/PQMF invertibility; Python tests for reconstruction recipe stages, corpus flatten (x+y), mapping unpaired rejection; DAW scenarios in `quickstart.md`

**Target Platform**: Desktop AU/VST3, macOS first; same-machine Pair capture; Single capture needs one instance

**Project Type**: Single desktop audio plug-in with embedded Python workers

**Performance Goals**: 60 FPS UI under train load; zero audio-thread allocations; train never blocks `processBlock`; loss UI ≥ ~1 Hz; **TCN/effect** graphs keep <7 ms live / <5 ms Gold @ 256; **RAVE** graphs display causal delay (may exceed those rows — Complexity Tracking)

**Constraints**: VST-only UI; one Train panel; causal live+train; no acids-rave CLI; copyright before first Train; mixed SR blocked; reconstruction channels match graph; Stop ≠ success auto-load

**Scale/Scope**: One train job/master; reconstruction up to 2e6 steps (user may Stop); library mixed pairs+clips (disk-limited); layouts ~tens of nodes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Single Interface, Decoupled Compute`: **PASS** — no RAVE CLI/app; same detached worker.
- `Dual-Engine Execution Model`: **PASS** — Blue RAVE elements; Gold TorchScript with methods; Unfreeze → Blue + weights.
- `Manual Granular Freeze Policy`: **PASS** — Train auto-load is explicit completion; Freeze/Unfreeze remain; optional checkpoint load is user action.
- `Shape Integrity & Legal Constraints`: **PASS** — domain/rate/nBand gates; copyright reused.
- `Zero Audio Allocations / Non-Blocking Audio Thread`: **PASS** — streaming leftover rings prepared off audio thread; train IPC off audio thread.
- `Latency <5 ms Gold / <7 ms live`: **JUSTIFIED EXCEPTION** for RAVE PQMF+stride graphs only — see Complexity Tracking. Non-RAVE graphs unchanged.

**Post-Design Re-Check**: **PASS** with the RAVE delay exception documented. No other unjustified violations.

## Project Structure

### Documentation (this feature)

```text
specs/005-rave-architecture-training/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── rave-graph-ui-contract.md
│   ├── unified-train-ipc.md
│   ├── library-capture-extension.md
│   └── rave-gold-runtime-contract.md
├── checklists/
│   └── requirements.md
└── tasks.md                    # /speckit-tasks (not this command)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── graph/           # Node types, domains, layouts, Gold ports, fidelity
├── dsp/             # PQMF, RateConv, Bottleneck, NoiseSynth, streaming state
├── freeze/          # Export encode/decode/forward when bottleneck present
├── train/           # TrainCoordinator: objective, stages, reconstruction gate
├── capture/         # Capture kind Pair | Single
├── library/         # kind pair|clip, tags, selection rules
├── ui/              # Train objective, Library filter/warn, Capture kind, layouts
├── PluginEditor.*
└── PluginProcessor.*  # persist last-used objective
Backend/
├── freeze_worker.py
└── train_worker.py    # objective mapping | reconstruction
Tests/
├── LiveGraphEngineTests.cpp (domains, PQMF invert, causal rate conv)
└── test_train_worker.py (stages, flatten x+y, mapping unpaired)
```

**Structure Decision**: Extend the existing OpenYourBox + Backend layout. Do not add a RAVE-only executable.

## Phase 0: Research — Complete

Resolved in `research.md`: unified objective IPC; domains/rate; node set; causal streaming; PQMF; v2 two-stage recipe; Gold methods; library tags; capture kind; delay exception; reconstruction graph gate; NOTICE.

## Phase 1: Design — Complete

Delivered: `data-model.md`, four contracts, `quickstart.md`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| RAVE Gold/live delay may exceed constitution 5 ms / 7 ms @ 256 | Official RAVE uses 16-band PQMF and stride product 2048; causal padding delay is inherent | Non-causal/offline RAVE rejected by spec; shrinking to raspberry-only layouts would not match original/latest continuous layouts |

Mitigation: delay readout; preallocated streaming; 5/7 ms still required for non-rate-reducing graphs.

## Execution Notes

Suggested implementation order (for `/speckit-tasks`):

1. Shape domains + `rateConv` + PQMF live invertibility  
2. Bottleneck + noise + layouts (random weights audible)  
3. Library kind/tags + Capture Single + Train objective persist + gates  
4. Worker reconstruction stages + IPC + freeze multi-method export  
5. Gold forward/encode/decode + fidelity always-on + Unfreeze  
6. NOTICE; tests; quickstart 1–9 (smoke steps for long recipe)

**Artifact map**

| Artifact | Path |
|----------|------|
| Spec | `spec.md` |
| Research | `research.md` |
| Data model | `data-model.md` |
| Graph UI | `contracts/rave-graph-ui-contract.md` |
| Train IPC | `contracts/unified-train-ipc.md` |
| Library/Capture | `contracts/library-capture-extension.md` |
| Gold runtime | `contracts/rave-gold-runtime-contract.md` |
| Quickstart | `quickstart.md` |
