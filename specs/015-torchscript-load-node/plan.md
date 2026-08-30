# Implementation Plan: TorchScript Checkpoint Loader Node

**Branch**: `015-torchscript-load-node` | **Date**: 2026-08-30 | **Spec**: `specs/015-torchscript-load-node/spec.md`

**Input**: Feature specification from `specs/015-torchscript-load-node/spec.md`

## Summary

Add a Factory palette **TorchScript Load** element that points at an external TorchScript checkpoint on disk, loads it off the audio thread into the existing frozen BlackBox registry (`TorchScriptBlackBoxFactory`), and runs it as opaque Gold processing. Reuse the current BlackBox execute path (forward / encode→decode / optional conditioning / fidelity). Extend `BlackBoxOrigin` with `externalLoad` so Unfreeze is disabled and empty/error/passthrough policies from the clarifications apply. On load: detect encode/decode via `find_method`, conditioning via the existing probe, compactness via artifact attrs; infer channel counts and expose editable overrides; persist path + overrides; re-prepare the registry on session/preset restore.

## Technical Context

**Language/Version**: C++17 (VST / live engine / UI); Python 3.10+ unchanged for this feature (no new worker types required for v1)

**Primary Dependencies**: JUCE, Dear ImGui, imgui-node-editor, LibTorch (`torch::jit::load`, existing `TorchScriptBlackBoxFactory` / `FrozenBlackBoxKernel`), existing `NodeGraph` / `LiveGraphEngine` / `PluginProcessor` artifact registry, `WeightLoader` / `FileChooser` patterns

**Storage**: Graph `ValueTree` / patch / preset persistence for `artifactPath` / `weightsPath`, `blackBoxOrigin`, channel override properties, fidelity / compactness fields; path reference only (no embed-in-project in v1); in-memory `publishedFrozenArtifacts` registry keyed by absolute path

**Testing**: C++ `Tests/LiveGraphEngineTests.cpp` (+ focused BlackBox / load tests); manual scenarios in `quickstart.md` with a known-good RAVE (or fixture) `.pt`

**Target Platform**: macOS VST3/AU (primary); Windows/Linux secondary if build allows

**Project Type**: Desktop audio plugin with embedded ImGui node editor + detached Python backend (backend unused for external load itself)

**Performance Goals**: Load and prepare off audio thread; atomic swap into live graph; after ready, Gold latency budget (&lt; 5 ms @ 256 samples on reference i7); 60 FPS UI during load; no audio-thread I/O or `jit::load`

**Constraints**: VST-only UI; zero audio-thread allocations; Shape Integrity on inferred/override channels; no Unfreeze to modular Blue for `externalLoad`; empty → dry passthrough; error with no prior model → silence; retain prior model during failed/in-progress reload until success or clear

**Scale/Scope**: One Factory element (reuses `NodeType::blackBox` + new origin); load/prepare/reload plumbing; pin surface morph (latent / control); channel override properties; property-panel browse + status; restore-time registry rehydrate

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Single Interface, Decoupled Compute**: Pass. Users pick and run checkpoints entirely inside the VST; no terminal or standalone loader. Python backend is not required for external load (optional later for conversion helpers).
- **Dual-Engine Execution Model**: Pass. Loaded checkpoints run as Gold BlackBox via LibTorch TorchScript; Blue modular graph remains unchanged. Element presents as Gold when ready.
- **Manual Granular Freeze Policy**: Pass. This feature does not add auto-freeze. External loads are not Unfreezeable into Blue (no OpenYourBox source subgraph). Freeze Selection on other Blue nodes remains unchanged.
- **Shape Integrity & Legal Constraints**: Pass. Illegal cables refused using inferred/override channel counts with existing mismatch feedback. Copyright modal unchanged (user-supplied files; no train button gate change).
- **Zero Audio-Thread Allocation Rule**: Pass. `jit::load`, probes, registry publish, and pin rebuilds happen on message/background threads; audio only uses an already-published factory/kernel via atomic pointer swap.
- **Complexity Justification**: Pass. Extending BlackBox origin + prepare/reload is required by the clarified external-load workflows; a separate live Blue module would duplicate the frozen engine and violate Dual-Engine.

**Post–Phase 1 re-check**: Still Pass. Contracts keep load/prepare/pin morph on the GUI/message path; dry-passthrough vs silence policies are recoverable (no host stop); encode/decode and Control pins follow existing Gold conventions without new audio-thread work.

## Project Structure

### Documentation (this feature)

```text
specs/015-torchscript-load-node/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── factory-palette-torchscript-load-contract.md
│   ├── external-checkpoint-load-contract.md
│   ├── pin-surface-and-channels-contract.md
│   └── persist-and-restore-contract.md
└── tasks.md                 # /speckit-tasks (not created here)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── graph/
│   ├── GraphTypes.h              # BlackBoxOrigin::externalLoad; optional channel-override fields on GraphNode
│   ├── NodeGraph.cpp / .h        # makeNode path for palette external load; pin morph; serialize overrides;
│   │                             # disable Unfreeze for externalLoad; setExternalCheckpoint / clear
│   └── NodeRenderer.cpp / .h     # Factory palette entry; path browse/clear; status; override + fidelity rows
├── dsp/
│   ├── TorchScriptBlackBox.cpp/.h  # Reuse load/probe; optional latent-width helper if needed
│   └── LiveGraphEngine.cpp/.h      # Empty dry-passthrough; error silence; retain prior factory on failed reload
├── PluginProcessor.cpp / .h      # prepareExternalArtifact; resolve; restore-time re-prepare registry
├── PluginEditor.cpp              # Wire browse → prepare (not WeightLoader-only path)
└── dsp/WeightLoader.*            # Optional: validate file exists / extension; prepare owns jit load
Tests/
├── LiveGraphEngineTests.cpp      # Passthrough / silence / encode-decode / conditioning / fidelity gates
└── (optional) Graph serialize tests for origin + overrides
```

**Structure Decision**: Extend the existing OpenYourBox VST layout. Do **not** add a Python worker type for v1. Prefer reusing `NodeType::blackBox` + `BlackBoxOrigin::externalLoad` and the published-artifact registry over a parallel live module.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| New `BlackBoxOrigin::externalLoad` + pin morph | Spec requires Factory placement, no Unfreeze, and dynamic latent/Control ports | Reusing train/freeze origins alone would enable Unfreeze or wrong labels; static pins would show dead Control/latent on plain models |
| Restore-time registry re-prepare | Spec FR-007 / SC-005; registry is in-memory only today | Path-only restore leaves Gold nodes silent after relaunch |
| Channel override properties | Clarified FR-011 | Auto-infer alone fails on odd exports; user-only fields break RAVE one-click load |
