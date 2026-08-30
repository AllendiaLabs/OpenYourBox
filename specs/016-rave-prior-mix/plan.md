# Implementation Plan: RAVE Prior Mix & Insert Catalog

**Branch**: `016-rave-prior-mix` | **Date**: 2026-08-30 | **Spec**: `specs/016-rave-prior-mix/spec.md`

**Input**: Feature specification from `specs/016-rave-prior-mix/spec.md`

## Summary

Give every **RAVE-capable** Gold box (learned OYB RAVE and TorchScript Load with encode/decode) a continuous **`priorMix`** element property (0 = full forward, 1 = full prior), replace the latent **input** pin with **bias** and **scale** pins, sample the effective latent after mix→bias/scale, publish that tensor on **latent out**, and skip `encode` at full prior. Separately, rebuild **right-click Pin Add / Link Insert** menus so they list the **current shared Factory catalog** (no stale subset) and an **expandable User Library** hierarchy (006/011 insert semantics). Old latent-in project migration is out of scope.

## Technical Context

**Language/Version**: C++17 (VST / live engine / UI); Python 3.10+ only if OYB RAVE export must expose `(μ, σ)` for faithful scale on learned Gold (see research)

**Primary Dependencies**: JUCE, Dear ImGui, imgui-node-editor, LibTorch; existing `NodeGraph` / `LiveGraphEngine` / `TorchScriptBlackBox` / `VariationalBottleneck`; `UserBoxLibrary` / `UserBoxLibraryPanel` for insert trees

**Storage**: Graph `ValueTree` / preset for `priorMix` (and mirrored `GraphNode` field); bias/scale as ordinary pin connections; no new on-disk formats for catalog

**Testing**: `Tests/LiveGraphEngineTests.cpp` (prior mix, encoder skip, bias/scale defaults, latent-out identity); optional UI/catalog unit or smoke coverage for palette vs context menu parity; manual scenarios in `quickstart.md`

**Target Platform**: macOS VST3/AU (primary)

**Project Type**: Desktop audio plugin with embedded ImGui node editor

**Performance Goals**: Full prior skips encode for that box; Gold buffer budget still &lt; 5 ms @ 256 samples on reference i7; zero audio-thread allocations (preallocate noise / work tensors at prepare); 60 FPS UI

**Constraints**: VST-only UI; Shape Integrity on bias/scale vs effective latent width; frozen Gold may edit `priorMix` like `fidelity`; Blue modular prior-mix rebuild out of scope; right-click menus only for catalog FR-012–015; no latent-in migration

**Scale/Scope**: One runtime path change for RAVE-capable `blackBox` execute + pin surface; one UI catalog change for Pin/Link context menus; property mirror pattern cloned from fidelity/gain

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Single Interface, Decoupled Compute**: Pass. Prior mix, bias/scale, and right-click insert stay inside the VST; no terminal workflows.
- **Dual-Engine Execution Model**: Pass. Changes target Gold BlackBox (train autoload + external load). Blue modular encoder/bottleneck/decoder prior-mix surface remains out of scope.
- **Manual Granular Freeze Policy**: Pass. No auto-freeze. Unfreeze policy for external load unchanged.
- **Shape Integrity & Legal Constraints**: Pass. Bias/scale connections use existing latent-domain shape checks; illegal cables refused with tooltips.
- **Zero Audio-Thread Allocation Rule**: Pass. Pin morph, property edits, and menu building on GUI thread; sampling noise and work buffers allocated at prepare/compile; audio only reads published state.
- **Complexity Justification**: Pass. Spec requires distributional mix + pin replacement on Gold/load boxes and a stale context-menu catalog fix; extending BlackBox execute + shared palette source is the minimal constitution-aligned path.

**Post–Phase 1 re-check**: Still Pass. Contracts keep prepare/GUI work off the audio thread; encoder skip reduces work at full prior; catalog fix reuses Factory + UserBoxLibrary without a second product surface.

## Project Structure

### Documentation (this feature)

```text
specs/016-rave-prior-mix/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── rave-prior-mix-runtime-contract.md
│   ├── bias-scale-pin-surface-contract.md
│   └── right-click-insert-catalog-contract.md
└── tasks.md                 # /speckit-tasks (not created here)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── graph/
│   ├── GraphTypes.h                 # priorMix field/helpers; bias/scale pin labels; isBiasPin/isScalePin
│   ├── NodeGraph.cpp / .h           # pin surface for RAVE Gold + external load; priorMix property; serialize
│   ├── FactoryPalette.h / .cpp      # (new or extracted) shared Factory catalog for left panel + context menus
│   ├── NodeRenderer.cpp / .h        # context Add/Insert menus; priorMix property row; remove latent-in UI assumptions
│   └── UserBoxLibraryPanel.cpp / .h # reuse/extract expandable tree helpers for ImGui menus
├── dsp/
│   ├── LiveGraphEngine.cpp / .h     # blackBox execute: prior mix, skip encode, bias/scale, sample, latent out
│   ├── TorchScriptBlackBox.cpp / .h # optional encode_distribution / (μ,σ) probe for OYB + external
│   └── VariationalBottleneck.cpp/.h # helpers for softplus std + sample if Gold path needs them
Backend/                             # only if OYB export must expose (μ, σ) for learned Gold scale fidelity
Tests/
└── LiveGraphEngineTests.cpp         # prior mix / bias-scale / latent-out / encoder-skip cases
```

**Structure Decision**: Extend existing OpenYourBox VST layout. Extract a **single Factory palette catalog** shared by `renderPalette` and Pin/Link context menus; reuse **UserBoxLibrary** insert APIs for menu placement. Do not add a parallel Blue prior-mix graph.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Live reparameterization on RAVE Gold path (today μ-only) | Spec requires spread interpolation, scale×spread, and sampled latent out | Keeping μ-only cannot honor scale or “sample once after mix” |
| Shared Factory catalog extraction | Context menus and left Factory both hard-code the same arrays; menus must stay current | Duplicating yet another list would stay stale; auto-including every `NodeType` would show non-palette types |
| Bias/scale pins + compile indices | Spec replaces latent-in drive | Remapping latent-in as “bias only” breaks RAVE-VST semantics |
