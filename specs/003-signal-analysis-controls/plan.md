# Implementation Plan: Signal Analysis & Expressive Input Controls (Phase 2.2)

**Branch**: `003-signal-analysis-controls` | **Date**: 2026-08-19 | **Spec**: `specs/003-signal-analysis-controls/spec.md`

**Input**: Feature specification from `specs/003-signal-analysis-controls/spec.md`

## Summary

Extend the existing JUCE + Dear ImGui plug-in with Phase 2.2 capabilities: per-element analysis views (transfer, frequency, phase) showing **chain** and **element-only** response families for **all channel/feature dimensions**; inline **Gain** on Activation and TCN nodes; and new graph source elements **Knob Input** and **XY Trackpad** supplying runtime conditioning **c** (steerable NAfx g(x, c) model), routable directly or — primarily — through **Merge** alongside audio **x**. Heavy analysis stays off the audio thread; Gold BlackBox nodes retain analysis parity with Blue live nodes.

## Technical Context

**Language/Version**: C++17 for plug-in/runtime code, Python 3 for the embedded freeze worker

**Primary Dependencies**: JUCE 8, Dear ImGui, `imgui-node-editor`, LibTorch, embedded Python worker for freeze compilation

**Storage**: JUCE `ValueTree` graph document for topology, node properties, viewport, seeds, conditioning element state (Knob/XY values, positions), and per-node analysis view preferences; local TorchScript artifacts for frozen nodes

**Testing**: CTest console apps (`OpenYourBoxProcessorTests`, `OpenYourBoxLiveGraphTests`, `OpenYourBoxTests`) plus Python freeze-worker tests

**Target Platform**: Desktop audio plug-in on macOS first (AU/VST3 via CMake)

**Project Type**: Single desktop audio plug-in with embedded worker and native tests

**Performance Goals**: 60 FPS UI during audio processing; frozen latency < 5 ms and live latency < 7 ms at 256-sample buffers; static analysis refresh < 500 ms perceived latency; zero audible glitches on Gain/Knob/XY/Merge edits

**Constraints**: No standalone app; zero audio-thread allocations; no audio-thread blocking; extend Phase 2 graph model (no replacement); Knob/XY excluded from freeze subgraph compilation

**Scale/Scope**: One editor surface; tens of nodes per session; analysis plots up to full channel count at selected tensor shape (mono through 64+ latent features); three analysis views × two curve families × N dimensions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Single Interface, Decoupled Compute`: PASS. All capabilities remain inside the plug-in editor.
- `Dual-Engine Execution Model`: PASS. Analysis parity for Blue and Gold preserved.
- `Manual Granular Freeze Policy`: PASS. Freeze policy unchanged; conditioning UI nodes excluded from compiled subgraphs.
- `Shape Integrity & Legal Constraints`: PASS. Port-type validation extended for conditioning vs audio; existing shape gates retained.
- `Zero Audio Allocations / Non-Blocking Audio Thread`: PASS. Analysis, plot generation, and conditioning state commits occur on message/background threads with atomic runtime handoff.

**Post-Design Re-Check**: PASS. Design keeps runtime/UI separation, persists conditioning elements in graph state, and documents Merge + dual-curve analysis without constitution violations.

## Project Structure

### Documentation (this feature)

```text
specs/003-signal-analysis-controls/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── analysis-runtime-contract.md
│   └── graph-control-ui-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── graph/          # GraphTypes, NodeGraph, NodeRenderer — new node types, pin kinds, persistence
├── dsp/            # LiveGraphEngine — gain, conditioning merge, dual analysis snapshots
├── ui/             # InfoPanel — chain/element plots, N-channel overlay, transfer marker
├── PluginEditor.*  # Orchestration, analysis requests, invalidation
└── PluginProcessor.* # Live capture publication, revision tokens
Tests/
├── LiveGraphEngineTests.cpp
└── ProcessorIntegrationTests.cpp
```

**Structure Decision**: Extend existing single-project layout. New `NodeType` values (`knobInput`, `xyTrackpad`), pin/signal kind metadata, analysis snapshot types, and InfoPanel rendering are the primary touchpoints.

## Phase 0: Research Focus

- Background-thread dual snapshot architecture (chain vs element-only) without audio-thread work
- N-channel/feature-dimension plot series layout and legend strategy
- Conditioning source elements + Merge extension (audio + scalar conditioning lanes)
- Gold BlackBox analysis at compiled boundary with same view semantics as Blue
- Transfer live marker sampling from published live capture during playback

## Phase 1: Design Focus

- Graph document entities for Knob Input, XY Trackpad, extended Merge, Gain property, analysis preferences
- Analysis/runtime and graph-control UI contracts
- End-to-end quickstart covering stereo and multi-channel paths, Merge routing, Gold parity, state recall

## Complexity Tracking

No constitution violations require justification.

## Execution Notes

Implementation follows `tasks.md`. Shared plumbing first, then US1 (analysis), US2 (Gain), US3 (Knob Input), US4 (XY Trackpad).

**Artifact map**
- Data model: `data-model.md`
- Contracts: `contracts/analysis-runtime-contract.md`, `contracts/graph-control-ui-contract.md`
- Validation: `quickstart.md`

**Implementation anchors**
- Graph: `GraphTypes.h`, `NodeGraph.*`, `NodeRenderer.*`
- Analysis UI: `ui/InfoPanel.*`, `PluginEditor.*`
- Runtime/analysis: `dsp/LiveGraphEngine.*`, `PluginProcessor.*`

**Cross-link verification (T001)**
- `data-model.md` entities (Gain, Knob/XY, Merge lanes, AnalysisSnapshot) map to `GraphTypes.h` and `LiveGraphEngine.h`.
- `contracts/analysis-runtime-contract.md` maps to `LiveGraphEngine` snapshot production, `PluginProcessor` revision/live-capture, and `InfoPanel` consumption.
- `contracts/graph-control-ui-contract.md` maps to `NodeGraph` validation/persistence and `NodeRenderer` menu/controls.
- `quickstart.md` scenarios 1–9 are the independent tests listed on US1–US4 in `tasks.md`.
