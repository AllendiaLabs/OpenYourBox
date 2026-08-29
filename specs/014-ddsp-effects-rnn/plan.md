# Implementation Plan: DDSP Effects & Recurrent Layers

**Branch**: `014-ddsp-effects-rnn` | **Date**: 2026-08-30 | **Spec**: `specs/014-ddsp-effects-rnn/spec.md`

**Input**: Feature specification from `specs/014-ddsp-effects-rnn/spec.md`

## Summary

Add five Magenta DDSP-style effect elements (Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, ModDelay) under an **Effects** factory category, plus single-layer **LSTM** and **RNN** under a **Neural / Sequence** category. Effects and recurrent nodes run in the Live (Blue) engine with LibTorch, persist through graph save/reload, and participate in freeze/train via Python `build_module` branches. Audible semantics follow Magenta `ddsp/effects.py` (ported to PyTorch/LibTorch, not TensorFlow). LSTM/RNN carry hidden state across live buffers, reset on structural/lifecycle events, use Activation/TCN-shared activation + gain in-cell, and infer input size from upstream channels.

## Technical Context

**Language/Version**: C++17 (VST / live engine / UI); Python 3.10+ (freeze/train workers)

**Primary Dependencies**: JUCE, Dear ImGui, imgui-node-editor, LibTorch (`torch::nn` / tensors), PyTorch (workers), existing `NodeGraph` / `LiveGraphEngine` / `NoiseSynthesizer` FFT helpers, Magenta DDSP effect semantics (Apache 2.0 reference)

**Storage**: Graph `ValueTree` / patch JSON for node type, ports, properties, trainable weights (IR, magnitude grids, recurrent weights); no new database. NOTICE attribution for Magenta/DDSP.

**Testing**: C++ `Tests/LiveGraphEngineTests.cpp` (+ focused DSP tests if needed); Python `Tests/test_freeze_worker.py`, `Tests/test_train_worker.py`; manual scenarios in `quickstart.md`

**Target Platform**: macOS VST3/AU (primary); Windows/Linux secondary if build allows

**Project Type**: Desktop audio plugin with embedded ImGui node editor + detached Python backend

**Performance Goals**: Live Blue latency budget (&lt; 7 ms @ 256 samples on reference i7) for default effect/recurrent configs; 60 FPS UI; freeze of legal subgraphs &lt; 2 s for small graphs; non-blocking warn when reverb IR length exceeds live-safe threshold (InfoPanel-style)

**Constraints**: VST-only UI; zero audio-thread allocations; Shape Integrity on all new ports; manual freeze only; Magenta semantics without shipping TensorFlow; long `reverb_length` allowed with soft warn (no hard clamp for length alone); stay within or thoughtfully raise `maximumHistorySamples` so compile remains safe

**Scale/Scope**: 7 new `NodeType`s; Effects + Neural/Sequence palette grouping; Reverb optional IR input + blend; magnitude-grid dims + init/randomize (no cell painter); single-layer LSTM/RNN with bidirectional, bias, activation, gain; full freeze/train/randomize/persist parity

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Single Interface, Decoupled Compute**: Pass. Effects and recurrent layers are graph elements in the VST; freeze/train remain backend workers managed by the plugin. No standalone app or terminal workflow.
- **Dual-Engine Execution Model**: Pass. Blue live modules in `LiveGraphEngine`; Gold via existing TorchScript BlackBox after freeze. Randomize applies to trainable tensors on GUI thread with atomic swap.
- **Manual Granular Freeze Policy**: Pass. No auto-freeze. New types register in freeze/train `build_module` so Freeze Selection / Unfreeze continue to work when shapes are legal.
- **Shape Integrity & Legal Constraints**: Pass. Illegal cables refused with mismatch tooltips; LSTM/RNN input size inferred from upstream; magnitude/dimension illegality refused in property UI. Copyright modal unchanged.
- **Zero Audio-Thread Allocation Rule**: Pass. IR buffers, magnitude grids, recurrent weights, and hidden-state slots are prepared in compile/`prepare` on the message thread; audio only runs preallocated tensors. Hidden-state reset coincides with rebuild/prepare/randomize, not mid-buffer alloc.
- **Complexity Justification**: Pass. Seven types + category grouping are required by the clarified spec; custom LSTM/RNN cells are required for in-cell activation/gain parity with Activation/TCN (stock `nn.LSTM` cannot substitute).

**Post–Phase 1 re-check**: Still Pass. Contracts keep all mutations and warnings on the GUI/message path; IR blend and empty-IR fallback are recoverable (no host stop); soft performance warning mirrors InfoPanel RF pattern; Magenta attribution via NOTICE does not change licensing of core Apache 2.0 code.

## Project Structure

### Documentation (this feature)

```text
specs/014-ddsp-effects-rnn/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── element-palette-categories-contract.md
│   ├── ddsp-effects-contract.md
│   ├── recurrent-layers-contract.md
│   └── freeze-train-type-registry-contract.md
└── tasks.md                 # /speckit-tasks (not created here)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── graph/
│   ├── GraphTypes.h              # NodeType + predicates + chrome; activation/gain constants reuse
│   ├── NodeGraph.cpp / .h        # makeNode defaults, nodeTypeName/FromName, persist gate, shape rules
│   └── NodeRenderer.cpp / .h     # Categorized Factory palette; property rows; reverb length warn affordance
├── dsp/
│   ├── LiveGraphEngine.cpp / .h  # CompiledElement + process; hidden-state slots; IR/magnitude runtime
│   ├── NoiseSynthesizer.*        # Reuse / extend fftConvolve & amplitude→IR helpers where DDSP aligns
│   ├── DdspEffects.* (new)       # ExpDecay IR, LTV-FIR, ModDelay, Reverb blend helpers (optional split)
│   └── RecurrentLayers.* (new)   # Single-layer RNN/LSTM cells with in-cell activation + gain
├── ui/
│   └── InfoPanel.cpp             # Optional: surface live-safe IR length warning alongside RF warning
Backend/
├── freeze_worker.py              # build_module branches for 7 types
└── train_worker.py               # build_module (+ DAG path if multi-input Reverb)
NOTICE                            # Magenta DDSP Apache 2.0 attribution block
Tests/
├── LiveGraphEngineTests.cpp      # Live audio/feature behavior, state carry/reset, shape refuse
├── test_freeze_worker.py         # Script/export for new types
└── test_train_worker.py          # Trainable IR/magnitudes/recurrent weights
```

**Structure Decision**: Extend the existing OpenYourBox + Backend monorepo layout. Register seven new `NodeType` values through the same factory / serialize / live / freeze / train pipeline used by Activation, TCN, and Noise Synth. Prefer shared DSP helpers next to `NoiseSynthesizer` rather than a separate package. Palette gains category grouping without a new app surface.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Custom LSTM/RNN cells (not stock `nn.LSTM`/`nn.RNN`) | Spec requires in-cell activation replacement + Activation/TCN gain | Stock cells hard-code tanh/sigmoid gates and ignore gain |
| Categorized Factory palette | Spec FR-001 / FR-010 discovery (Effects vs Neural/Sequence) | Flat alphabetical list buries new types and fails SC-004 |
| Reverb dual IR + blend | Clarified FR-006 / FR-006b | Internal-only IR drops external-IR creative path; hard fail on empty IR stops audio |
| Soft live-safe IR warning (no length clamp) | FR-006a / SC-008 | Hard clamp/refuse contradicts clarifications; silent allow hides RT risk |
