# Contract: RAVE Graph UI

## Purpose

Live palette, shape rules, layouts, and fidelity control for RAVE-shaped graphs inside the existing node editor.

## Palette (additions)

| Label | Type | Trainable |
|-------|------|-----------|
| PQMF Analysis | `pqmfAnalysis` | no (fixed filters; not absorbed as “weights” unless later learned — v1 fixed bank) |
| PQMF Synthesis | `pqmfSynthesis` | no (fixed bank) |
| Rate Conv | `rateConv` | yes |
| Variational Bottleneck | `variationalBottleneck` | yes (encoder head) |
| Noise Synth | `noiseSynthesizer` | no (IR × noise; amplitude net is user-wired) |

PQMF nodes are not armable as learned weights in v1 (fixed analysis/synthesis). Rate conv, bottleneck parameters, TCN, Linear, Conv1D, PReLU remain armable per Phase 3. Noise Synth has no learned weights: users wire the acids-rave amplitude stack (strided Conv1D + LeakyReLU, then `mod_sigmoid(x − 5)`) themselves.

## Properties

- PQMF: `nBand` default 16 (legal set: powers of two used by layouts, min 2)
- Rate conv: `stride`, `direction` downsample|upsample, kernel, dilation, channels
- Bottleneck: `latentSize` default 128; **Fidelity** 0–100 always visible (live and after train)
- Noise: `noiseBands` default 5 (IR bins per output channel), `windowSize` default 64 (`prod([4,4,4])` in acids-rave v1). Input channels must be `k * noiseBands`; output channels are `k`. Temporal rate upsamples by `windowSize` (hop fold). No internal conv stack.

## Shape

- Refuse audio ↔ latent, audio ↔ multiband, mismatched `nBand`, mismatched temporal rate
- Tooltip states which field failed
- Mono layout on stereo host: illegal unless user adds explicit channel adapters (no silent downmix)

## Layouts

Menu: **Insert RAVE layout → Original | Latest continuous**, then **Mono | Stereo**.

- Original: PQMF 16, strided encoder `[4,4,4,2]`, bottleneck 128, decoder waveform × loudness + noise
- Latest continuous: PQMF 16, residual dilated stacks + same ratios, amplitude modulation, bottleneck 128, optional noise off unless user adds it

Layouts insert ordinary Blue nodes (editable, armable, freezable).

## Fidelity

- Control on bottleneck and on Gold RAVE BlackBox
- Always applies (including Gold), same class of live control as Knob/XY → FiLM
- Unfreeze copies current percent onto restored bottleneck

## Freeze

- Freeze of a RAVE subgraph uses freeze worker extended to emit encode/decode/forward when a bottleneck is present
- Otherwise existing single-`forward` freeze unchanged

## Implementation anchors

- `OpenYourBox/Source/graph/GraphTypes.h`, `NodeGraph.cpp`, `NodeRenderer.cpp`
- `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` (+ new modules: PQMF, RateConv, Bottleneck, NoiseSynth)
- Layout builders: `OpenYourBox/Source/graph/` (e.g. `RaveLayouts.cpp`)
