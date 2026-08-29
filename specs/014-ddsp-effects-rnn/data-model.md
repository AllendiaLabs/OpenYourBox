# Data Model: DDSP Effects & Recurrent Layers

## Overview

Adds seven graph `NodeType` values, palette category metadata, Reverb dual-IR blend state, magnitude-grid tensors, and recurrent hidden-state runtime slots. Persistence remains the existing graph `ValueTree` / freeze-train JSON fragment model.

## Entities

### Palette Category

| Field | Type | Notes |
|-------|------|-------|
| `id` | enum/string | `effects`, `neural_sequence`, plus existing default groups |
| `label` | string | UI section header (“Effects”, “Neural / Sequence”) |
| `nodeTypes[]` | NodeType | Ordered insertable types in that section |

**Rules**
- Effects: `reverb`, `expDecayReverb`, `filteredNoiseReverb`, `firFilter`, `modDelay`
- Neural / Sequence: `lstm`, `rnn`
- Categories affect Factory presentation only; persistence key remains `NodeType` string

---

### Effect Element (shared)

**Base fields**
| Field | Type | Notes |
|-------|------|-------|
| `type` | NodeType | One of five effect types |
| `add_dry` | bool | Default `true` where Magenta exposes it (FIRFilter has no dry-add) |
| `hasWeights` | bool | True when IR or magnitudes are trainable |
| ports | audio in → audio out | Shape passthrough for channels/hop unless noted |

**Chrome**: Helper / effects family (same visual family as Noise Synth / Activation helpers unless `isTrainableType` paints learned blue for weighted effects — prefer learned blue when `hasWeights`).

---

### Reverb

| Property | Type | Default | Validation |
|----------|------|---------|------------|
| `reverb_length` | int (samples) | 48000 | ≥ 1; soft warn if ms &gt; live-safe threshold |
| `add_dry` | bool | true | — |
| `ir_blend` | float 0–1 | 0.0 | Used when external IR connected; 0 = internal, 1 = external |
| `seed` | int | graph | IR init |

**Ports**
- Input `audio` (required)
- Input `ir` (optional): external impulse response time series
- Output `audio`

**Trainable / runtime**
| Field | Notes |
|-------|-------|
| `ir_internal` | Tensor length `reverb_length`; trainable; first tap masked dry per Magenta |
| `warning_empty_external_ir` | Recoverable; set when `ir` connected but empty/zero-length |

**Behavior**
- Disconnected `ir`: convolve with internal IR
- Connected non-empty: blend internal/external then convolve
- Connected empty: internal only + recoverable warning
- Wet = FFT convolve; output `wet + dry` if `add_dry`

**State transitions**
```
ir_port: disconnected → connected(nonempty) → blend active
                      → connected(empty) → fallback internal + warn
length edit → realloc IR (message thread) → optional soft warn
```

---

### ExpDecayReverb

| Property | Type | Default | Validation |
|----------|------|---------|------------|
| `gain` | float | Magenta-like usable default (e.g. mapped from init 2.0 via scale) | finite; UI range TBD to keep audible |
| `decay` | float | Magenta-like default | finite |
| `reverb_length` | int | 48000 | ≥ 1; soft live-safe warn |
| `add_dry` | bool | true | — |

**IR recipe**: `ir = scale(gain) * exp(-(2 + exp(decay)) * t) * noise`, `t ∈ [0,1]` over length (Magenta `_get_ir`).

**Ports**: audio in → audio out (no external IR).

---

### FilteredNoiseReverb

| Property | Type | Default | Validation |
|----------|------|---------|------------|
| `n_frames` | int | 1000 (or smaller live-friendly default if documented) | ≥ 1 |
| `n_filter_banks` | int | 16 | ≥ 1 |
| `window_size` | int | 257 | odd ≥ 1 per Magenta practice |
| `reverb_length` | int | 48000 | ≥ 1; soft warn |
| `add_dry` | bool | true | — |
| `seed` | int | graph | magnitude init |

**Trainable**: `magnitudes` `[n_frames, n_filter_banks]`.

**UI**: dimension editors + init/randomize; no cell painter.

**IR**: Magenta FilteredNoise synth over `reverb_length` samples.

---

### FIRFilter

| Property | Type | Default | Validation |
|----------|------|---------|------------|
| `n_frames` | int | sensible default (≥ 1) | ≥ 1 |
| `n_filter_banks` | int | 16 | ≥ 1 |
| `window_size` | int | 257 | odd ≥ 1 |
| `seed` | int | graph | — |

**Trainable**: `magnitudes` `[n_frames, n_filter_banks]`.

**Behavior**: Magenta `frequency_filter` LTV-FIR; no `add_dry`.

---

### ModDelay

| Property | Type | Default | Validation |
|----------|------|---------|------------|
| `center_ms` | float | 15.0 | ≥ 0 |
| `depth_ms` | float | 10.0 | ≥ 0; if `center_ms - depth_ms < 0`, clamp or refuse with property validation message |
| `gain` | float | Magenta-scaled default | finite |
| `phase` | float | 0.5 (mid after sigmoid) or editable control | typically [0,1] after scale |
| `add_dry` | bool | true | — |

**Runtime**: delay line sized for `(center_ms + depth_ms)` at current sample rate; preallocated in `prepare`.

**Ports**: audio in → audio out.

---

### Recurrent Element (LSTM / RNN shared)

| Property | Type | Default | Validation |
|----------|------|---------|------------|
| `hidden_size` | int | 16 (or project-sensible) | ≥ 1 |
| `bidirectional` | bool | false | — |
| `bias` | bool | true | — |
| `activation` | choice | Tanh (index of `ActivationType`) | 0–4 same as Activation/TCN |
| `negative_slope` | float | 0.01 | shown iff LeakyReLU; [0, 1] |
| `gain` | float | 1.0 | [0.1, 10] same as Activation/TCN |
| `seed` | int | graph | weight init |

**Not present**: `num_layers`, `input_size`.

**Ports**
- Input: feature/audio sequence; `input_size` inferred from upstream `channels`
- Output: same time length; `channels = hidden_size` or `2 * hidden_size` if bidirectional

**Trainable**: weight/bias tensors for one layer (forward + optional backward).

**Runtime state** (`LiveGraphRuntime`)
| Field | Notes |
|-------|-------|
| `h` / `c` | Hidden (and LSTM cell) state; carried across buffers |
| reset | On new `prepare`, reconnect/rebuild/freeze swap, or randomize/re-init of this node |

**In-cell nonlinearity**
- RNN: activation(+gain) replaces cell nonlinearity
- LSTM: activation(+gain) replaces primary candidate/`tanh` nonlinearity; gates stay sigmoid

**Shape rules**
- Missing upstream → cannot infer input size → connection/compile refused with explanation
- Channel mismatch after `hidden_size` / bidirectional edit → outgoing cables may invalidate (existing Shape Integrity)

---

### Live-Safe Performance Warning

| Field | Type | Notes |
|-------|------|-------|
| `threshold_ms` | float | Default 1000 (align InfoPanel RF) |
| `active` | bool | True when any effect IR length ms &gt; threshold |
| `message` | string | Non-blocking; does not mutate value |

---

### Persistence (ValueTree / JSON)

| Key | Presence |
|-----|----------|
| `type` | `reverb`, `exp_decay_reverb`, `filtered_noise_reverb`, `fir_filter`, `mod_delay`, `lstm`, `rnn` |
| `properties[]` | As tables above |
| weights / artifacts | Existing weight path / inline tensor provenance for IR, magnitudes, recurrent weights |
| connections | Standard; Reverb may have second inbound link to `ir` |

**Save/reload**: SC-006 — all seven types restore type, cables, and property values.

---

### Freeze / Train Module Mapping

| NodeType string | Module responsibility |
|-----------------|----------------------|
| `reverb` | Conv/FFT reverb module; optional IR input; blend param; trainable IR |
| `exp_decay_reverb` | Build IR from gain/decay/length; convolve |
| `filtered_noise_reverb` | Magnitudes → filtered-noise IR → convolve |
| `fir_filter` | LTV frequency filter from magnitudes |
| `mod_delay` | Variable delay from center/depth/gain/phase |
| `lstm` / `rnn` | Custom single-layer cell; bidirectional concat; scriptable forward |

Training arms weighted effects and recurrent types via `isTrainableType` / `armedForTraining` rules consistent with Linear/Conv.

## Relationships

```text
PaletteCategory 1──* EffectElement | RecurrentElement
Reverb 1──0..1 ExternalIR (via optional port)
FilteredNoiseReverb / FIRFilter 1──1 MagnitudeGrid (trainable tensor)
LSTM / RNN 1──1 HiddenStateSlot (runtime only; not persisted across sessions)
GraphDocument 1──* all node entities (existing)
FreezeJob / TrainJob *──* new types via JSON type registry
```
