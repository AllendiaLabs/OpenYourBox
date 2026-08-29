# Contract: DDSP Effects Elements

## Purpose

Behavioral and parameter contract for Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, and ModDelay in live, freeze, and train paths. Semantics align with Magenta `ddsp/effects.py` at user-audible / parameter level.

## Shared rules

| Rule | Requirement |
|------|-------------|
| Placement | Same graph interactions as other processing elements (add/move/connect/delete) |
| Live edit | Numeric/bool property changes apply without stopping audio when Blue |
| Shape | Illegal cables refused with mismatch explanation |
| Dry mix | Where `add_dry` exists: `true` → wet+dry; `false` → wet only |
| Long IR | Any `reverb_length` ≥ 1 allowed; if length_ms &gt; live-safe threshold → non-blocking warning; no clamp/refuse for length alone |
| Randomize | Randomizes trainable tensors (IR, magnitudes) only; flags like `add_dry` unchanged unless project rules say otherwise |
| Persist | Type, properties, connections, weights survive save/reload |
| Freeze/Train | Legal graphs containing these types remain eligible |

## Type contracts

### Reverb (`reverb`)

**Properties**: `reverb_length`, `add_dry`, `ir_blend` (0 = internal … 1 = external), seed/weights as applicable.

**Ports**: required `audio` in; optional `ir` in; `audio` out.

| Condition | Behavior |
|-----------|----------|
| `ir` disconnected | Convolve dry with internal IR of `reverb_length` |
| `ir` connected, non-empty | Convolve with blend of internal and external IR per `ir_blend` |
| `ir` connected, empty/zero-length | Internal IR only; recoverable warning; audio continues |
| `add_dry` | Magenta dry-mask on IR + optional dry add on output |

### ExpDecayReverb (`exp_decay_reverb`)

**Properties**: `gain`, `decay`, `reverb_length`, `add_dry`.

**Behavior**: Build exponential-decay noise IR; FFT convolve; dry-add per flag.

### FilteredNoiseReverb (`filtered_noise_reverb`)

**Properties**: `n_frames`, `n_filter_banks`, `window_size`, `reverb_length`, `add_dry`.

**Magnitudes**: trainable `[n_frames × n_filter_banks]`; edit via dimensions + init/randomize (+ train); **no** cell painter required.

### FIRFilter (`fir_filter`)

**Properties**: `n_frames`, `n_filter_banks`, `window_size`.

**Magnitudes**: same grid rules as FilteredNoiseReverb; LTV-FIR / frequency-filter wet path; no `add_dry`.

### ModDelay (`mod_delay`)

**Properties**: `center_ms`, `depth_ms`, `gain`, `phase`, `add_dry`.

**Validation**: If modulation would imply negative delay (`center_ms - depth_ms < 0`), clamp or refuse with clear property message.

**Behavior**: Time-varying delay coloration; `depth_ms = 0` is static delay at center; `depth_ms > 0` audible modulation.

## Live-safe warning contract

- **Trigger**: Effect IR / reverb length in ms above threshold (default 1000 ms at current sample rate).
- **Presentation**: Non-blocking (InfoPanel and/or Parameters/property affordance); orange/warn styling consistent with RF warning.
- **Non-effects**: Must not stop audio, must not rewrite the user’s length value.

## Out of scope

- Compressor and other TODO effects
- Cell-by-cell magnitude painter
- TensorFlow Magenta runtime
