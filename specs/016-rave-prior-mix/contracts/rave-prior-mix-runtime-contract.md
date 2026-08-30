# Contract: RAVE Prior-Mix Runtime

**Feature**: `016-rave-prior-mix`  
**Applies to**: `LiveGraphEngine` BlackBox execute path when kernel `hasEncodeDecode()` and node is RAVE-capable Gold (`trainAutoload` or `externalLoad` with latent surface)

## Inputs

| Name | Source | Notes |
|------|--------|-------|
| Audio (main) | Upstream audio pin | Used only when `priorMix < 1` (encode runs) |
| `priorMix` | `RuntimeControlState` / compiled element | `[0, 1]` |
| `fidelityPercent` + compactness | Existing | When ready, applied to encoder distribution before mix |
| Bias tensor | Bias pin or constant 0 | Shape vs effective latent |
| Scale tensor | Scale pin or constant 1 | Shape vs effective latent |
| `ε` | Preallocated noise buffer | Filled without audio-thread heap growth |

## Processing order (normative)

1. If `priorMix >= 1 − ε_full`: **skip encode**; `(μ_e, σ_e) ← (0, 1)`.
2. Else: `encode` → `(μ_e, σ_e)` (see distribution contract below); apply fidelity/compactness to the encoder-side distribution per existing RAVE rules.
3. `α ← priorMix`; `μ ← (1−α)μ_e`; `σ ← (1−α)σ_e + α`.
4. `μ ← μ + bias`; `σ ← σ ⊙ scale`.
5. `z ← μ + σ ⊙ ε`.
6. Store `z` in `latentOutputs[element]`.
7. `audio ← decode(z)`.

## Distribution acquisition

| Source | Behavior |
|--------|----------|
| OYB-trained / freeze with `(μ, σ)` available | Use softplus-std convention (`σ = softplus(·) + softplusEpsilon`) |
| External / encode returns single latent only | `μ_e ← encode(·)`, `σ_e ← 1` |

## Encoder skip observability

At `priorMix = 1` with fixed bias/scale, changing only the audio input MUST NOT change `z` or audio out (within numerical noise). Encode MUST not be invoked for that element on that buffer.

## Non-goals

- Blue modular variational bottleneck prior-mix surface
- Host-only APVTS parameter for `priorMix`
- Latent-in migration
