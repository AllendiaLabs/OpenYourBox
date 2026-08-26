# Data Model: RAVE Variational Bottleneck Parity

## Overview

Extends spec **005** `variationalBottleneck`, compactness, and reconstruction train entities. No new `NodeType`.

## Entities

### Variational Bottleneck (extended)

**Properties**
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `latent_size` | int | 128 | Must be even (split across μ/var groups) |
| `kernel_size` | int | **5** | ≥ 1; causal padding |
| `fidelity` | float 0–100 | 99 | UI control; inactive until compactness ready |
| `seed` | int | graph | Weight init |

**Ports**
- Input `features`: domain `multiband` or latent-rate feature tensor; **`channels` must be even**
- Output `z`: domain `latent`; `channels = latent_size`

**Runtime state (live)**
| Field | Type | Notes |
|-------|------|-------|
| `weights[0]` | Tensor | Grouped conv weight (μ+var branches) |
| `causalHistory` | Tensor | Preallocated `[1, C, k−1]` off audio thread |
| `compactnessReady` | bool | PCA buffers valid |
| `latentMean` | Tensor `[latent]` | From validation PCA |
| `latentPca` | Tensor `[latent, latent]` | Row basis |
| `cumulativeVariance` | Tensor `[latent]` | Singular-value cumulative ratios |

**Behavior**
- Live / Gold encode: **μ only** → optional fidelity crop → output `z`
- Worker train forward: sample `z = μ + σ⊙ε` (stage 1 only)
- Worker eval / validation pass: **μ only**

**Validation**
- `input_channels % 2 == 0` else connection/arm refused
- `latent_size % 2 == 0` else property refused
- `kernel_size >= 1`

### Compactness Basis

**Fields**
- `latentMean`: float vector length `latent_size`
- `latentPca`: matrix `[latent_size × latent_size]` (rows = components)
- `cumulativeSingularRatio`: float vector length `latent_size`
- `ready`: bool
- `validationSegmentCount`: int (audit)
- `computedAtStage`: `"representation_end"` (v1 only)

**Lifecycle**
1. Created at stage-1 end from validation μ
2. Embedded in TorchScript on export / auto-load
3. Copied to Blue bottleneck on Unfreeze
4. Cleared on weight randomize or failed PCA

### Reconstruction Corpus Split

**Fields**
- `trainPaths[]`: ~98% of selected corpus
- `valPaths[]`: ~2%, seed 42, `len(valPaths) <= 1000`
- `splitSeed`: 42 (fixed v1)
- `splitRatio`: 98 (train percent)

**Rules**
- Split computed once per reconstruction job from selected library paths
- Stage 1 minibatches sample **trainPaths** only
- Validation pass iterates **valPaths** entirely (segmented like train)

### Fidelity Control (extended)

**States**
| State | UI | Effective behavior |
|-------|-----|-------------------|
| `not_ready` | Disabled / label "Compactness not ready" | Full width (keep = latent_size) |
| `ready` | Active 0–100% | Crop per cumulative singular ratio |

**Rules**
- Mid-stage-1 checkpoints: `not_ready`
- Post-success auto-load: `ready` if PCA succeeded
- Insufficient val data: `not_ready`, train still succeeds

### RAVE Gold Artifact (extended buffers)

Unchanged ports/methods from spec 005. Additional TorchScript buffers:
- `latent_mean`, `latent_pca`, `cumulative_variance`
- Method `encode` applies μ-only then fidelity when `compactness.ready`

**Metadata** (`blackbox_metadata.compactness`)
```json
{
  "ready": true,
  "validation_segments": 42,
  "latent_size": 128
}
```

### Graph Node persistence (ValueTree)

Extended keys on `variational_bottleneck` nodes:
- `kernel_size` (default 5)
- `compactnessReady`
- Optional serialized PCA tensors or artifact-relative paths (implementation choice in tasks; must round-trip Unfreeze)

## Relationships

```text
Reconstruction Job
  ├── Corpus Split (train/val)
  ├── Variational Bottleneck (armed graph)
  └── on stage-1 end → Compactness Basis → Gold Artifact
                                        └→ Unfreeze → Blue Bottleneck state
```

## State Transitions

### Bottleneck compactness

```text
[untrained] compactnessReady=false
    → train stage-1 running → still false (checkpoints)
    → stage-1 end PCA ok → true (Gold + optional Blue after load)
    → Unfreeze → true (copied)
    → randomize weights → false
```

### Fidelity control

```text
not_ready → (PCA success) → ready
ready → (randomize / new train) → not_ready
```

## Validation Rules Summary

| Rule | Error surface |
|------|----------------|
| Encoder features odd channel count | Wire/arm tooltip |
| `latent_size` odd | Property edit refused |
| Val split empty but train ok | PCA fallback, status message |
| Legacy 1×1 checkpoint load | Load failure message, retrain |
