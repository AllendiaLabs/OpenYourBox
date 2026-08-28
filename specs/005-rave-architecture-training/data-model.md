# Data Model: RAVE Architecture & Training

## Overview

This feature extends Phase 3 graph, library, capture, and train documents. Mapping entities remain as in `specs/004-steerable-discovery-training/data-model.md`. New/extended entities below.

## Entities

### Signal Shape (extended)

**Fields**
- `channels`: int
- `domain`: `audio` | `multiband` | `latent`
- `temporalRate`: rational or int hop product (1 at host audio)

**Validation**
- Connection legal iff domain, temporalRate, and channel compatibility match
- Host I/O: domain `audio`, temporalRate 1, stereo or graph-declared width

### Graph Node (RAVE types)

**New `NodeType` values**
- `pqmfAnalysis`, `pqmfSynthesis`
- `rateConv` — `stride` ≥ 1, `direction`: `downsample` | `upsample`, `kernelSize`, `dilation`, `channels`
- `variationalBottleneck` — `latentSize` (default 128), `fidelityPercent` 0–100 (default 99)
- `noiseSynthesizer` — `noiseBands` (default 5), `windowSize` (default 64). No learned weights. Input `[B, dataSize * noiseBands, frames]` → output `[B, dataSize, frames * windowSize]`.

**Extended fields on bottleneck / RAVE Gold**
- `fidelityPercent`: float 0–100
- `compactnessReady`: bool
- `informativeDimCount`: int (from PCA, optional)

**Ports**
- PQMF analysis: audio in → multiband out (`channels = nBand * audioChannels`)
- PQMF synthesis: inverse
- Bottleneck: multiband or feature in → latent out (`channels = latentSize`)
- Gold RAVE: audio in, audio out, latent out (encode), latent in (decode)

**Validation**
- Analysis/synthesis `nBand` must match if both present on a train path
- `rateConv` live is causal only
- Reconstruction Train requires armed bottleneck + decode path to audio

### RAVE Layout

**Fields**
- `id`: `original` | `latest_continuous`
- `channelWidth`: `1` | `2` (chosen at insert)
- `nodes[]`, `connections[]` — ordinary graph document

**Rules**
- Insert is a graph mutation (user-editable afterward)
- No silent host downmix; illegal host vs graph width is a shape error unless user wires adapters

### Training Library Entry (extended)

**Fields** (in addition to Phase 3 pair fields)
- `kind`: `pair` | `clip`
- `tags[]`: includes system `pair` or `unpaired`; optional user tags
- For `clip`: `audioPath` (single file); `xPath`/`yPath` unused or empty
- For `pair`: `xPath`, `yPath` as today
- `source`: `capture` | `import`
- `captureKind`: `pair` | `single` | null (imports)

**Rules**
- Delete removes owned files
- Mixed SR across **selected** set → Train blocked
- Reconstruction: selected pairs expand to two clips (x and y); clips used as-is
- Mapping: any selected `clip`/`unpaired` → error, Run blocked

### Training Library (extended)

**Fields**
- `entries[]` — mixed pair and clip
- `selectedIds[]`
- Tag index for filter

**UI rules**
- Warn and filter by current Train objective
- Same list+detail panel as Phase 3

### Capture Session (extended)

**Fields**
- `captureKind`: `pair` | `single`
- Pair: existing pairing + Clean/Processed
- Single: no peer; records this instance input to a `clip` entry

**Rules**
- Default bypass still true during capture
- No max duration
- Single does not assign slave UI

### Train Objective State

**Fields**
- `objective`: `mapping` | `reconstruction`
- Persisted **per plugin instance**; last-used restored; default `mapping`

### Training Job (extended)

**Fields**
- `objective`
- `stage`: `representation` | `quality` | `idle` (reconstruction only)
- `step`, `stageSteps`, `totalSteps` (sum of stages when reconstruction)
- `loss` plus optional `loss_kl`, `loss_adv`, `loss_fm`
- `checkpointPath` (optional hear-while-training)
- Mapping: unchanged ~2500-step recipe

**Rules**
- One active job per master
- Snapshot of armed subgraph at Run
- Reconstruction success swap only after both stages; Stop ≠ success
- Optional checkpoint load does not end job

### Compactness Basis

**Fields** (in artifact + bottleneck)
- `latentMean[latentSize]`
- `latentPca[latentSize][latentSize]` (or V from SVD)
- `cumulativeVariance[latentSize]`
- Applied at inference with `fidelityPercent` → keep `r_f` dims, fill rest with prior noise, project back

**Rules**
- Computed during/after representation validation; if too few batches, `compactnessReady = false`, fidelity is identity

### Gold BlackBox (RAVE)

**Fields**
- `origin`: `trainAutoload` | `manualFreeze`
- `methods`: `forward`, `encode`, `decode`
- `fidelityPercent` (live control, not frozen weights)
- `sourceSubgraph` for Unfreeze
- Weights path as Phase 3

**Rules**
- Control sources never absorbed
- Unfreeze restores Blue nodes + trained weights + current fidelity on bottleneck
- Optional mid-run checkpoint load uses same Gold prepare/swap path as mapping checkpoints

## Relationships

```text
PluginInstance --owns--> TrainObjective (last-used)
PluginInstance --owns--> TrainingLibrary
CaptureSession --kind pair--> Pairing + SamplePair entry
CaptureSession --kind single--> Clip entry
LibraryEntry --tags--> system+user
TrainJob --reads--> selected entries + armed RAVE Graph
TrainJob --writes--> compactness + Gold artifact
Gold RAVE --controls--> fidelityPercent (always-on)
```

## State Transitions

### Reconstruction job

`idle → representation → quality → success (auto-load) | stopped | failed`

Pause/resume allowed in both stages. Stop from either → `stopped` (no success auto-load).

### Capture

`idle → capturing (pair|single) → library append → idle`
