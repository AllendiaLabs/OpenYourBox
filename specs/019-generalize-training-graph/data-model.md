# Data Model: Generalize Training Graph

**Feature**: `019-generalize-training-graph`  
**Date**: 2026-09-04  
**Depends on**: `research.md`, `spec.md`

## Entity Overview

```text
TrainingLibrary (existing pool)
        │ binds materials
        ▼
DataLoader ──outputs──► DataLoaderCable ──► external source pins / sources
        │
        └── defines DataLoaderPath
                │
                ├── ArmedOnPath     → TrainableSubgraph (params updated)
                ├── PassthroughOnPath (disarmed or Gold)
                └── OffPath (unchanged)

Architecture outputs / intermediates ──► LossNode ◄── target feeds (DataLoader outputs)
        │
        └── referenced by LossStageSchedule inside TrainingConfiguration

TrainingConfiguration ──user library──► TrainingConfigLibrary
                     └──project───────► ProjectTrainConfigSnapshot
```

---

## DataLoader

| Field | Type | Notes |
|-------|------|-------|
| `id` | ElementId | Graph node id |
| `type` | `"dataLoader"` | Palette insert |
| `output_count` | int ≥ 1 | Controllable |
| `outputs[]` | DataLoaderOutput | Renamable pins |
| Training-only | bool | Ignored by live audible path |

### DataLoaderOutput

| Field | Type | Notes |
|-------|------|-------|
| `pin_index` | int | 0..N-1 |
| `label` | string | User-renamable (e.g. input, target, cond) |
| `binding` | TrainingMaterialBinding | Per-output ordered list or constant utility |

### TrainingMaterialBinding

| Field | Type | Notes |
|-------|------|-------|
| `kind` | `audio_list` \| `constant_scalar` \| (extensible) | |
| `entries[]` | LibraryRef \| path | Ordered examples for `audio_list` |
| `scalar_value` | float | For constant copied across examples |
| `example_count` | int | Derived; used in equal-count gate |

**Validation**:
- At Start (active loader only): all **connected** outputs MUST share equal `example_count`.
- Unconnected outputs ignored for the gate.
- Constant/scalar utility produces count matching peer outputs after user applies equalization utilities as needed.
- Mixed sample rates across audio entries remain blocked (existing library rule).

**Utilities** (user-initiated):
- Copy/repeat entries on a shorter output to match another’s count.
- Assign constant/scalar copied for all examples (Knob/XY feeds).

---

## DataLoaderCable

| Field | Type | Notes |
|-------|------|-------|
| `source` | DataLoader output pin | |
| `destination` | External/inference input pin or source node feed | |
| `visual` | Distinct color; RMS = N/A | |
| Coexistence | May share destination with one live cable | Live vs train selection by runtime |

**Connection rules**:
- ALLOW: empty pin; pin fed by Audio In or group-input hub; Loss **target**.
- REFUSE: pin driven by upstream processing; Loss **prediction**; Audio Out; another Data Loader.
- Live playback ignores these cables.

---

## ActiveDataLoaderSelection

| Field | Type | Notes |
|-------|------|-------|
| `active_data_loader_id` | ElementId \| null | Set in Train panel |
| Default | Sole Data Loader on canvas → that id | |
| Multi + null | Start refused | Clear message |

---

## LossNode

| Field | Type | Notes |
|-------|------|-------|
| `id` | ElementId | |
| `type` | `"loss"` | Or typed subtypes; see catalog |
| `loss_type` | enum | `mr_stft` \| `spectral_distance` \| `kl` \| `adversarial` \| `feature_matching` |
| `properties` | typed map | FFT sizes, windows, etc. — **no `weight`** |
| Pins | prediction (0) / target (1) | Live→prediction only; Data Loader→target only |
| Training-only | bool | Ignored live |

**Validation at Start**:
- At least one validly wired loss required (FR-010).
- Loss prediction pin MUST be reachable from the active data-loader path (else refuse).
- Stage schedule references MUST resolve to existing wired losses.

---

## LossStageSchedule

| Field | Type | Notes |
|-------|------|-------|
| `stages[]` | LossStage | Ordered |
| Empty / absent | Implicit single stage: all wired losses at weight **1.0** for `total_steps` | |

### LossStage

| Field | Type | Notes |
|-------|------|-------|
| `name` | string optional | e.g. representation, quality |
| `steps` | int > 0 | Stage duration |
| `losses[]` | `{ loss_node_id, weight }` | Active set; **weight required** (defaults to 1.0 if omitted) |
| `freeze_element_ids[]` | ElementId[] | Optional; armed layers listed here get `requires_grad=False` for this stage only |

---

## TrainableSubgraph / Passthrough / OffPath

| Concept | Definition | Param updates |
|---------|------------|---------------|
| On path | Reachable from active Data Loader outputs via training cables + downstream processing | — |
| Armed on path | On path ∧ `armedForTraining` ∧ has trainable params ∧ not Gold | Yes |
| Passthrough | On path ∧ (¬armed ∨ Gold) | No (forward only) |
| Off path | Not reachable | No |

**Defaults / rules**:
- New processing elements: `armedForTraining = true`.
- Gold: cannot arm; always passthrough on path.
- Start requires ≥1 armed on-path trainable element.

**Train-tab visualization**:
- Armed on path → normal opacity.
- Passthrough / off path → slightly transparent.

---

## TrainingConfiguration

Named bundle of train settings (not architecture identity, not audio corpus).

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | User-facing |
| `optimizer` | string | e.g. adam |
| `device` | string | auto\|cpu\|mps\|cuda |
| `total_steps` | int | Used when schedule empty; else sum of stage steps may drive run length |
| `learning_rate` / schedules | floats | General HP surface |
| `segment_length` | int | Windowing |
| `batch_size` | int | |
| `checkpoint_interval` | int | |
| `export_checkpoints` | bool | Hear-while-training |
| `rf_aware_crop` | bool | |
| `loss_stage_schedule` | LossStageSchedule | |
| Adversarial helpers | disc period, phase mangle, dequant, disc LR, … | Exposed as general options when adversarial losses used |
| KL helpers | beta, warmup | May live on `kl` loss properties and/or config |
| Forward-compat | unknown fields ignored; missing → defaults + warn | |

### TrainingConfigLibrary (user-level)

| Field | Type | Notes |
|-------|------|-------|
| `entries[]` | `{ id, name, config, updated_at }` | Under plugin user data |
| Ops | list, save, load, rename, delete | Cross-project |

### ProjectTrainConfigSnapshot

| Field | Type | Notes |
|-------|------|-------|
| `config` | TrainingConfiguration | Stored with patch/session |
| Ops | save/keep with project; restore on load | |

**Removed**: `lastTrainObjective` / `TrainObjective` persistence.

---

## TrainRequest (logical package)

Assembled at Start for local and cloud (see `contracts/generalized-train-ipc.md`).

| Field | Notes |
|-------|-------|
| `operation` | `train_graph` |
| `active_data_loader_id` | |
| `armed_element_ids` | Armed on-path only (backprop set) |
| `graph_fragment` | Includes Data Loader + Loss + processing on path (and required helpers) |
| `data_loader_bindings` | Resolved paths / constants per output |
| `loss_schedule` | Stages |
| `train_options` | HP surface |
| `capture_set` | Deprecated for new runs; materials come from bindings (worker may still accept flattened file lists inside bindings) |

---

## Group (delta)

| Rule | Change |
|------|--------|
| Min members | **1** (was 2) |
| Audio I/O | Still excluded |
| Hubs | Dedupe by `(kind, memberPinId)` — shared live + Data Loader on one pin → **one** input hub |

---

## Palette removals

| Type | Status |
|------|--------|
| `tcn` | Not offered for new insert |
| `linear` | Not offered for new insert |
| Equivalents | Stacked `conv1d`; `conv1d` with stride=kernel=dilation=1 |

---

## Example artifacts (product content)

| Artifact | Contents |
|----------|----------|
| Mapping-style example graph | Effect-style chain + Data Loader + `mr_stft` loss wiring |
| Reconstruction-style example graph | Encoder/decoder-style + Data Loader + spectral/KL/(stage2) adv+FM losses |
| Example training configs | HP + stage schedules matching prior recipe defaults |

Presented as **examples/templates**, not Train modes.

---

## State Transitions

### Start training gates (ordered, informative failures)

1. Copyright acknowledged  
2. Active Data Loader resolved (sole / picker / refuse)  
3. Connected outputs equal-count  
4. Required external sources on path have data-loader feeds  
5. ≥1 valid loss wiring on path  
6. Stage schedule refs valid (if present)  
7. ≥1 armed on-path trainable element  
8. Destination-specific cloud entitlement (unchanged)

### Run lifecycle

`idle → running (stages…) → success|stopped|failure`  
Success → absorb snapshot → Gold (neutral label). Live audio uninterrupted throughout.
