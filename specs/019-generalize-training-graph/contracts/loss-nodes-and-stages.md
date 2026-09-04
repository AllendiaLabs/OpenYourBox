# Contract: Loss Nodes and Stage Schedules

## Purpose

Define Loss palette elements, wiring, and Training Configuration stage schedules (with weights) so users can reproduce prior mapping and reconstruction (RAVE-like) recipes without architecture modes.

## Loss node

- Palette element **Loss** with selectable `loss_type`:
  - `mr_stft` — multiresolution STFT (default sizes matching prior mapping recipe)
  - `spectral_distance` — RAVE v1 spectral / AudioDistanceV1 behavior
  - `kl` — variational / bottleneck KL regularization
  - `adversarial` — train-only multi-scale discriminator loss
  - `feature_matching` — feature matching (typically paired with adversarial)
- Pins: **prediction** (index 0) and **target** (index 1).
- **No `weight` property on the box** — weights live only on the stage schedule.
- Training-only: ignored on live audible path.
- Loading legacy graphs MUST strip any persisted Loss `weight` property.

## Wiring rules

| Source | Allowed Loss pin |
|--------|------------------|
| Live / processing path | **prediction** only |
| Data Loader | **target** only |

- Refuse Data Loader → prediction and live → target at connect time.
- Start MUST refuse if no usable loss wiring exists.
- Start MUST refuse if a loss prediction is outside the active data-loader-reachable path (prediction must come from the train path, not the loader).

## Combining losses

### Single stage (default / empty schedule)

Worker runs one stage for `total_steps` with every validly wired loss at weight **1.0**.

### Multi-stage schedule

Training Configuration / Train panel stores ordered stages:

```json
{
  "stages": [
    {
      "name": "representation",
      "steps": 1000000,
      "losses": [
        { "loss_node_id": 21, "weight": 1.0 },
        { "loss_node_id": 22, "weight": 0.1 }
      ]
    },
    {
      "name": "quality",
      "steps": 1000000,
      "losses": [
        { "loss_node_id": 21, "weight": 1.0 },
        { "loss_node_id": 23, "weight": 1.0 },
        { "loss_node_id": 24, "weight": 10.0 }
      ]
    }
  ]
}
```

- Each stage loss entry **MUST** carry `weight` (Train UI edits it next to the loss checkbox).
- Each stage MAY include `freeze_element_ids` (armed element ids frozen for that stage).
- Config persistence uses `losses: [{ loss_node_id, weight }]` and optional `freeze_element_ids` (legacy `loss_node_ids` arrays load with weight 1.0).
- Missing/unwired `loss_node_id` in a stage → refuse Start with clear message.
- Progress UI SHOULD show current stage name/index.

## Per-stage freeze UI

- Collapsed by default under each stage (`Freeze` tree).
- Tree lists **only armable/trainable** leaves (and ancestor groups that contain them).
- Disarmed armable leaves: checkbox **checked + disabled + gray**.
- Group check → all armable descendants; partial selection → **mixed** (minus) parent checkbox.
- Worker rebuilds trainable Adam param set at each stage boundary from `armed_element_ids ∖ freeze_element_ids`.
- Train-tab group chrome: if all armable leaves under a group are transparent, the group box is transparent too.

## Train-only adversarial machinery

- Discriminators exist only inside the worker for stages that include `adversarial` / `feature_matching`.
- Never loaded into the VST live engine or Gold artifact as separate user nodes.
- Related HP (`discriminator_lr`, `update_discriminator_every`, augmentations) live on Training Configuration general options.

## Capability parity (acceptance bar)

Users MUST be able to express:

1. **Mapping-style**: Data Loader input≠target materials + `mr_stft` (or equivalent) + mapping-like HP defaults via example config.
2. **Reconstruction-style**: same-data materials + spectral + KL (+ later adversarial/FM) with two-stage schedule and stage weights matching prior defaults via example config.

## Implementation anchors

- `OpenYourBox/Source/graph/*` — loss element; connect pin rules
- `OpenYourBox/Source/ui/TrainPanel.cpp` — stage editor with per-loss weight
- `OpenYourBox/Source/PluginEditor.cpp` — `loss_schedule` packaging
- `Backend/train_worker.py` — evaluate schedule; default weight 1.0
- `Tests/test_train_worker.py` / `Tests/GeneralizedTrainGraphTests.cpp`
